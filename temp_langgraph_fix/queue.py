import asyncio
import os

import structlog
from langsmith import env as ls_env

from langgraph_runtime_inmem import database, ops

logger = structlog.stdlib.get_logger(__name__)

WORKERS: set[asyncio.Task] = set()

SHUTDOWN_GRACE_PERIOD_SECS = 5


def get_num_workers():
    return len(WORKERS)


async def queue():
    # Time and tide and asynchronous queues wait for no mortal,
    # As threads of our processes dance in delicate harmony,
    # Woven into the cosmic fabric of the server's eternal loom.
    # Imports delayed, like quantum particles, appearing only when observed.
    from langgraph_api import config, graph, webhook, worker

    concurrency = config.N_JOBS_PER_WORKER
    loop = asyncio.get_running_loop()
    last_stats_secs: int | None = None
    last_sweep_secs: int | None = None
    semaphore = asyncio.Semaphore(concurrency)
    WEBHOOKS: set[asyncio.Task] = set()
    enable_blocking = os.getenv("LANGGRAPH_ALLOW_BLOCKING", "false").lower() == "true"
    # raise exceptions when a blocking call is detected inside an async function
    if enable_blocking:
        bb = None
        await logger.awarning(
            "Heads up: You've set --allow-blocking, which allows synchronous blocking I/O operations."
            " Be aware that blocking code in one run may tie up the shared event loop"
            " and slow down ALL other server operations. For best performance, either convert blocking"
            " code to async patterns or set BG_JOB_ISOLATED_LOOPS=true in production"
            " to isolate each run in its own event loop."
        )
    else:
        bb = _enable_blockbuster()

    def cleanup(task: asyncio.Task):
        WORKERS.remove(task)
        semaphore.release()
        try:
            if task.cancelled():
                return
            exc = task.exception()
            if exc and not isinstance(exc, asyncio.CancelledError):
                logger.exception(
                    f"Background worker failed for task {task}", exc_info=exc
                )
                return
            result = task.result()
            if result and result["webhook"]:
                hook_task = loop.create_task(
                    webhook.call_webhook(result),
                    name=f"webhook-{result['run']['run_id']}",
                )
                WEBHOOKS.add(hook_task)
                hook_task.add_done_callback(WEBHOOKS.remove)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Background worker cleanup failed", exc_info=exc)

    await logger.ainfo(f"Starting {concurrency} background workers")
    try:
        run = None
        while True:
            try:
                # check if we need to sweep runs
                do_sweep = (
                    last_sweep_secs is None
                    or loop.time() - last_sweep_secs > config.BG_JOB_HEARTBEAT * 2
                )
                # check if we need to update stats
                if calc_stats := (
                    last_stats_secs is None
                    or loop.time() - last_stats_secs > config.STATS_INTERVAL_SECS
                ):
                    last_stats_secs = loop.time()
                    active = len(WORKERS)
                    await logger.ainfo(
                        "Worker stats",
                        max=concurrency,
                        available=concurrency - active,
                        active=active,
                    )
                # wait for semaphore to respect concurrency
                await semaphore.acquire()
                # skip the wait, if 1st time, or got a run last time
                wait = run is None and last_stats_secs is not None
                # try to get a run, handle it
                run = None
                async for run, attempt in ops.Runs.next(wait=wait, limit=1):
                    graph_id = (
                        run["kwargs"]
                        .get("config", {})
                        .get("configurable", {})
                        .get("graph_id")
                    )

                    if graph_id and graph.is_js_graph(graph_id):
                        task_name = f"js-run-{run['run_id']}-attempt-{attempt}"
                    else:
                        task_name = f"run-{run['run_id']}-attempt-{attempt}"
                    task = asyncio.create_task(
                        worker.worker(run, attempt, loop),
                        name=task_name,
                    )
                    task.add_done_callback(cleanup)
                    WORKERS.add(task)
                else:
                    semaphore.release()
                # run stats and sweep if needed
                if calc_stats or do_sweep:
                    async with database.connect() as conn:
                        # update stats if needed
                        if calc_stats:
                            stats = await ops.Runs.stats(conn)
                            await logger.ainfo("Queue stats", **stats)
                        # sweep runs if needed
                        if do_sweep:
                            last_sweep_secs = loop.time()
                            run_ids = await ops.Runs.sweep(conn)
                            logger.info("Swept runs", run_ids=run_ids)
            except Exception as exc:
                # keep trying to run the scheduler indefinitely
                logger.exception("Background worker scheduler failed", exc_info=exc)
                semaphore.release()
                await exit.aclose()
    finally:
        if bb:
            bb.deactivate()
        logger.info("Shutting down background workers")
        for task in WORKERS:
            task.cancel("Shutting down background workers.")
        for task in WEBHOOKS:
            task.cancel("Shutting down webhooks for background workers.")
        await asyncio.wait_for(
            asyncio.gather(*WORKERS, *WEBHOOKS, return_exceptions=True),
            SHUTDOWN_GRACE_PERIOD_SECS,
        )


def _enable_blockbuster():
    _patch_blocking_error()
    from blockbuster import BlockBuster

    ls_env.get_runtime_environment()  # this gets cached
    bb = BlockBuster(excluded_modules=[])
    for module, func in (
        # Note, we've cached this call in langsmith==0.3.21 so it shouldn't raise anyway
        # but we don't want to raise teh minbound just for that.
        ("langsmith/client.py", "_default_retry_config"),
        # Only triggers in python 3.11 for getting subgraphs
        # Will be unnecessary once we cache the assistant schemas
        ("langgraph/pregel/utils.py", "get_function_nonlocals"),
        ("importlib/metadata/__init__.py", "metadata"),
        ("importlib/metadata/__init__.py", "read_text"),
    ):
        bb.functions["io.TextIOWrapper.read"].can_block_in(module, func)

    bb.functions["os.path.abspath"].can_block_in("inspect.py", "getmodule")

    for module, func in (
        ("memory/__init__.py", "sync"),
        ("memory/__init__.py", "load"),
        ("memory/__init__.py", "dump"),
    ):
        bb.functions["io.TextIOWrapper.read"].can_block_in(module, func)
        bb.functions["io.TextIOWrapper.write"].can_block_in(module, func)
        bb.functions["io.BufferedWriter.write"].can_block_in(module, func)
        bb.functions["io.BufferedReader.read"].can_block_in(module, func)

        bb.functions["os.remove"].can_block_in(module, func)
        bb.functions["os.rename"].can_block_in(module, func)

    for module, func in (
        ("uvicorn/lifespan/on.py", "startup"),
        ("uvicorn/lifespan/on.py", "shutdown"),
        ("ansitowin32.py", "write_plain_text"),
        ("logging/__init__.py", "flush"),
        ("logging/__init__.py", "emit"),
    ):
        bb.functions["io.TextIOWrapper.write"].can_block_in(module, func)
        bb.functions["io.BufferedWriter.write"].can_block_in(module, func)
    # Support pdb
    bb.functions["builtins.input"].can_block_in("bdb.py", "trace_dispatch")
    bb.functions["builtins.input"].can_block_in("pdb.py", "user_line")
    to_disable = [
        "os.stat",
        # This is used by tiktoken for get_encoding_for_model
        # as well as importlib.metadata.
        "os.listdir",
        "os.remove",
        # If people are using threadpoolexecutor, etc. they'd be using this.
        "threading.Lock.acquire",
    ]

    for function in bb.functions:
        if function.startswith("os.path."):
            to_disable.append(function)
    for function in to_disable:
        func = bb.functions.pop(function, None)
        if func:
            func.deactivate()
    bb.activate()

    return bb


def _patch_blocking_error():
    from blockbuster.blockbuster import BlockingError

    original = BlockingError.__init__

    def init(self, func: str, *args, **kwargs):
        msg_ = func + (
            "\n\n"
            "Heads up! LangGraph dev identified a synchronous blocking call in your code. "
            "When running in an ASGI web server, blocking calls can degrade performance for everyone since they tie up the event loop.\n\n"
            "Here are your options to fix this:\n\n"
            "1. Best approach: Convert any blocking code to use async/await patterns\n"
            "   For example, use 'await aiohttp.get()' instead of 'requests.get()'\n\n"
            "2. Quick fix: Move blocking operations to a separate thread\n"
            "   Example: 'await asyncio.to_thread(your_blocking_function)'\n\n"
            "3. Override (if you can't change the code):\n"
            "   - For development: Run 'langgraph dev --allow-blocking'\n"
            "   - For deployment: Set 'BG_JOB_ISOLATED_LOOPS=true' environment variable\n\n"
            "These blocking operations can prevent health checks and slow down other runs in your deployment. "
            "Following these recommendations will help keep your LangGraph application running smoothly!"
        )
        original(self, msg_, *args, **kwargs)

    BlockingError.__init__ = init
