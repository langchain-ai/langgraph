import asyncio
import functools
from typing import Callable, Optional, TypeVar

import anyio
from aiokafka import AIOKafkaConsumer
from langchain_core.runnables import RunnableConfig
from typing_extensions import ParamSpec

from langgraph.pregel import Pregel
from langgraph.pregel.types import StateSnapshot
from langgraph.scheduler.kafka.executor import KafkaExecutor
from langgraph.scheduler.kafka.orchestrator import KafkaOrchestrator
from langgraph.scheduler.kafka.types import MessageToOrchestrator, Topics

C = ParamSpec("C")
R = TypeVar("R")


def timeout(delay: int):
    def decorator(func: Callable[C, R]) -> Callable[C, R]:
        @functools.wraps(func)
        async def new_func(*args: C.args, **kwargs: C.kwargs) -> R:
            async with asyncio.timeout(delay):
                return await func(*args, **kwargs)

        return new_func

    return decorator


@timeout(20)
async def drain_topics(
    topics: Topics,
    graph: Pregel,
    config: RunnableConfig,
    *,
    until: Callable[[StateSnapshot], bool],
    debug: bool = False,
) -> tuple[list[MessageToOrchestrator], list[MessageToOrchestrator]]:
    scope: Optional[anyio.CancelScope] = None
    orch_msgs = []
    exec_msgs = []
    errors = []

    async def orchestrator() -> None:
        async with KafkaOrchestrator(graph, topics) as orch:
            async for msgs in orch:
                orch_msgs.extend(msgs)
                if debug:
                    print("\n---\norch", len(msgs), msgs)

    async def executor() -> None:
        async with KafkaExecutor(graph, topics) as exec:
            async for msgs in exec:
                exec_msgs.extend(msgs)
                if debug:
                    print("\n---\nexec", len(msgs), msgs)

    async def error_consumer() -> None:
        async with AIOKafkaConsumer(topics.error) as consumer:
            async for msg in consumer:
                errors.append(msg)
                if scope:
                    scope.cancel()

    async def poller(expected_next: tuple[str, ...]) -> None:
        while True:
            await asyncio.sleep(0.5)
            state = await graph.aget_state(config)
            if until(state):
                break
        if scope:
            scope.cancel()

    # start error consumer and poller
    error_task = asyncio.create_task(error_consumer(), name="error_consumer")
    poller_task = asyncio.create_task(poller(()), name="poller")

    # run the orchestrator and executor until break_when
    async with anyio.create_task_group() as tg:
        scope = tg.cancel_scope
        tg.start_soon(orchestrator, name="orchestrator")
        tg.start_soon(executor, name="executor")

    # cancel error consumer and poller
    error_task.cancel()
    poller_task.cancel()

    try:
        await asyncio.gather(error_task, poller_task)
    except asyncio.CancelledError:
        pass

    # check no errors
    assert not errors, errors

    return orch_msgs, exec_msgs
