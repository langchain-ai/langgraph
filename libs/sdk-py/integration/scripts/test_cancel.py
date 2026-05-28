"""Exercise mid-run cancellation against the integration API.

Strategy: start a run on a fresh thread, capture the run id, then
cancel via the runs REST client while events are still flowing. The
projection iterator must terminate without hanging, no exception
should escape, and the thread's persisted status must reflect a
non-success terminal state.

The graph normally interrupts at `ask_human`; cancel must take effect
before or after that interrupt, and either way the run must end up in
a non-success state from the server's perspective.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any

from _common import (
    ASSISTANT_ID,
    check_api_reachable,
    header,
    make_async_client,
    make_sync_client,
)

_CANCEL_GRACE_SECONDS = 10.0


async def _cancel_after_first_event(
    runs_client: Any,
    thread_id: str,
    run_id_future: asyncio.Future[str],
) -> None:
    """Wait for the run id, briefly let events flow, then cancel."""
    run_id = await run_id_future
    # Allow a beat of events to flow so cancel hits mid-stream rather
    # than racing with the run.start handshake.
    await asyncio.sleep(0.1)
    with contextlib.suppress(Exception):
        await runs_client.cancel(thread_id, run_id, wait=False)


async def run_async() -> None:
    header("async mid-run cancel")
    threads, raw = make_async_client()
    # Cancel goes through the runs REST surface, not the stream proxy.
    from langgraph_sdk._async.http import HttpClient
    from langgraph_sdk._async.runs import RunsClient

    runs_client = RunsClient(HttpClient(raw))
    try:
        async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            run_id_future: asyncio.Future[str] = (
                asyncio.get_running_loop().create_future()
            )
            start_result = await thread.run.start(
                input={"messages": [], "value": "init", "items": []}
            )
            run_id = start_result.get("run_id")
            assert run_id, f"run.start returned no run_id: {start_result!r}"
            run_id_future.set_result(run_id)

            canceller = asyncio.create_task(
                _cancel_after_first_event(runs_client, thread.thread_id, run_id_future)
            )

            snapshots: list[dict] = []
            started = time.monotonic()
            iteration_error: BaseException | None = None
            try:
                async for snap in thread.values:
                    snapshots.append(snap)
                    if time.monotonic() - started > _CANCEL_GRACE_SECONDS:
                        raise AssertionError(
                            f"values iterator did not terminate within "
                            f"{_CANCEL_GRACE_SECONDS}s of cancel"
                        )
            except BaseException as err:
                iteration_error = err

            await canceller

            persisted = await threads.get(thread.thread_id)
            status = persisted.get("status")
            print(f"  snapshots before cancel: {len(snapshots)}")
            print(f"  thread.thread_id={thread.thread_id}")
            print(f"  iteration_error={iteration_error!r}")
            print(f"  persisted status={status!r}")
            assert iteration_error is None, (
                f"values iterator raised after cancel: {iteration_error!r}"
            )
            assert status != "success", (
                f"expected non-success terminal status after cancel, got {status!r}"
            )
    finally:
        await raw.aclose()


def _cancel_after_first_event_sync(
    runs_client: Any,
    thread_id: str,
    run_id_event: threading.Event,
    run_id_holder: dict[str, str],
) -> None:
    run_id_event.wait(timeout=10.0)
    run_id = run_id_holder.get("run_id")
    if not run_id:
        return
    time.sleep(0.1)
    with contextlib.suppress(Exception):
        runs_client.cancel(thread_id, run_id, wait=False)


def run_sync() -> None:
    header("sync mid-run cancel")
    threads, raw = make_sync_client()
    from langgraph_sdk._sync.http import SyncHttpClient
    from langgraph_sdk._sync.runs import SyncRunsClient

    runs_client = SyncRunsClient(SyncHttpClient(raw))
    try:
        with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            run_id_event = threading.Event()
            run_id_holder: dict[str, str] = {}
            start_result = thread.run.start(
                input={"messages": [], "value": "init", "items": []}
            )
            run_id = start_result.get("run_id")
            assert run_id, f"run.start returned no run_id: {start_result!r}"
            run_id_holder["run_id"] = run_id
            run_id_event.set()

            canceller = threading.Thread(
                target=_cancel_after_first_event_sync,
                args=(runs_client, thread.thread_id, run_id_event, run_id_holder),
                daemon=True,
                name="cancel-worker",
            )
            canceller.start()

            snapshots: list[dict] = []
            started = time.monotonic()
            iteration_error: BaseException | None = None
            try:
                for snap in thread.values:
                    snapshots.append(snap)
                    if time.monotonic() - started > _CANCEL_GRACE_SECONDS:
                        raise AssertionError(
                            f"values iterator did not terminate within "
                            f"{_CANCEL_GRACE_SECONDS}s of cancel"
                        )
            except BaseException as err:
                iteration_error = err

            canceller.join(timeout=5)

            persisted = threads.get(thread.thread_id)
            status = persisted.get("status")
            print(f"  snapshots before cancel: {len(snapshots)}")
            print(f"  thread.thread_id={thread.thread_id}")
            print(f"  iteration_error={iteration_error!r}")
            print(f"  persisted status={status!r}")
            assert iteration_error is None, (
                f"values iterator raised after cancel: {iteration_error!r}"
            )
            assert status != "success", (
                f"expected non-success terminal status after cancel, got {status!r}"
            )
    finally:
        with contextlib.suppress(Exception):
            raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
