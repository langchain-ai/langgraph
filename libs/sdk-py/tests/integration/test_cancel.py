"""Mid-run cancellation via `runs.cancel(...)`."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any

import pytest

from .conftest import ASSISTANT_ID

pytestmark = pytest.mark.integration

_CANCEL_GRACE_SECONDS = 10.0


async def _cancel_after_first_event(
    runs_client: Any,
    thread_id: str,
    run_id_future: asyncio.Future[str],
) -> None:
    run_id = await run_id_future
    await asyncio.sleep(0.1)
    with contextlib.suppress(Exception):
        await runs_client.cancel(thread_id, run_id, wait=False)


async def test_cancel_async(async_threads) -> None:
    from langgraph_sdk._async.http import HttpClient
    from langgraph_sdk._async.runs import RunsClient

    threads, raw = async_threads
    runs_client = RunsClient(HttpClient(raw))

    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        run_id_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        start_result = await thread.run.start(
            input={"messages": [], "value": "init", "items": []}
        )
        run_id = start_result.get("run_id")
        assert run_id, f"run.start returned no run_id: {start_result!r}"
        run_id_future.set_result(run_id)

        canceller = asyncio.create_task(
            _cancel_after_first_event(runs_client, thread.thread_id, run_id_future)
        )

        started = time.monotonic()
        iteration_error: BaseException | None = None
        try:
            async for _snap in thread.values:
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
        assert iteration_error is None, (
            f"values iterator raised after cancel: {iteration_error!r}"
        )
        assert status != "success", (
            f"expected non-success terminal status after cancel, got {status!r}"
        )


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


def test_cancel_sync(sync_threads) -> None:
    from langgraph_sdk._sync.http import SyncHttpClient
    from langgraph_sdk._sync.runs import SyncRunsClient

    threads, raw = sync_threads
    runs_client = SyncRunsClient(SyncHttpClient(raw))

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

        started = time.monotonic()
        iteration_error: BaseException | None = None
        try:
            for _snap in thread.values:
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
        assert iteration_error is None, (
            f"values iterator raised after cancel: {iteration_error!r}"
        )
        assert status != "success", (
            f"expected non-success terminal status after cancel, got {status!r}"
        )
