"""`threads.update_state(...)` during an interrupt persists the mutation."""

from __future__ import annotations

import asyncio
import time

import pytest

from langgraph_sdk.errors import ConflictError

from .conftest import ASSISTANT_ID

pytestmark = pytest.mark.integration

_PATCHED_VALUE = "patched"
_UPDATE_STATE_RETRY_BUDGET = 5.0


async def _update_state_with_retry_async(threads, thread_id, values) -> None:
    """`thread.interrupted` flips before the server commits the run row; retry briefly."""
    delay = 0.05
    deadline = asyncio.get_running_loop().time() + _UPDATE_STATE_RETRY_BUDGET
    last_err: Exception | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            await threads.update_state(thread_id, values)
            return
        except ConflictError as err:
            last_err = err
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.5)
    raise AssertionError(
        f"update_state never accepted within {_UPDATE_STATE_RETRY_BUDGET}s: {last_err!r}"
    )


def _update_state_with_retry_sync(threads, thread_id, values) -> None:
    delay = 0.05
    deadline = time.monotonic() + _UPDATE_STATE_RETRY_BUDGET
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            threads.update_state(thread_id, values)
            return
        except ConflictError as err:
            last_err = err
            time.sleep(delay)
            delay = min(delay * 2, 0.5)
    raise AssertionError(
        f"update_state never accepted within {_UPDATE_STATE_RETRY_BUDGET}s: {last_err!r}"
    )


async def test_update_state_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})

        async for _ in thread.values:
            if thread.interrupted:
                break
        assert thread.interrupted, "expected interrupt before update_state"

        pre_state = await threads.get_state(thread.thread_id)
        pre_value = (pre_state.get("values") or {}).get("value")
        # `stream_message` overwrites value="init" with "x" before the interrupt.
        assert pre_value == "x", f"unexpected pre-update value: {pre_value!r}"

        await _update_state_with_retry_async(
            threads, thread.thread_id, {"value": _PATCHED_VALUE}
        )

        post_state = await threads.get_state(thread.thread_id)
        post_value = (post_state.get("values") or {}).get("value")
        assert post_value == _PATCHED_VALUE, (
            f"update_state did not persist: value={post_value!r}"
        )


def test_update_state_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})

        for _ in thread.values:
            if thread.interrupted:
                break
        assert thread.interrupted, "expected interrupt before update_state"

        pre_state = threads.get_state(thread.thread_id)
        pre_value = (pre_state.get("values") or {}).get("value")
        assert pre_value == "x", f"unexpected pre-update value: {pre_value!r}"

        _update_state_with_retry_sync(
            threads, thread.thread_id, {"value": _PATCHED_VALUE}
        )

        post_state = threads.get_state(thread.thread_id)
        post_value = (post_state.get("values") or {}).get("value")
        assert post_value == _PATCHED_VALUE, (
            f"update_state did not persist: value={post_value!r}"
        )
