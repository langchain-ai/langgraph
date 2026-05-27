"""Exercise `threads.update_state(...)` mid-run against the integration API.

Flow:

1. Stream the canonical graph; `run.start` with `value="init"`.
2. Drain values until the interrupt fires at `ask_human`.
3. Call `threads.update_state(thread_id, {"value": "patched"})` to mutate
   the persisted state while the run is paused.
4. Read state back via `threads.get_state(thread_id)` and assert
   `state["values"]["value"] == "patched"`.

Why no respond afterwards: in langgraph-api, `update_state` against an
interrupted thread commits a new checkpoint that consumes the
outstanding interrupt. A subsequent `run.respond(...)` then fails with
`no_such_interrupt`. The meaningful integration invariant here is just
that the REST mutation lands on the same persisted thread the streaming
proxy was driving (no thread_id drift between client and server).
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from _common import (
    ASSISTANT_ID,
    check_api_reachable,
    header,
    make_async_client,
    make_sync_client,
)

from langgraph_sdk.errors import ConflictError

_PATCHED_VALUE = "patched"
_UPDATE_STATE_RETRY_BUDGET = 5.0


async def _update_state_with_retry_async(threads, thread_id: str, values: dict) -> None:
    """Retry update_state on ConflictError until the server's run row settles.

    `thread.interrupted` flips when the client sees the `input.requested`
    lifecycle event, which can land before the server commits the run row
    to a non-busy state. Retry with backoff for a few seconds.
    """
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


def _update_state_with_retry_sync(threads, thread_id: str, values: dict) -> None:
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


async def run_async() -> None:
    header("async update_state during interrupt")
    threads, raw = make_async_client()
    try:
        async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            await thread.run.start(input={"messages": [], "value": "init", "items": []})

            async for _ in thread.values:
                if thread.interrupted:
                    break
            assert thread.interrupted, "expected interrupt before update_state"

            pre_state = await threads.get_state(thread.thread_id)
            pre_value = (pre_state.get("values") or {}).get("value")
            print(f"  pre-update value={pre_value!r}")
            # `stream_message` overwrites value="init" with value="x" before the
            # interrupt; verify we're starting from the expected pre-update state.
            assert pre_value == "x", f"unexpected pre-update value: {pre_value!r}"

            await _update_state_with_retry_async(
                threads, thread.thread_id, {"value": _PATCHED_VALUE}
            )

            post_state = await threads.get_state(thread.thread_id)
            post_values = post_state.get("values") or {}
            post_value = post_values.get("value")
            print(f"  thread_id={thread.thread_id}")
            print(f"  post-update value={post_value!r}")
            assert post_value == _PATCHED_VALUE, (
                f"update_state did not persist: value={post_value!r}"
            )
    finally:
        await raw.aclose()


def run_sync() -> None:
    header("sync update_state during interrupt")
    threads, raw = make_sync_client()
    try:
        with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            thread.run.start(input={"messages": [], "value": "init", "items": []})

            for _ in thread.values:
                if thread.interrupted:
                    break
            assert thread.interrupted, "expected interrupt before update_state"

            pre_state = threads.get_state(thread.thread_id)
            pre_value = (pre_state.get("values") or {}).get("value")
            print(f"  pre-update value={pre_value!r}")
            # `stream_message` overwrites value="init" with value="x" before the
            # interrupt; verify we're starting from the expected pre-update state.
            assert pre_value == "x", f"unexpected pre-update value: {pre_value!r}"

            _update_state_with_retry_sync(
                threads, thread.thread_id, {"value": _PATCHED_VALUE}
            )

            post_state = threads.get_state(thread.thread_id)
            post_values = post_state.get("values") or {}
            post_value = post_values.get("value")
            print(f"  thread_id={thread.thread_id}")
            print(f"  post-update value={post_value!r}")
            assert post_value == _PATCHED_VALUE, (
                f"update_state did not persist: value={post_value!r}"
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
