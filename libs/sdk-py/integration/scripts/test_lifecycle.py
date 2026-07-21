"""Exercise lifecycle state + `thread.run.respond(...)` against the integration API.

The example graph's `ask_human` node calls `interrupt("Are we good?")`. This
script:

1. Starts a run.
2. Waits until `thread.interrupted` becomes True (an `input.requested`
   lifecycle event lands).
3. Inspects `thread.interrupts` to see the outstanding payload.
4. Calls `thread.run.respond("yes")` to resume.
5. Awaits `thread.output` for the final state.
"""

from __future__ import annotations

import asyncio

from _common import (
    ASSISTANT_ID,
    check_api_reachable,
    header,
    make_async_client,
    make_sync_client,
)


async def run_async() -> None:
    header("async lifecycle + respond")
    threads, raw = make_async_client()
    try:
        async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            await thread.run.start(input={"messages": [], "value": "init", "items": []})

            # Drain values until interrupt fires. (`thread.values` ends when
            # the run terminates OR when the run is paused on an interrupt;
            # the latter sets `thread.interrupted` mid-iteration.)
            saw_interrupt = False
            async for _snap in thread.values:
                if thread.interrupted:
                    saw_interrupt = True
                    break

            print(f"  thread.interrupted = {thread.interrupted}")
            print(f"  thread.interrupts  = {thread.interrupts!r}")
            assert thread.interrupted, "expected an interrupt before the run completed"
            assert thread.interrupts, "expected interrupts list to be populated"

            await thread.run.respond("yes")

            final = await thread.output
            print(f"  final output items: {final.get('items')!r}")
            assert "asked" in final.get("items", []), (
                "expected ask_human to have run after respond"
            )
            print(f"  saw_interrupt before respond = {saw_interrupt}")
    finally:
        await raw.aclose()


def run_sync() -> None:
    header("sync lifecycle + respond")
    threads, raw = make_sync_client()
    try:
        with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            thread.run.start(input={"messages": [], "value": "init", "items": []})

            saw_interrupt = False
            for _snap in thread.values:
                if thread.interrupted:
                    saw_interrupt = True
                    break

            print(f"  thread.interrupted = {thread.interrupted}")
            print(f"  thread.interrupts  = {thread.interrupts!r}")
            assert thread.interrupted, "expected an interrupt before the run completed"
            assert thread.interrupts, "expected interrupts list to be populated"

            thread.run.respond("yes")

            final = thread.output
            print(f"  final output items: {final.get('items')!r}")
            assert "asked" in final.get("items", []), (
                "expected ask_human to have run after respond"
            )
            print(f"  saw_interrupt before respond = {saw_interrupt}")
    finally:
        raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
