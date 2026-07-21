"""Exercise `thread.messages` against the integration API.

The ``stream_message`` node in ``streaming_graph`` invokes a fake
``FakeMessagesListChatModel`` whose ``_stream`` callbacks drive
langgraph's ``StreamMessagesHandlerV2`` -> ``MessagesTransformer``,
so the v3 ``messages`` channel emits the normalized delta lifecycle
(``message-start`` -> ``content-block-start`` ->
``content-block-delta`` -> ``content-block-finish`` ->
``message-finish``) at root namespace.

Pattern note: drain the outer iterator first (list comprehension)
before consuming each handle's chunks -- the outer iterator yields
on ``message-start`` but the inner ``chunk`` stream only completes
when ``message-finish`` is processed by the outer iter. Iterating
chunks while the outer is suspended at ``yield`` deadlocks.
"""

from __future__ import annotations

import asyncio

from _common import (
    ASSISTANT_ID,
    auto_respond_async,
    auto_respond_sync,
    check_api_reachable,
    header,
    make_async_client,
    make_sync_client,
)


async def run_async() -> None:
    header("async messages")
    threads, raw = make_async_client()
    try:
        async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            await thread.run.start(input={"messages": [], "value": "init", "items": []})
            # Graph interrupts at `ask_human`; background responder
            # unblocks so terminal lifecycle fires and the messages
            # iterator exits cleanly.
            responder = auto_respond_async(thread)

            streams = [s async for s in thread.messages]
            await responder
            print(f"  total streams: {len(streams)}")
            for stream in streams:
                text = "".join([t async for t in stream.text])
                msg_id = getattr(stream, "message_id", None) or "?"
                print(f"  message {msg_id}: {text!r}")
            assert streams, "expected at least one streamed message"
    finally:
        await raw.aclose()


def run_sync() -> None:
    header("sync messages")
    threads, raw = make_sync_client()
    try:
        with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            thread.run.start(input={"messages": [], "value": "init", "items": []})
            responder = auto_respond_sync(thread)

            streams = list(thread.messages)
            responder.join(timeout=5)
            print(f"  total streams: {len(streams)}")
            for stream in streams:
                text = "".join(list(stream.text))
                msg_id = getattr(stream, "message_id", None) or "?"
                print(f"  message {msg_id}: {text!r}")
            assert streams, "expected at least one streamed message"
    finally:
        raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
