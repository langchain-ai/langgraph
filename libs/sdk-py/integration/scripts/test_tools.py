"""Exercise `thread.tool_calls` against the `tools_agent` graph.

`tools_agent` (`graph/tools_agent.py`) wraps a
`FakeMessagesListChatModel` in `create_agent` with a real `search`
tool. The first scripted model turn returns an `AIMessage` with a
`tool_calls=[search(query="v3")]`; langchain's tool node then executes
`search` and surfaces a `ToolMessage`; the second turn returns a final
`AIMessage("done.")` that terminates the agent.

Compared to `test_tool_calls.py` (which targets the synthetic
`streaming_graph` and never produces real tool-call telemetry), this
test verifies the v3 ``tools`` channel actually fires when the
canonical langchain-agent surface is in play.

Pattern note: `thread.tool_calls` yields handles incrementally, but
each handle's `deltas` and `output` are only completed when the
*outer* iterator processes the matching `tool-finished` event. Drain
the outer iterator to completion FIRST (via a list comprehension),
then inspect handles -- this is the same pattern used in
`tests/streaming/test_tool_calls_projection.py`. Iterating
`handle.deltas` while the outer iterator is still suspended at its
`yield` deadlocks.
"""

from __future__ import annotations

import asyncio

from _common import check_api_reachable, header, make_async_client, make_sync_client

TOOLS_ASSISTANT_ID = "tools_agent"


async def run_async() -> None:
    header("async tools_agent tool_calls")
    threads, raw = make_async_client()
    try:
        async with threads.stream(assistant_id=TOOLS_ASSISTANT_ID) as thread:
            await thread.run.start(
                input={"messages": [{"role": "human", "content": "search for v3"}]}
            )

            # Drain the outer iterator first; lifecycle-terminal triggers
            # the None sentinel via the shared SSE fanout once the run
            # completes naturally.
            handles = [h async for h in thread.tool_calls]
            print(f"  total handles: {len(handles)}")
            for handle in handles:
                deltas = [d async for d in handle.deltas]
                output = await handle.output
                joined = "".join(deltas)
                print(
                    f"  tool {handle.name}({handle.tool_call_id}): "
                    f"args_stream={joined!r} output={output!r}"
                )
            assert any(h.name == "search" for h in handles), (
                "expected `search` tool call"
            )
    finally:
        await raw.aclose()


def run_sync() -> None:
    header("sync tools_agent tool_calls")
    threads, raw = make_sync_client()
    try:
        with threads.stream(assistant_id=TOOLS_ASSISTANT_ID) as thread:
            thread.run.start(
                input={"messages": [{"role": "human", "content": "search for v3"}]}
            )

            handles = list(thread.tool_calls)
            print(f"  total handles: {len(handles)}")
            for handle in handles:
                deltas = list(handle.deltas)
                output = handle.output
                joined = "".join(deltas)
                print(
                    f"  tool {handle.name}({handle.tool_call_id}): "
                    f"args_stream={joined!r} output={output!r}"
                )
            assert any(h.name == "search" for h in handles), (
                "expected `search` tool call"
            )
    finally:
        raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
