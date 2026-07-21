"""`thread.tool_calls` against the `tools_agent` graph."""

from __future__ import annotations

import pytest

from .conftest import TOOLS_ASSISTANT_ID

pytestmark = pytest.mark.integration


async def test_tools_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(assistant_id=TOOLS_ASSISTANT_ID) as thread:
        await thread.run.start(
            input={"messages": [{"role": "human", "content": "search for v3"}]}
        )

        # Drain the outer iterator first; iterating each handle's `.deltas`
        # while the outer is suspended deadlocks.
        handles = [h async for h in thread.tool_calls]
        assert handles, "expected at least one tool call handle"
        assert any(h.name == "search" for h in handles), "expected `search` tool call"

        for handle in handles:
            deltas = "".join([d async for d in handle.deltas])
            output = await handle.output
            assert output.get("status") == "success", (
                f"tool {handle.name} non-success output: {output!r}"
            )
            # The `tools_agent` fake model returns the tool call args pre-built,
            # so the streamed args buffer is empty by design.
            assert isinstance(deltas, str)


def test_tools_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=TOOLS_ASSISTANT_ID) as thread:
        thread.run.start(
            input={"messages": [{"role": "human", "content": "search for v3"}]}
        )

        handles = list(thread.tool_calls)
        assert handles, "expected at least one tool call handle"
        assert any(h.name == "search" for h in handles), "expected `search` tool call"

        for handle in handles:
            deltas = "".join(list(handle.deltas))
            output = handle.output
            assert output.get("status") == "success"
            assert isinstance(deltas, str)
