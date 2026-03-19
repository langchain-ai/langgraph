import asyncio

import pytest
from typing_extensions import TypedDict

from saf_python_sdk.advanced_graph import AdvancedStateGraph, Context
from saf_python_sdk.types import Command, Send

pytestmark = pytest.mark.anyio


class StreamState(TypedDict):
    done: bool


async def test_custom_stream_receive_and_close() -> None:
    graph: AdvancedStateGraph[StreamState] = AdvancedStateGraph(StreamState)
    graph.add_custom_outout_stream("high", dict[str, int | str])
    graph.add_custom_outout_stream("regular", dict[str, int | str])

    async def start_node(ctx: Context, state: StreamState) -> Command:
        ctx.send_custom_stream_event("high", {"step": "start", "value": 1})
        await asyncio.sleep(0.08)
        ctx.send_custom_stream_event("regular", {"step": "start", "value": 2})
        return Command(update=state, goto=Send("finish_node", None))

    async def finish_node(state: StreamState) -> dict[str, bool]:
        return {"done": True}

    graph.add_entry_node(start_node)
    graph.add_finish_node(finish_node)

    handler = await graph.compile().astart({"done": False}, stream_mode="custom")

    event = await handler.receive_stream("high")
    assert isinstance(event, dict)
    assert event["step"] == "start"
    assert event["value"] == 1

    event_regular = await handler.receive_stream("regular")
    assert isinstance(event_regular, dict)
    assert event_regular["value"] == 2

    handler.close_all_streams()
    assert await handler.receive_stream("high") is None
    assert await handler.receive_stream("regular") is None

    result = await handler.aresult()
    assert result["done"] is True


async def test_only_custom_stream_mode_supported() -> None:
    graph: AdvancedStateGraph[StreamState] = AdvancedStateGraph(StreamState)
    graph.add_custom_outout_stream("regular", dict[str, str])

    async def start_node(ctx: Context, state: StreamState) -> Command:
        ctx.send_custom_stream_event("regular", {"hello": "world"})
        return Command(update=state)

    graph.add_entry_node(start_node)

    handler = await graph.compile().astart({"done": False}, stream_mode="values")
    with pytest.raises(Exception, match="only `custom` is supported"):
        await handler.aresult()
