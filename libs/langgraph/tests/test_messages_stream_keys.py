"""`stream_mode="messages"` can be restricted to specific state keys (#6798).

Default behaviour must stay unchanged: without configuration, every message
found in the state is streamed.
"""

from operator import add
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.pregel._messages import StreamMessagesHandlerV2

pytestmark = pytest.mark.anyio

PUBLIC = "public reply"
SECRET = "internal reasoning - should not leak"


class State(TypedDict):
    messages: Annotated[list, add]
    agent_messages: Annotated[list, add]


def node(state: State) -> State:
    return {
        "messages": [AIMessage(content=PUBLIC)],
        "agent_messages": [AIMessage(content=SECRET)],
    }


def build():
    b = StateGraph(State)
    b.add_node("n", node)
    b.add_edge(START, "n")
    b.add_edge("n", END)
    return b.compile()


INPUT = {"messages": [], "agent_messages": []}


def streamed(config=None):
    chunks = list(build().stream(INPUT, config=config, stream_mode="messages"))
    return [c[0].content for c in chunks]


async def astreamed(config=None):
    chunks = [
        chunk
        async for chunk in build().astream(INPUT, config=config, stream_mode="messages")
    ]
    return [c[0].content for c in chunks]


def test_default_streams_every_message_field():
    """Backwards compatibility: no config means everything is streamed."""
    contents = streamed()
    assert PUBLIC in contents
    assert SECRET in contents


def test_restricted_to_configured_keys_only():
    config = {"configurable": {"__pregel_stream_messages_keys": ["messages"]}}
    contents = streamed(config)
    assert PUBLIC in contents
    assert SECRET not in contents


def test_other_key_can_be_selected_instead():
    config = {"configurable": {"__pregel_stream_messages_keys": ["agent_messages"]}}
    contents = streamed(config)
    assert PUBLIC not in contents
    assert SECRET in contents


def test_unknown_key_streams_nothing():
    config = {"configurable": {"__pregel_stream_messages_keys": ["nope"]}}
    assert streamed(config) == []


async def test_astream_default_streams_every_message_field():
    """`astream` must behave exactly like `stream` when no keys are configured."""
    contents = await astreamed()
    assert PUBLIC in contents
    assert SECRET in contents


async def test_astream_restricted_to_configured_keys_only():
    config = {"configurable": {"__pregel_stream_messages_keys": ["messages"]}}
    contents = await astreamed(config)
    assert PUBLIC in contents
    assert SECRET not in contents


async def test_astream_unknown_key_streams_nothing():
    config = {"configurable": {"__pregel_stream_messages_keys": ["nope"]}}
    assert await astreamed(config) == []


def test_v2_handler_accepts_state_keys_without_error():
    """Regression: StreamMessagesHandlerV2 must accept ``state_keys`` so that the
    v2-flagged ``stream_mode=\"messages\"`` path (which passes it unconditionally in
    main.py) does not raise ``TypeError``. See review on PR #8868.
    """
    h = StreamMessagesHandlerV2(lambda c: None, False, state_keys=["messages"])
    assert h.state_keys == ["messages"]
    h2 = StreamMessagesHandlerV2(lambda c: None, False)
    assert h2.state_keys is None


def test_v2_path_restricted_to_configured_keys_only():
    """Full round-trip through the v2-flagged messages handler with keys set."""
    config = {
        "configurable": {
            "__pregel_stream_messages_v2": True,
            "__pregel_stream_messages_keys": ["messages"],
        }
    }
    contents = [
        c["data"][0].content
        for c in build().stream(
            INPUT, config=config, stream_mode="messages", version="v2"
        )
    ]
    assert PUBLIC in contents
    assert SECRET not in contents


async def test_astream_v2_path_restricted_to_configured_keys_only():
    config = {
        "configurable": {
            "__pregel_stream_messages_v2": True,
            "__pregel_stream_messages_keys": ["messages"],
        }
    }
    contents = [
        c["data"][0].content
        async for c in build().astream(
            INPUT, config=config, stream_mode="messages", version="v2"
        )
    ]
    assert PUBLIC in contents
    assert SECRET not in contents
