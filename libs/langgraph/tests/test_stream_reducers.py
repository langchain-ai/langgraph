from typing import Any

import pytest

from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.chat_model_stream import ChatModelStream
from langgraph.stream.transformers import MessagesTransformer, ValuesTransformer


def _event(
    mode: str,
    data: Any,
    ns: list[str] | None = None,
    node: str | None = None,
) -> ProtocolEvent:
    ev = convert_to_protocol_event(tuple(ns or []), mode, data, node=node)
    assert ev is not None
    return ev


# -- ValuesTransformer ---------------------------------------------------------


@pytest.mark.anyio
async def test_values_captures_values_events():
    reducer = ValuesTransformer()
    reducer.init()
    reducer.process(_event("values", {"a": 1}))
    reducer.process(_event("values", {"b": 2}))
    reducer.finalize()

    collected = []
    async for item in reducer.values_log.subscribe(0):
        collected.append(item)
    assert len(collected) == 2
    assert collected[0]["data"] == {"a": 1}
    assert collected[1]["data"] == {"b": 2}


@pytest.mark.anyio
async def test_values_ignores_other_modes():
    reducer = ValuesTransformer()
    reducer.init()
    reducer.process(_event("updates", {"x": 1}))
    reducer.process(_event("messages", {"event": "message-start"}))
    reducer.finalize()

    collected = []
    async for item in reducer.values_log.subscribe(0):
        collected.append(item)
    assert len(collected) == 0


@pytest.mark.anyio
async def test_values_latest_per_namespace():
    reducer = ValuesTransformer()
    reducer.init()
    reducer.process(_event("values", {"v": 1}, ns=["child:0"]))
    reducer.process(_event("values", {"v": 2}, ns=["child:0"]))
    assert reducer.get_latest("child:0") == {"v": 2}


@pytest.mark.anyio
async def test_values_finalize_closes_log():
    reducer = ValuesTransformer()
    reducer.init()
    reducer.process(_event("values", {"a": 1}))
    reducer.finalize()
    assert reducer.values_log.closed


# -- MessagesTransformer -------------------------------------------------------


def _msg_start(ns=None, node=None, message_id="msg-1"):
    return _event(
        "messages",
        {"event": "message-start", "message_id": message_id},
        ns=ns,
        node=node,
    )


def _content_delta(text, ns=None, node=None):
    return _event(
        "messages",
        {
            "event": "content-block-delta",
            "content_block": {"type": "text", "text": text},
        },
        ns=ns,
        node=node,
    )


def _msg_finish(ns=None, node=None):
    return _event(
        "messages",
        {"event": "message-finish", "reason": "stop"},
        ns=ns,
        node=node,
    )


@pytest.mark.anyio
async def test_messages_groups_lifecycle():
    reducer = MessagesTransformer()
    reducer.init()
    reducer.process(_msg_start())
    reducer.process(_content_delta("hi"))
    reducer.process(_msg_finish())
    reducer.finalize()

    collected = []
    async for stream in reducer.messages_log.subscribe(0):
        collected.append(stream)
    assert len(collected) == 1
    assert isinstance(collected[0], ChatModelStream)
    assert collected[0].done


@pytest.mark.anyio
async def test_messages_multiple_sequential():
    reducer = MessagesTransformer()
    reducer.init()
    reducer.process(_msg_start(message_id="m1"))
    reducer.process(_msg_finish())
    reducer.process(_msg_start(message_id="m2"))
    reducer.process(_msg_finish())
    reducer.finalize()

    collected = []
    async for stream in reducer.messages_log.subscribe(0):
        collected.append(stream)
    assert len(collected) == 2


@pytest.mark.anyio
async def test_messages_namespace_filter():
    reducer = MessagesTransformer(namespace=["root"])
    reducer.init()
    reducer.process(_msg_start(ns=["root"]))
    reducer.process(_msg_finish(ns=["root"]))
    reducer.process(_msg_start(ns=["other"], message_id="m2"))
    reducer.process(_msg_finish(ns=["other"]))
    reducer.finalize()

    collected = []
    async for stream in reducer.messages_log.subscribe(0):
        collected.append(stream)
    assert len(collected) == 1


@pytest.mark.anyio
async def test_messages_node_filter():
    reducer = MessagesTransformer(node_filter="agent")
    reducer.init()
    reducer.process(_msg_start(node="agent"))
    reducer.process(_msg_finish(node="agent"))
    reducer.process(_msg_start(node="tools", message_id="m2"))
    reducer.process(_msg_finish(node="tools"))
    reducer.finalize()

    collected = []
    async for stream in reducer.messages_log.subscribe(0):
        collected.append(stream)
    assert len(collected) == 1


@pytest.mark.anyio
async def test_messages_error_event():
    """An error event should fail the active ChatModelStream."""
    reducer = MessagesTransformer()
    reducer.init()
    reducer.process(_msg_start())
    reducer.process(_content_delta("partial"))
    reducer.process(
        _event("messages", {"event": "error", "message": "connection lost"}),
    )
    reducer.finalize()

    collected: list[ChatModelStream] = []
    async for stream in reducer.messages_log.subscribe(0):
        collected.append(stream)
    assert len(collected) == 1
    assert collected[0].done


@pytest.mark.anyio
async def test_messages_fail_propagates_to_active():
    """transformer.fail() should propagate the error to any active streams."""
    reducer = MessagesTransformer()
    reducer.init()
    reducer.process(_msg_start())
    reducer.process(_content_delta("partial"))
    reducer.fail(RuntimeError("graph failed"))

    # The messages log should be failed too
    with pytest.raises(RuntimeError, match="graph failed"):
        async for _ in reducer.messages_log.subscribe(0):
            pass


@pytest.mark.anyio
async def test_values_fail_propagates():
    reducer = ValuesTransformer()
    reducer.init()
    reducer.process(_event("values", {"a": 1}))
    reducer.fail(RuntimeError("graph failed"))

    with pytest.raises(RuntimeError, match="graph failed"):
        async for _ in reducer.values_log.subscribe(0):
            pass
