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


def test_values_captures_values_events():
    reducer = ValuesTransformer()
    reducer.init()
    reducer.process(_event("values", {"a": 1}))
    reducer.process(_event("values", {"b": 2}))
    reducer.finalize()

    assert len(reducer.values_log) == 2
    assert reducer.values_log[0]["data"] == {"a": 1}
    assert reducer.values_log[1]["data"] == {"b": 2}


def test_values_ignores_other_modes():
    reducer = ValuesTransformer()
    reducer.init()
    reducer.process(_event("updates", {"x": 1}))
    reducer.process(_event("messages", {"event": "message-start"}))
    reducer.finalize()

    assert len(reducer.values_log) == 0


def test_values_latest_per_namespace():
    reducer = ValuesTransformer()
    reducer.init()
    reducer.process(_event("values", {"v": 1}, ns=["child:0"]))
    reducer.process(_event("values", {"v": 2}, ns=["child:0"]))
    assert reducer.get_latest("child:0") == {"v": 2}


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


def test_messages_groups_lifecycle():
    reducer = MessagesTransformer()
    reducer.init()
    reducer.process(_msg_start())
    reducer.process(_content_delta("hi"))
    reducer.process(_msg_finish())
    reducer.finalize()

    assert len(reducer.messages_log) == 1
    assert isinstance(reducer.messages_log[0], ChatModelStream)
    assert reducer.messages_log[0].done


def test_messages_multiple_sequential():
    reducer = MessagesTransformer()
    reducer.init()
    reducer.process(_msg_start(message_id="m1"))
    reducer.process(_msg_finish())
    reducer.process(_msg_start(message_id="m2"))
    reducer.process(_msg_finish())
    reducer.finalize()

    assert len(reducer.messages_log) == 2


def test_messages_namespace_filter():
    reducer = MessagesTransformer(namespace=["root"])
    reducer.init()
    reducer.process(_msg_start(ns=["root"]))
    reducer.process(_msg_finish(ns=["root"]))
    reducer.process(_msg_start(ns=["other"], message_id="m2"))
    reducer.process(_msg_finish(ns=["other"]))
    reducer.finalize()

    assert len(reducer.messages_log) == 1


def test_messages_node_filter():
    reducer = MessagesTransformer(node_filter="agent")
    reducer.init()
    reducer.process(_msg_start(node="agent"))
    reducer.process(_msg_finish(node="agent"))
    reducer.process(_msg_start(node="tools", message_id="m2"))
    reducer.process(_msg_finish(node="tools"))
    reducer.finalize()

    assert len(reducer.messages_log) == 1


def test_messages_error_event():
    """An error event should fail the active ChatModelStream."""
    reducer = MessagesTransformer()
    reducer.init()
    reducer.process(_msg_start())
    reducer.process(_content_delta("partial"))
    reducer.process(
        _event("messages", {"event": "error", "message": "connection lost"}),
    )
    reducer.finalize()

    assert len(reducer.messages_log) == 1
    assert reducer.messages_log[0].done


def test_messages_fail_propagates_to_active():
    """transformer.fail() should mark active streams as done."""
    reducer = MessagesTransformer()
    reducer.init()
    reducer.process(_msg_start())
    reducer.process(_content_delta("partial"))
    reducer.fail(RuntimeError("graph failed"))

    assert len(reducer.messages_log) == 1
    assert reducer.messages_log[0].done
