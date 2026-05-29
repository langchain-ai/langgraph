"""Unit tests for the per-channel Decoders.

Each test drives a single decoder with synthetic events from `_events` and
asserts the items the decoder yields.
"""

from __future__ import annotations

from typing import Any

from langgraph_sdk.stream.decoders import (
    DataDecoder,
    ExtensionsDecoder,
    MessagesDecoder,
    SubgraphsDecoder,
    ToolCallsDecoder,
)
from streaming._events import (
    checkpoints_event,
    custom_event,
    lifecycle_completed_event,
    lifecycle_started_event,
    message_error_event,
    message_finish_event,
    message_start_event,
    message_text_delta_event,
    tasks_result_event,
    tasks_start_event,
    tool_error_event,
    tool_finished_event,
    tool_output_delta_event,
    tool_started_event,
    updates_event,
    values_event,
)


def test_data_decoder_yields_params_data():
    decoder = DataDecoder("values")
    assert list(decoder.feed(values_event(seq=1, x=1))) == [{"x": 1}]
    assert list(decoder.feed(values_event(seq=2, x=2, y=3))) == [{"x": 2, "y": 3}]


def test_data_decoder_ignores_other_methods():
    decoder = DataDecoder("values")
    assert list(decoder.feed(lifecycle_completed_event(seq=1))) == []
    assert list(decoder.feed(updates_event(seq=2, foo=1))) == []


def test_data_decoder_handles_updates_checkpoints_tasks_methods():
    assert list(DataDecoder("updates").feed(updates_event(seq=1, node={"x": 1}))) == [
        {"node": {"x": 1}}
    ]
    assert list(
        DataDecoder("checkpoints").feed(checkpoints_event(seq=2, ts="t", v=4))
    ) == [{"ts": "t", "v": 4}]
    # tasks payloads pass through verbatim as data dicts
    [item] = list(DataDecoder("tasks").feed(tasks_start_event(seq=3, task_id="a")))
    assert item["id"] == "a"


def test_data_decoder_namespace_none_yields_regardless_of_namespace():
    decoder = DataDecoder("checkpoints", namespace=None)
    assert list(decoder.feed(checkpoints_event(seq=1, namespace=["child"], v=1))) == [
        {"v": 1}
    ]


def test_data_decoder_namespace_filter_drops_non_matching_namespace():
    decoder = DataDecoder("checkpoints", namespace=[])
    # root-namespace event is yielded; child-namespace event is filtered out
    assert list(decoder.feed(checkpoints_event(seq=1, v=1))) == [{"v": 1}]
    assert list(decoder.feed(checkpoints_event(seq=2, namespace=["child"], v=2))) == []


class _FakeStream:
    """Stand-in for AsyncChatModelStream/ChatModelStream in decoder tests."""

    def __init__(self, *, namespace, node, message_id):
        self.namespace = namespace
        self.node = node
        self.message_id = message_id
        self.dispatched: list[dict] = []

    def dispatch(self, data):
        self.dispatched.append(data)


def _factory(*, namespace, node, message_id):
    return _FakeStream(namespace=namespace, node=node, message_id=message_id)


def test_messages_decoder_yields_stream_on_message_start():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    streams = list(
        decoder.feed(message_start_event(seq=1, message_id="m-1", node="agent"))
    )
    assert len(streams) == 1
    assert streams[0].message_id == "m-1"
    assert streams[0].node == "agent"
    # The start event is dispatched into the stream too (matches stream.py:432).
    assert (
        streams[0].dispatched and streams[0].dispatched[0]["event"] == "message-start"
    )


def test_messages_decoder_dispatches_delta_to_active_stream():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    delta = message_text_delta_event(seq=2, message_id="m-1", text="hi")
    assert list(decoder.feed(delta)) == []
    assert stream.dispatched[-1]["event"] == "content-block-delta"


def test_messages_decoder_finish_retires_stream():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    list(decoder.feed(message_finish_event(seq=2, message_id="m-1")))
    assert stream.dispatched[-1]["event"] == "message-finish"
    assert all(s is not stream for s in decoder._active.values())
    [again] = list(decoder.feed(message_start_event(seq=3, message_id="m-1")))
    assert again is not stream


def test_messages_decoder_error_retires_stream():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    list(decoder.feed(message_error_event(seq=2, message_id="m-1", message="boom")))
    assert stream.dispatched[-1]["event"] == "error"
    assert all(s is not stream for s in decoder._active.values())


def test_messages_decoder_single_fallback_routes_idless_events():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    list(decoder.feed(message_text_delta_event(seq=2, text="x")))  # no message_id
    assert stream.dispatched[-1]["event"] == "content-block-delta"


def test_messages_decoder_drops_idful_events_for_unknown_stream():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    list(decoder.feed(message_text_delta_event(seq=2, message_id="ghost", text="x")))
    assert all(d["event"] != "content-block-delta" for d in stream.dispatched)


def test_messages_decoder_ignores_other_namespaces():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    assert (
        list(
            decoder.feed(
                message_start_event(seq=1, namespace=["child"], message_id="m-1")
            )
        )
        == []
    )


def test_messages_decoder_drops_idless_delta_when_multiple_active():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [a] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    [b] = list(decoder.feed(message_start_event(seq=2, message_id="m-2")))
    # id-less delta is ambiguous with two active streams -> dropped, routed to neither
    list(decoder.feed(message_text_delta_event(seq=3, text="x")))
    assert all(d["event"] != "content-block-delta" for d in a.dispatched)
    assert all(d["event"] != "content-block-delta" for d in b.dispatched)


class _FakeToolHandle:
    def __init__(self, *, tool_call_id, name, input, namespace):
        self.tool_call_id = tool_call_id
        self.name = name
        self.input = input
        self.namespace = namespace
        self.deltas: list[str] = []
        self.finished_output: Any = None
        self.finished = False
        self.error: BaseException | None = None

    def _push_delta(self, delta):
        self.deltas.append(delta)

    def _finish(self, output):
        self.finished = True
        self.finished_output = output

    def _fail(self, exc):
        self.error = exc


def _tool_factory(*, tool_call_id, name, input, namespace):
    return _FakeToolHandle(
        tool_call_id=tool_call_id, name=name, input=input, namespace=namespace
    )


def test_tool_calls_decoder_yields_handle_on_start():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [handle] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    assert handle.tool_call_id == "tc-1"
    assert handle.name == "search"


def test_tool_calls_decoder_routes_delta_finish_and_error():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    list(decoder.feed(tool_output_delta_event(seq=2, tool_call_id="tc-1", delta="x")))
    assert h.deltas == ["x"]
    list(
        decoder.feed(
            tool_finished_event(seq=3, tool_call_id="tc-1", output={"ok": True})
        )
    )
    assert h.finished and h.finished_output == {"ok": True}

    decoder2 = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h2] = list(
        decoder2.feed(
            tool_started_event(seq=1, tool_call_id="tc-2", tool_name="search")
        )
    )
    list(decoder2.feed(tool_error_event(seq=2, tool_call_id="tc-2", message="boom")))
    assert isinstance(h2.error, RuntimeError) and "boom" in str(h2.error)


def test_tool_calls_decoder_drops_events_for_unknown_id():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    assert (
        list(
            decoder.feed(
                tool_output_delta_event(seq=1, tool_call_id="ghost", delta="x")
            )
        )
        == []
    )


def test_tool_calls_decoder_finish_and_error_retire_handle():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    list(decoder.feed(tool_finished_event(seq=2, tool_call_id="tc-1")))
    # retired: a late delta for the same id is now dropped
    list(
        decoder.feed(tool_output_delta_event(seq=3, tool_call_id="tc-1", delta="late"))
    )
    assert h.deltas == []


def test_tool_calls_decoder_ignores_other_namespaces():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    assert (
        list(
            decoder.feed(
                tool_started_event(seq=1, namespace=["child"], tool_call_id="tc-1")
            )
        )
        == []
    )


def test_tool_calls_decoder_error_retires_handle():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    list(decoder.feed(tool_error_event(seq=2, tool_call_id="tc-1", message="boom")))
    # retired: a late delta for the same id is now dropped
    list(
        decoder.feed(tool_output_delta_event(seq=3, tool_call_id="tc-1", delta="late"))
    )
    assert h.deltas == []


def test_tool_calls_decoder_skips_non_str_tool_call_id():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    # Build a tools event whose data.tool_call_id is not a string.
    bad = tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search")
    bad["params"]["data"]["tool_call_id"] = 123
    assert list(decoder.feed(bad)) == []


def test_tool_calls_decoder_skips_non_str_delta():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    evt = tool_output_delta_event(seq=2, tool_call_id="tc-1", delta="x")
    evt["params"]["data"]["delta"] = 123  # non-str delta must be ignored
    list(decoder.feed(evt))
    assert h.deltas == []


def test_tool_calls_decoder_defaults_missing_tool_name_to_empty():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    evt = tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search")
    del evt["params"]["data"]["tool_name"]  # absent tool_name -> handle.name == ""
    [h] = list(decoder.feed(evt))
    assert h.name == ""


def test_tool_calls_decoder_error_message_defaults_when_blank():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    evt = tool_error_event(seq=2, tool_call_id="tc-1")
    evt["params"]["data"]["message"] = ""  # blank -> default message
    list(decoder.feed(evt))
    assert str(h.error) == "Tool call errored"


class _FakeScopedHandle:
    def __init__(self, *, path, graph_name, trigger_call_id):
        self.path = path
        self.graph_name = graph_name
        self.trigger_call_id = trigger_call_id
        self.status = "started"
        self.error = None
        self.events: list[dict] = []

    def _push_event(self, event):
        self.events.append(event)

    def _finish(self, status, error=None):
        if self.status != "started":
            return
        self.status = status
        self.error = error


def _scoped_factory(*, path, graph_name, trigger_call_id):
    return _FakeScopedHandle(
        path=path, graph_name=graph_name, trigger_call_id=trigger_call_id
    )


def test_subgraphs_decoder_discovers_on_lifecycle_started_once():
    decoder = SubgraphsDecoder(scope=(), handle_factory=_scoped_factory)
    [h] = list(decoder.feed(lifecycle_started_event(seq=1, namespace=["child"])))
    assert h.path == ("child",)
    assert list(decoder.feed(lifecycle_started_event(seq=2, namespace=["child"]))) == []


def test_subgraphs_decoder_discovers_on_tasks_start_without_result():
    decoder = SubgraphsDecoder(scope=(), handle_factory=_scoped_factory)
    [h] = list(decoder.feed(tasks_start_event(seq=1, namespace=["child"])))
    assert h.path == ("child",)


def test_subgraphs_decoder_parses_graph_name_and_trigger_from_segment():
    decoder = SubgraphsDecoder(scope=(), handle_factory=_scoped_factory)
    [h] = list(decoder.feed(tasks_start_event(seq=1, namespace=["agent:call-1"])))
    assert h.graph_name == "agent"
    assert h.trigger_call_id == "call-1"


def test_subgraphs_decoder_fans_out_events_to_active_child():
    decoder = SubgraphsDecoder(scope=(), handle_factory=_scoped_factory)
    [h] = list(decoder.feed(lifecycle_started_event(seq=1, namespace=["child"])))
    inner = message_start_event(seq=2, namespace=["child"], message_id="m")
    assert list(decoder.feed(inner)) == []
    assert inner in h.events  # whole event pushed, not just data


def test_subgraphs_decoder_fans_out_grandchild_to_direct_child():
    decoder = SubgraphsDecoder(scope=(), handle_factory=_scoped_factory)
    [h] = list(decoder.feed(lifecycle_started_event(seq=1, namespace=["child"])))
    grand = message_start_event(seq=2, namespace=["child", "grand"], message_id="m")
    list(decoder.feed(grand))
    assert grand in h.events


def test_subgraphs_decoder_tasks_result_at_parent_finalizes_child():
    decoder = SubgraphsDecoder(scope=(), handle_factory=_scoped_factory)
    # child discovered with a colon segment -> trigger_call_id == "call-1"
    [h] = list(decoder.feed(tasks_start_event(seq=1, namespace=["agent:call-1"])))
    assert h.trigger_call_id == "call-1"
    # the finalizing tasks-result is emitted at the PARENT (root) namespace,
    # with id (task_id) matching the child's trigger_call_id
    list(
        decoder.feed(
            tasks_result_event(seq=2, namespace=[], task_id="call-1", result={"ok": 1})
        )
    )
    assert h.status == "completed"
    # finalized + removed from active: later child-namespace events no longer fan out
    later = message_start_event(seq=3, namespace=["agent:call-1"], message_id="m")
    list(decoder.feed(later))
    assert later not in h.events


def test_subgraphs_decoder_tasks_result_failed_status():
    decoder = SubgraphsDecoder(scope=(), handle_factory=_scoped_factory)
    [h] = list(decoder.feed(tasks_start_event(seq=1, namespace=["agent:call-1"])))
    list(
        decoder.feed(
            tasks_result_event(seq=2, namespace=[], task_id="call-1", error="boom")
        )
    )
    assert h.status == "failed"
    assert h.error == "boom"


def test_subgraphs_decoder_tasks_result_interrupted_status():
    decoder = SubgraphsDecoder(scope=(), handle_factory=_scoped_factory)
    [h] = list(decoder.feed(tasks_start_event(seq=1, namespace=["agent:call-1"])))
    list(
        decoder.feed(
            tasks_result_event(
                seq=2, namespace=[], task_id="call-1", interrupts=[{"id": "i-1"}]
            )
        )
    )
    assert h.status == "interrupted"


def test_subgraphs_decoder_ignores_unrelated_and_scope_itself():
    decoder = SubgraphsDecoder(scope=("root",), handle_factory=_scoped_factory)
    # not a direct child of ("root",): wrong depth / wrong prefix
    assert list(decoder.feed(lifecycle_started_event(seq=1, namespace=["other"]))) == []
    # the scope's own namespace is not a discovery
    assert list(decoder.feed(lifecycle_started_event(seq=2, namespace=["root"]))) == []


def test_extensions_decoder_yields_full_data_for_matching_name():
    decoder = ExtensionsDecoder(name="foo")
    # custom_event(name="foo", x=1) -> params.data = {"name": "foo", "x": 1}
    assert list(decoder.feed(custom_event(seq=1, name="foo", x=1))) == [
        {"name": "foo", "x": 1}
    ]


def test_extensions_decoder_ignores_other_extension_names():
    decoder = ExtensionsDecoder(name="foo")
    assert list(decoder.feed(custom_event(seq=1, name="bar", x=1))) == []


def test_extensions_decoder_ignores_non_custom_methods():
    decoder = ExtensionsDecoder(name="foo")
    assert list(decoder.feed(lifecycle_completed_event(seq=1))) == []


def test_extensions_decoder_ignores_non_dict_data():
    decoder = ExtensionsDecoder(name="foo")
    evt = custom_event(seq=1, name="foo", x=1)
    evt["params"]["data"] = "not-a-dict"
    assert list(decoder.feed(evt)) == []


def test_extensions_decoder_rejects_empty_name():
    import pytest

    with pytest.raises(ValueError):
        ExtensionsDecoder(name="")
