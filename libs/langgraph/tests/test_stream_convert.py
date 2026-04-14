from langgraph.stream._convert import STREAM_V2_MODES, convert_to_protocol_event


def test_values_mode():
    evt = convert_to_protocol_event((), "values", {"x": 1})
    assert evt is not None
    assert evt["method"] == "values"
    assert evt["params"]["data"] == {"x": 1}


def test_updates_mode():
    evt = convert_to_protocol_event((), "updates", {"node": "out"})
    assert evt is not None
    assert evt["method"] == "updates"


def test_messages_mode():
    evt = convert_to_protocol_event((), "messages", {"event": "msg"})
    assert evt is not None
    assert evt["method"] == "messages"


def test_custom_mode():
    evt = convert_to_protocol_event((), "custom", "hello")
    assert evt is not None
    assert evt["method"] == "custom"
    assert evt["params"]["data"] == "hello"


def test_debug_mode():
    evt = convert_to_protocol_event((), "debug", {})
    assert evt is not None
    assert evt["method"] == "debug"


def test_checkpoints_mode():
    evt = convert_to_protocol_event((), "checkpoints", {})
    assert evt is not None
    assert evt["method"] == "checkpoints"


def test_tasks_mode():
    evt = convert_to_protocol_event((), "tasks", {})
    assert evt is not None
    assert evt["method"] == "tasks"


def test_namespace_passthrough():
    evt = convert_to_protocol_event(("agent", "0"), "values", {})
    assert evt is not None
    assert evt["params"]["namespace"] == ["agent", "0"]


def test_timestamp_populated():
    evt = convert_to_protocol_event((), "values", {})
    assert evt is not None
    assert isinstance(evt["params"]["timestamp"], int)
    assert evt["params"]["timestamp"] > 0


def test_unknown_mode_returns_none():
    assert convert_to_protocol_event((), "unknown_mode", {}) is None


def test_node_parameter():
    evt = convert_to_protocol_event((), "values", {}, node="agent")
    assert evt is not None
    assert evt["params"]["node"] == "agent"


def test_type_is_event():
    evt = convert_to_protocol_event((), "values", {})
    assert evt is not None
    assert evt["type"] == "event"


def test_stream_v2_modes_complete():
    assert set(STREAM_V2_MODES) == {
        "values",
        "updates",
        "messages",
        "custom",
        "checkpoints",
        "tasks",
        "debug",
    }
