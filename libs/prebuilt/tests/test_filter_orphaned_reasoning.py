"""Tests for filter_orphaned_reasoning_messages pre_model_hook."""

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.graph.message import RemoveMessage

from langgraph.prebuilt.chat_agent_executor import (
    _is_reasoning_only,
    filter_orphaned_reasoning_messages,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reasoning_msg(id: str = "msg-r1") -> AIMessage:
    return AIMessage(
        content=[{"type": "reasoning", "id": "rs_1", "summary": []}],
        id=id,
    )


def _text_msg(id: str = "msg-t1") -> AIMessage:
    return AIMessage(content="here is my answer", id=id)


def _state(messages: list) -> dict:
    return {"messages": messages}


# ---------------------------------------------------------------------------
# _is_reasoning_only
# ---------------------------------------------------------------------------


def test_is_reasoning_only_pure_reasoning() -> None:
    assert _is_reasoning_only(_reasoning_msg()) is True


def test_is_reasoning_only_text_is_false() -> None:
    assert _is_reasoning_only(_text_msg()) is False


def test_is_reasoning_only_empty_list_is_false() -> None:
    assert _is_reasoning_only(AIMessage(content=[], id="e")) is False


def test_is_reasoning_only_empty_string_is_false() -> None:
    assert _is_reasoning_only(AIMessage(content="", id="e")) is False


def test_is_reasoning_only_mixed_reasoning_and_text_is_false() -> None:
    msg = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "text", "text": "here is my answer"},
        ],
        id="mixed",
    )
    assert _is_reasoning_only(msg) is False


def test_is_reasoning_only_with_tool_calls_is_false() -> None:
    msg = AIMessage(
        content=[{"type": "reasoning", "id": "rs_1", "summary": []}],
        tool_calls=[ToolCall(name="my_tool", args={}, id="call-1")],
        id="tc",
    )
    assert _is_reasoning_only(msg) is False


# ---------------------------------------------------------------------------
# filter_orphaned_reasoning_messages
# ---------------------------------------------------------------------------


def test_no_orphans_returns_none() -> None:
    state = _state([HumanMessage(content="hi"), _text_msg()])
    assert filter_orphaned_reasoning_messages(state) is None


def test_orphaned_reasoning_message_removed() -> None:
    orphan = _reasoning_msg("orphan-id")
    state = _state([HumanMessage(content="hi"), orphan, HumanMessage(content="retry")])

    result = filter_orphaned_reasoning_messages(state)

    assert result is not None
    removes = result["messages"]
    assert len(removes) == 1
    assert isinstance(removes[0], RemoveMessage)
    assert removes[0].id == "orphan-id"


def test_empty_content_message_not_removed() -> None:
    state = _state([HumanMessage(content="hi"), AIMessage(content=[], id="e1")])
    assert filter_orphaned_reasoning_messages(state) is None


def test_tool_call_message_not_removed() -> None:
    ai_with_tool = AIMessage(
        content=[{"type": "reasoning", "id": "rs_1", "summary": []}],
        tool_calls=[ToolCall(name="my_tool", args={}, id="call-1")],
        id="tool-msg",
    )
    state = _state(
        [
            HumanMessage(content="hi"),
            ai_with_tool,
            ToolMessage(content="result", tool_call_id="call-1", name="my_tool"),
        ]
    )
    assert filter_orphaned_reasoning_messages(state) is None


def test_multiple_orphans_all_removed() -> None:
    r1 = _reasoning_msg("r1")
    r2 = _reasoning_msg("r2")
    state = _state([HumanMessage(content="hi"), r1, r2, HumanMessage(content="retry")])

    result = filter_orphaned_reasoning_messages(state)

    assert result is not None
    removed_ids = {m.id for m in result["messages"]}
    assert removed_ids == {"r1", "r2"}


def test_only_human_messages_returns_none() -> None:
    state = _state([HumanMessage(content="hi")])
    assert filter_orphaned_reasoning_messages(state) is None


def test_orphan_without_id_is_skipped() -> None:
    no_id_orphan = AIMessage(
        content=[{"type": "reasoning", "id": "rs_1", "summary": []}],
        id=None,
    )
    state = _state([HumanMessage(content="hi"), no_id_orphan])
    assert filter_orphaned_reasoning_messages(state) is None


def test_idempotent_after_removal() -> None:
    """Second call on a cleaned-up state returns None."""
    orphan = _reasoning_msg("orphan-id")
    messages = [HumanMessage(content="hi"), orphan]

    result = filter_orphaned_reasoning_messages(_state(messages))
    assert result is not None

    # Simulate state update: remove the orphan
    messages = [m for m in messages if getattr(m, "id", None) != "orphan-id"]

    result = filter_orphaned_reasoning_messages(_state(messages))
    assert result is None


def test_complete_turn_with_reasoning_not_removed() -> None:
    """A message with both reasoning AND text blocks is a complete turn — keep it."""
    complete = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "text", "text": "here is my answer"},
        ],
        id="complete",
    )
    state = _state([HumanMessage(content="hi"), complete])
    assert filter_orphaned_reasoning_messages(state) is None
