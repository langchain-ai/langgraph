"""Compile-time warning when a checkpointer can't reconstruct `DeltaChannel`.

`DeltaChannel` reconstructs state via `BaseCheckpointSaver`'s delta history
API (`get_delta_channel_history`, added in `langgraph-checkpoint>=4.1.0`). A
saver from an older package lacks that method, so reconstruction fails at
runtime. `StateGraph.compile` warns at compile time so users upgrade.
"""

from __future__ import annotations

import warnings
from typing import Annotated

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.message import _messages_delta_reducer


class _DeltaState(TypedDict):
    messages: Annotated[
        list, DeltaChannel(_messages_delta_reducer, snapshot_frequency=50)
    ]


class _PlainState(TypedDict):
    values: Annotated[list, lambda a, b: (a or []) + (b or [])]


def _delta_graph() -> StateGraph:
    return (
        StateGraph(_DeltaState)
        .add_node("n", lambda s: {"messages": []})
        .add_edge(START, "n")
    )


@pytest.fixture
def saver_without_delta_api(monkeypatch: pytest.MonkeyPatch) -> InMemorySaver:
    """A saver whose class tree lacks the delta history API (old package)."""
    for cls in (InMemorySaver, BaseCheckpointSaver):
        for name in ("get_delta_channel_history", "aget_delta_channel_history"):
            if name in cls.__dict__:
                monkeypatch.delattr(cls, name)
    saver = InMemorySaver()
    assert not hasattr(saver, "get_delta_channel_history")
    return saver


def test_delta_graph_warns_for_unsupported_checkpointer(
    saver_without_delta_api: InMemorySaver,
) -> None:
    with pytest.warns(UserWarning, match="does not support `DeltaChannel`"):
        _delta_graph().compile(checkpointer=saver_without_delta_api)


def test_delta_graph_no_warning_for_supported_checkpointer() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _delta_graph().compile(checkpointer=InMemorySaver())


def test_delta_graph_no_warning_without_checkpointer() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _delta_graph().compile()


def test_non_delta_graph_no_warning_with_unsupported_checkpointer(
    saver_without_delta_api: InMemorySaver,
) -> None:
    graph = (
        StateGraph(_PlainState)
        .add_node("n", lambda s: {"values": []})
        .add_edge(START, "n")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph.compile(checkpointer=saver_without_delta_api)
