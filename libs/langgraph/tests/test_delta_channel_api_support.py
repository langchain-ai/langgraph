"""Compile-time check that `DeltaChannel` graphs run under a supported API server.

`DeltaChannel` reconstruction relies on server-side support added in
`langgraph-api>=0.9.0`. When running under an older API server, delta channels
fail at runtime, so `StateGraph.compile` raises with an upgrade hint. The check
is skipped when `langgraph-api` is not installed (local execution).
"""

from __future__ import annotations

import warnings
from typing import Annotated

import pytest
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph import state as state_module
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
def api_version(monkeypatch: pytest.MonkeyPatch):
    """Override the reported `langgraph-api` version (or simulate absence)."""

    def _set(version: str | None) -> None:
        def fake_version(name: str) -> str:
            if name == "langgraph-api":
                if version is None:
                    raise state_module.metadata.PackageNotFoundError(name)
                return version
            return state_module.metadata.version(name)

        monkeypatch.setattr(state_module.metadata, "version", fake_version)

    return _set


def test_delta_graph_raises_for_old_api(api_version) -> None:
    api_version("0.8.9")
    with pytest.raises(RuntimeError, match="requires `langgraph-api>=0.9.0`"):
        _delta_graph().compile()


def test_delta_graph_ok_for_min_api(api_version) -> None:
    api_version("0.9.0")
    _delta_graph().compile()


def test_delta_graph_ok_for_newer_api(api_version) -> None:
    api_version("0.11.2")
    _delta_graph().compile()


def test_delta_graph_ok_when_api_not_installed(api_version) -> None:
    api_version(None)
    _delta_graph().compile()


def test_non_delta_graph_never_raises(api_version) -> None:
    api_version("0.1.0")
    graph = (
        StateGraph(_PlainState)
        .add_node("n", lambda s: {"values": []})
        .add_edge(START, "n")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph.compile()
