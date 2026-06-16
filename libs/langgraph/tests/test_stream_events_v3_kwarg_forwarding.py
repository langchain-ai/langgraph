"""Tests that ``(a)stream_events(version="v3")`` forwards extra kwargs to the
underlying ``(a)stream`` call, and rejects the kwargs v3 owns internally.

Background: prior to this change the v3 dispatcher silently dropped ``**kwargs``
on the v3 branch while forwarding them on v1/v2, so callers passing e.g.
``context=...`` saw their value disappear with no error. v3 now forwards
caller kwargs to the inner ``(a)stream`` call but rejects ``stream_mode`` and
``subgraphs`` since v3 owns them (``stream_mode`` is built from the
transformer mux; ``subgraphs`` is forced True so nested namespaces flow
through scoped muxes).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import pytest
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


@dataclass
class _Ctx:
    api_key: str


class _State(TypedDict):
    message: str


def _build_context_reading_graph():
    def read_context(state: _State, runtime: Runtime[_Ctx]) -> dict[str, Any]:
        return {"message": f"api key: {runtime.context.api_key}"}

    builder = StateGraph(state_schema=_State, context_schema=_Ctx)
    builder.add_node("read_context", read_context)
    builder.add_edge(START, "read_context")
    builder.add_edge("read_context", END)
    return builder.compile()


class TestKwargForwardingSync:
    def test_context_reaches_node(self) -> None:
        run = _build_context_reading_graph().stream_events(
            {"message": "hello"},
            version="v3",
            context=_Ctx(api_key="sk_sync"),
        )
        assert run.output == {"message": "api key: sk_sync"}

    def test_rejects_stream_mode(self) -> None:
        graph = _build_context_reading_graph()
        with pytest.raises(TypeError, match="stream_mode"):
            graph.stream_events(
                {"message": "hello"},
                version="v3",
                stream_mode=["values"],
            )

    def test_rejects_subgraphs(self) -> None:
        graph = _build_context_reading_graph()
        with pytest.raises(TypeError, match="subgraphs"):
            graph.stream_events(
                {"message": "hello"},
                version="v3",
                subgraphs=False,
            )


@pytest.mark.anyio
@NEEDS_CONTEXTVARS
class TestKwargForwardingAsync:
    async def test_context_reaches_node(self) -> None:
        run = await _build_context_reading_graph().astream_events(
            {"message": "hello"},
            version="v3",
            context=_Ctx(api_key="sk_async"),
        )
        output = await run.output()
        assert output == {"message": "api key: sk_async"}

    async def test_rejects_stream_mode(self) -> None:
        graph = _build_context_reading_graph()
        with pytest.raises(TypeError, match="stream_mode"):
            await graph.astream_events(
                {"message": "hello"},
                version="v3",
                stream_mode=["values"],
            )

    async def test_rejects_subgraphs(self) -> None:
        graph = _build_context_reading_graph()
        with pytest.raises(TypeError, match="subgraphs"):
            await graph.astream_events(
                {"message": "hello"},
                version="v3",
                subgraphs=False,
            )
