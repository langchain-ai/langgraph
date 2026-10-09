"""Tests that ``(a)stream_events(version="v3")`` forwards extra kwargs to the
underlying ``(a)stream`` call, and rejects the kwargs v3 owns internally.

Background: prior to this change the v3 dispatcher silently dropped ``**kwargs``
on the v3 branch while forwarding them on v1/v2, so callers passing e.g.
``context=...`` saw their value disappear with no error. v3 now forwards
caller kwargs to the inner ``(a)stream`` call but rejects ``stream_mode`` and
``subgraphs`` since v3 owns them (``stream_mode`` is built from the
transformer mux; ``subgraphs`` is forced True so nested namespaces flow
through scoped muxes).

A second regression is pinned here: #7677 (first released in 1.2.0a3)
declared `interrupt_before` / `interrupt_after` / `control` as named
parameters on the `Pregel.stream_events` / `astream_events` dispatchers
but forwarded them only on the v3 branch, silently dropping them on v1/v2
(where they had reached `(a)stream` through `**kwargs` before).
`TestAstreamKwargsForwardedOnEveryVersion` and friends pin that they
reach `(a)stream` on every version, with the v1/v2 passthrough semantics
restored (exactly what the caller passed, including an explicit `None`)
and v3's explicit-default semantics preserved.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.errors import GraphDrained
from langgraph.graph import StateGraph
from langgraph.runtime import RunControl, Runtime

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


_KWARG_NAMES = ("control", "interrupt_before", "interrupt_after")


def _build_two_step_graph(
    first: Callable[[_State], dict[str, Any]] | None = None,
) -> Any:
    """A `first -> second` graph with a checkpointer, for interrupt/drain tests."""

    def default_first(state: _State) -> dict[str, Any]:
        return {"message": state["message"] + " first"}

    def second(state: _State) -> dict[str, Any]:
        return {"message": state["message"] + " second"}

    builder = StateGraph(_State)
    builder.add_node("first", first or default_first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile(checkpointer=InMemorySaver())


async def _drive_astream_events(
    graph: Any, config: dict[str, Any], version: str, **kwargs: Any
) -> None:
    """Consume an astream_events run for `version` to completion."""
    if version == "v3":
        run = await graph.astream_events(
            {"message": "hi"}, config, version="v3", **kwargs
        )
        await run.output()
    else:
        async for _ in graph.astream_events(
            {"message": "hi"}, config, version=version, **kwargs
        ):
            pass


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore:astream_events version='v1' is deprecated")
@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
class TestAstreamKwargsForwardedOnEveryVersion:
    """`interrupt_before`/`interrupt_after`/`control` reach `astream` on every
    version.

    Regression test for #7677 (first released in 1.2.0a3): the dispatchers
    captured these parameters as named arguments but forwarded them only on
    the v3 branch, silently dropping them on v1/v2.
    """

    async def test_interrupt_before(self, version: str) -> None:
        graph = _build_two_step_graph()
        config = {"configurable": {"thread_id": "ib"}}
        await _drive_astream_events(graph, config, version, interrupt_before=["second"])
        state = await graph.aget_state(config)
        assert state.next == ("second",)
        assert state.values == {"message": "hi first"}

    async def test_interrupt_after(self, version: str) -> None:
        graph = _build_two_step_graph()
        config = {"configurable": {"thread_id": "ia"}}
        await _drive_astream_events(graph, config, version, interrupt_after=["first"])
        state = await graph.aget_state(config)
        assert state.next == ("second",)
        assert state.values == {"message": "hi first"}

    async def test_pre_drained_control(self, version: str) -> None:
        graph = _build_two_step_graph()
        config = {"configurable": {"thread_id": "drain"}}
        control = RunControl()
        control.request_drain("sigterm")
        with pytest.raises(GraphDrained, match="sigterm"):
            await _drive_astream_events(graph, config, version, control=control)


class TestStreamEventsV3SyncInterrupts:
    """Sync v3 static interrupts reach `stream()` after the kwargs rewire."""

    @pytest.mark.parametrize(
        ("kwarg", "node"),
        [("interrupt_before", "second"), ("interrupt_after", "first")],
    )
    def test_static_interrupt(self, kwarg: str, node: str) -> None:
        graph = _build_two_step_graph()
        config = {"configurable": {"thread_id": "sync"}}
        run = graph.stream_events(
            {"message": "hi"}, config, version="v3", **{kwarg: [node]}
        )
        list(run.values)
        state = graph.get_state(config)
        assert state.next == ("second",)
        assert state.values == {"message": "hi first"}


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore:astream_events version='v1' is deprecated")
@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
class TestAstreamMidRunDrain:
    """A drain requested from inside a node propagates out of `astream_events`.

    This is the graceful-shutdown scenario: `request_drain()` called while
    the run is in flight (e.g. from a signal handler), with v1/v2 running
    inside core's event-stream task. The caller's own `RunControl` is used
    (the drain reason proves identity), `GraphDrained` propagates to the
    consumer, and the checkpoint keeps the pending step.
    """

    async def test_drain_requested_inside_first_node(self, version: str) -> None:
        control = RunControl()

        def first(state: _State) -> dict[str, Any]:
            control.request_drain("sigterm-mid")
            return {"message": state["message"] + " first"}

        graph = _build_two_step_graph(first)
        config = {"configurable": {"thread_id": "midrun"}}

        with pytest.raises(GraphDrained, match="sigterm-mid"):
            await _drive_astream_events(graph, config, version, control=control)
        state = await graph.aget_state(config)
        assert state.next == ("second",)
        assert state.values == {"message": "hi first"}


def _record_astream_kwargs(graph: Any) -> list[dict[str, Any]]:
    """Patch `graph.astream` to record the kwargs each call receives."""
    received: list[dict[str, Any]] = []
    original = graph.astream

    async def recording_astream(
        input: Any, config: Any = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        received.append(kwargs)
        async for chunk in original(input, config, **kwargs):
            yield chunk

    graph.astream = recording_astream  # type: ignore[method-assign]
    return received


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore:astream_events version='v1' is deprecated")
@pytest.mark.parametrize("version", ["v1", "v2"])
class TestAstreamV1V2KwargsPassthrough:
    """v1/v2 forward to `astream` exactly what the caller passed.

    Pre-#7677 semantics: an explicit `None` is forwarded as `None`, and an
    omitted argument is not forwarded at all (so an override's own default
    would apply).
    """

    @pytest.mark.parametrize("name", _KWARG_NAMES)
    async def test_passed_values_reach_astream(self, version: str, name: str) -> None:
        graph = _build_two_step_graph()
        received = _record_astream_kwargs(graph)
        value: Any = RunControl() if name == "control" else ["second"]
        await _drive_astream_events(
            graph, {"configurable": {"thread_id": "rec"}}, version, **{name: value}
        )
        assert len(received) == 1
        assert received[0][name] == value
        if name == "control":
            assert received[0][name] is value

    @pytest.mark.parametrize("name", _KWARG_NAMES)
    async def test_explicit_none_is_forwarded(self, version: str, name: str) -> None:
        graph = _build_two_step_graph()
        received = _record_astream_kwargs(graph)
        await _drive_astream_events(
            graph, {"configurable": {"thread_id": "rec-none"}}, version, **{name: None}
        )
        assert len(received) == 1
        assert received[0][name] is None

    @pytest.mark.parametrize("name", _KWARG_NAMES)
    async def test_omitted_values_are_absent(self, version: str, name: str) -> None:
        graph = _build_two_step_graph()
        received = _record_astream_kwargs(graph)
        await _drive_astream_events(
            graph, {"configurable": {"thread_id": "rec-omit"}}, version
        )
        assert len(received) == 1
        assert name not in received[0]


@pytest.mark.anyio
class TestAstreamV3KwargsDefaults:
    """v3 keeps its since-inception explicit-default semantics (#7519).

    Omitted `interrupt_before`/`interrupt_after`/`control` are supplied to
    `astream` as `None`; passed values are forwarded as-is.
    """

    async def test_omitted_values_arrive_as_none(self) -> None:
        graph = _build_two_step_graph()
        received = _record_astream_kwargs(graph)
        await _drive_astream_events(
            graph, {"configurable": {"thread_id": "v3-rec"}}, "v3"
        )
        assert len(received) == 1
        assert received[0]["control"] is None
        assert received[0]["interrupt_before"] is None
        assert received[0]["interrupt_after"] is None

    @pytest.mark.parametrize("name", _KWARG_NAMES)
    async def test_passed_values_reach_astream(self, name: str) -> None:
        graph = _build_two_step_graph()
        received = _record_astream_kwargs(graph)
        value: Any = RunControl() if name == "control" else ["second"]
        await _drive_astream_events(
            graph, {"configurable": {"thread_id": "v3-rec-2"}}, "v3", **{name: value}
        )
        assert len(received) == 1
        assert received[0][name] == value
        if name == "control":
            assert received[0][name] is value


def _record_stream_kwargs(graph: Any) -> list[dict[str, Any]]:
    """Patch `graph.stream` to record the kwargs each call receives."""
    received: list[dict[str, Any]] = []
    original = graph.stream

    def recording_stream(input: Any, config: Any = None, **kwargs: Any) -> Any:
        received.append(kwargs)
        yield from original(input, config, **kwargs)

    graph.stream = recording_stream  # type: ignore[method-assign]
    return received


def _drive_stream_events_v3(graph: Any, config: dict[str, Any], **kwargs: Any) -> None:
    """Consume a sync v3 stream_events run to completion."""
    run = graph.stream_events({"message": "hi"}, config, version="v3", **kwargs)
    list(run.values)


class TestStreamEventsV3SyncKwargsDefaults:
    """Sync v3 keeps its since-inception explicit-default semantics (#7519).

    Mirror of `TestAstreamV3KwargsDefaults`: omitted
    `interrupt_before`/`interrupt_after`/`control` are supplied to `stream`
    as `None`; passed values are forwarded as-is. Pins the sync helper
    against a kwargs-only "simplification" that would change subclass
    default handling.
    """

    def test_omitted_values_arrive_as_none(self) -> None:
        graph = _build_two_step_graph()
        received = _record_stream_kwargs(graph)
        _drive_stream_events_v3(graph, {"configurable": {"thread_id": "s-rec"}})
        assert len(received) == 1
        assert received[0]["control"] is None
        assert received[0]["interrupt_before"] is None
        assert received[0]["interrupt_after"] is None

    @pytest.mark.parametrize("name", _KWARG_NAMES)
    def test_passed_values_reach_stream(self, name: str) -> None:
        graph = _build_two_step_graph()
        received = _record_stream_kwargs(graph)
        value: Any = RunControl() if name == "control" else ["second"]
        _drive_stream_events_v3(
            graph, {"configurable": {"thread_id": "s-rec-2"}}, **{name: value}
        )
        assert len(received) == 1
        assert received[0][name] == value
        if name == "control":
            assert received[0][name] is value
