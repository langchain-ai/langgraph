"""Tests for SubgraphTransformer.

Subscribes to `tasks` events and produces in-process `SubgraphRunStream`
handles backed by mini-muxes (built via `StreamMux._make_child`). The
synthetic-event tests isolate the inference / mini-mux wiring; the
real-graph tests exercise the end-to-end navigation path through
`stream_v2`.
"""

from __future__ import annotations

import operator
import time
from collections.abc import AsyncIterator
from functools import partial
from typing import Annotated, Any

import pytest
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.errors import GraphInterrupt
from langgraph.graph import StateGraph
from langgraph.pregel.main import _normalize_stream_transformer_factories
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.run_stream import (
    AsyncGraphRunStream,
    AsyncSubgraphRunStream,
    GraphRunStream,
    SubgraphRunStream,
)
from langgraph.stream.transformers import (
    LifecycleTransformer,
    MessagesTransformer,
    SubgraphTransformer,
    ValuesTransformer,
)

TS = int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tasks_start(
    namespace: list[str],
    *,
    task_id: str,
    name: str,
) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": {
                "id": task_id,
                "name": name,
                "input": None,
                "triggers": [],
            },
        },
    }


def _tasks_result(
    namespace: list[str],
    *,
    task_id: str,
    name: str,
    error: str | None = None,
    interrupts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": {
                "id": task_id,
                "name": name,
                "error": error,
                "interrupts": interrupts or [],
                "result": {},
            },
        },
    }


def _native_factories() -> list[Any]:
    """Mirror the factory list `Pregel.stream_v2` registers."""
    return [
        ValuesTransformer,
        MessagesTransformer,
        LifecycleTransformer,
        SubgraphTransformer,
    ]


def _stream_part(
    method: str,
    namespace: tuple[str, ...],
    data: Any,
) -> dict[str, Any]:
    return {"type": method, "ns": namespace, "data": data}


async def _astream_parts(*parts: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    for part in parts:
        yield part


def _arm(mux: StreamMux) -> None:
    """Pre-subscribe every projection in the mux so synthetic pushes accumulate.

    Real consumer code subscribes by iterating the projection; tests
    inspect `_items` directly, so the lazy-subscribe gate has to be
    flipped manually before any synthetic events are pushed.
    """
    mux._events._subscribed = True
    for value in mux.extensions.values():
        if hasattr(value, "_subscribed"):  # EventLog
            value._subscribed = True
        elif hasattr(value, "_log"):  # StreamChannel
            value._log._subscribed = True


def _arm_recursive(mux: StreamMux) -> None:
    """Arm `mux` and every mini-mux currently held by SubgraphTransformer handles.

    Mini-muxes are created during `mux.push(...)` when a new direct
    child is discovered. Tests must call this after each push that
    might have created a new mini-mux so subsequent pushes' projection
    side effects accumulate (rather than dropping silently against an
    unsubscribed log).
    """
    _arm(mux)
    for handle in _subgraph_transformer(mux)._handles.values():
        if handle._mux is not None:
            _arm_recursive(handle._mux)


def _build_root_mux(*, scope: tuple[str, ...] = ()) -> StreamMux:
    mux = StreamMux(
        factories=_native_factories(),
        scope=scope,
        is_async=False,
    )
    _arm(mux)
    return mux


def _subgraph_transformer(mux: StreamMux) -> SubgraphTransformer:
    transformer = mux.transformer_by_key("subgraphs")
    assert isinstance(transformer, SubgraphTransformer)
    return transformer


def _drain_subgraphs(mux: StreamMux) -> list[SubgraphRunStream]:
    return list(_subgraph_transformer(mux)._log._items)


def _child_mux(handle: SubgraphRunStream | AsyncSubgraphRunStream) -> StreamMux:
    assert handle._mux is not None
    return handle._mux


def _event_items(mux: StreamMux) -> list[ProtocolEvent]:
    return list(mux._events._items)


def _lifecycle_payloads(mux: StreamMux) -> list[dict[str, Any]]:
    lifecycle_t = mux.transformer_by_key("lifecycle")
    assert isinstance(lifecycle_t, LifecycleTransformer)
    return list(lifecycle_t._channel._log._items)


# ---------------------------------------------------------------------------
# Synthetic-event tests
# ---------------------------------------------------------------------------


def test_handle_created_on_first_direct_child_task() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    [handle] = _drain_subgraphs(mux)
    assert handle.path == ("agent:abc",)
    assert handle.graph_name == "agent"
    assert handle.trigger_call_id == "abc"
    assert handle.status == "started"
    _child_mux(handle)  # mini-mux backed


def test_handle_status_completes_on_parent_result() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent"))

    [handle] = _drain_subgraphs(mux)
    assert handle.status == "completed"
    assert handle.error is None


def test_handle_status_failed_with_error() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent", error="boom"))

    [handle] = _drain_subgraphs(mux)
    assert handle.status == "failed"
    assert handle.error == "boom"


def test_handle_status_interrupted() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(
        _tasks_result(
            [],
            task_id="abc",
            name="agent",
            interrupts=[{"value": "pause"}],
        )
    )

    [handle] = _drain_subgraphs(mux)
    assert handle.status == "interrupted"


def test_grandchild_discovered_via_child_mini_mux() -> None:
    """Each mini-mux owns its own scope; grandchildren live on the child handle."""
    mux = _build_root_mux()
    # Direct child started — creates the mini-mux.
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    # Pre-subscribe the freshly-created mini-mux so subsequent
    # forwarded events land on its projections (consumer would
    # subscribe naturally by iterating handle.subgraphs, but the
    # test inspects `_items` directly).
    _arm_recursive(mux)
    # Grandchild's first task event flows down into the child mini-mux.
    mux.push(_tasks_start(["agent:abc", "tool:def"], task_id="t2", name="deep"))

    [child_handle] = _drain_subgraphs(mux)
    assert child_handle.path == ("agent:abc",)
    # The grandchild appears on the CHILD'S subgraphs projection.
    grandchildren = list(child_handle.subgraphs._items)
    assert len(grandchildren) == 1
    assert grandchildren[0].path == ("agent:abc", "tool:def")


def test_finalize_completes_open_handles() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.close()
    [handle] = _drain_subgraphs(mux)
    assert handle.status == "completed"


def test_fail_marks_open_handles_interrupted_for_graph_interrupt() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.fail(GraphInterrupt())
    [handle] = _drain_subgraphs(mux)
    assert handle.status == "interrupted"


def test_fail_marks_open_handles_failed_for_other_errors() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.fail(RuntimeError("boom"))
    [handle] = _drain_subgraphs(mux)
    assert handle.status == "failed"
    assert handle.error == "boom"


def test_child_mux_requires_factories() -> None:
    """A mux constructed only from `transformers=` can't clone factories."""
    transformer = SubgraphTransformer()
    mux = StreamMux(transformers=[transformer], is_async=False)
    with pytest.raises(RuntimeError, match="factories"):
        mux._make_child(("anything",))


def test_subgraph_and_lifecycle_agree_on_terminal_status() -> None:
    """Both transformers consume the same tasks signal — no drift."""
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent", error="boom"))

    [handle] = _drain_subgraphs(mux)
    payloads = _lifecycle_payloads(mux)
    assert handle.status == "failed"
    assert payloads[-1]["event"] == "failed"
    assert handle.error == payloads[-1]["error"]


def test_required_stream_modes_declared() -> None:
    assert SubgraphTransformer.required_stream_modes == ("tasks",)


def test_tasks_events_suppressed_from_main_log() -> None:
    """Tasks events are folded into discovery and don't appear on the main log."""
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent"))

    methods = [evt["method"] for evt in _event_items(mux)]
    assert "tasks" not in methods


class _ChildEventObserver(StreamTransformer):
    """Records child-scope event identity without mutating it."""

    records: list[tuple[tuple[str, ...], int, int, bool]] = []

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        if self.scope and event["method"] == "values":
            self.records.append(
                (
                    self.scope,
                    id(event),
                    id(event["params"]["data"]),
                    "seq" in event,
                )
            )
        return True


def test_child_forwarding_reuses_event_without_assigning_seq() -> None:
    _ChildEventObserver.records = []
    mux = StreamMux(
        factories=[
            ValuesTransformer,
            MessagesTransformer,
            LifecycleTransformer,
            SubgraphTransformer,
            _ChildEventObserver,
        ],
        is_async=False,
    )
    _arm(mux)
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    data = {"x": 1}
    event: ProtocolEvent = {
        "type": "event",
        "method": "values",
        "params": {
            "namespace": ["agent:abc"],
            "timestamp": TS,
            "data": data,
        },
    }
    mux.push(event)

    assert _ChildEventObserver.records == [(("agent:abc",), id(event), id(data), False)]
    [root_event] = [evt for evt in _event_items(mux) if evt["method"] == "values"]
    assert root_event is event
    assert "seq" in root_event


class _AsyncProbeTransformer(StreamTransformer):
    """Async-only transformer used to verify mini-mux async dispatch."""

    required_stream_modes = ("tasks",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.seen: list[tuple[str, ...]] = []
        self.finalized = False
        self.failed: BaseException | None = None

    def init(self) -> dict[str, Any]:
        return {"async_probe": self}

    async def aprocess(self, event: ProtocolEvent) -> bool:
        self.seen.append(tuple(event["params"]["namespace"]))
        return True

    async def afinalize(self) -> None:
        self.finalized = True

    async def afail(self, err: BaseException) -> None:
        self.failed = err


@pytest.mark.anyio
async def test_async_child_mini_mux_uses_async_lane() -> None:
    mux = StreamMux(
        factories=[
            ValuesTransformer,
            MessagesTransformer,
            LifecycleTransformer,
            SubgraphTransformer,
            _AsyncProbeTransformer,
        ],
        is_async=True,
    )
    await mux.apush(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    handle = _subgraph_transformer(mux)._handles[("agent:abc",)]
    assert isinstance(handle, AsyncSubgraphRunStream)
    probe = _child_mux(handle).transformer_by_key("async_probe")
    assert isinstance(probe, _AsyncProbeTransformer)
    assert probe.seen == [("agent:abc",)]

    await mux.apush(_tasks_result([], task_id="abc", name="agent"))
    assert probe.finalized is True


@pytest.mark.anyio
async def test_async_child_mini_mux_fail_uses_async_lane() -> None:
    mux = StreamMux(
        factories=[
            ValuesTransformer,
            MessagesTransformer,
            LifecycleTransformer,
            SubgraphTransformer,
            _AsyncProbeTransformer,
        ],
        is_async=True,
    )
    await mux.apush(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    handle = _subgraph_transformer(mux)._handles[("agent:abc",)]
    probe = _child_mux(handle).transformer_by_key("async_probe")
    assert isinstance(probe, _AsyncProbeTransformer)

    err = RuntimeError("boom")
    await mux.afail(err)
    assert probe.failed is err


class _StandardCtorTransformer(StreamTransformer):
    """Transformer class that inherits the standard scoped constructor."""

    def init(self) -> dict[str, Any]:
        return {"standard_ctor": self}

    def process(self, event: ProtocolEvent) -> bool:
        return True


class _ScopedTransformer(StreamTransformer):
    """Transformer class that uses the inherited scoped construction."""

    def init(self) -> dict[str, Any]:
        return {"scoped": self}

    def process(self, event: ProtocolEvent) -> bool:
        return True


class _ConfigurableFactoryTransformer(StreamTransformer):
    """Transformer built by a configured per-scope factory."""

    def __init__(self, scope: tuple[str, ...] = (), *, label: str) -> None:
        super().__init__(scope)
        self.label = label

    def init(self) -> dict[str, Any]:
        return {"configurable": self}

    def process(self, event: ProtocolEvent) -> bool:
        return True


class _ChildExploder(StreamTransformer):
    """Raise from child mini-muxes to verify errors propagate upstream."""

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        if self.scope and event["method"] == "values":
            raise RuntimeError("child boom")
        return True


class _ChildFinalizeExploder(StreamTransformer):
    """Raise from child mini-mux finalization."""

    supports_sync = True

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        return True

    def finalize(self) -> None:
        if self.scope:
            raise RuntimeError("child finalize boom")

    async def afinalize(self) -> None:
        if self.scope:
            raise RuntimeError("child afinalize boom")


def test_normalize_transformer_factories_supports_scoped_classes() -> None:
    factories = _normalize_stream_transformer_factories(
        [_StandardCtorTransformer, _ScopedTransformer]
    )

    standard_ctor = factories[0](("child",))
    scoped = factories[1](("child",))
    assert isinstance(standard_ctor, _StandardCtorTransformer)
    assert standard_ctor.scope == ("child",)
    assert isinstance(scoped, _ScopedTransformer)
    assert scoped.scope == ("child",)


def test_normalize_transformer_factories_supports_configured_factories() -> None:
    factories = _normalize_stream_transformer_factories(
        [partial(_ConfigurableFactoryTransformer, label="configured")]
    )

    built = factories[0](("child",))
    assert isinstance(built, _ConfigurableFactoryTransformer)
    assert built.label == "configured"
    assert built.scope == ("child",)


def test_normalize_transformer_factories_rejects_instances() -> None:
    with pytest.raises(TypeError, match="pre-built instance"):
        _normalize_stream_transformer_factories([_StandardCtorTransformer()])


def test_child_forwarding_errors_fail_sync_run() -> None:
    mux = StreamMux(
        factories=[
            ValuesTransformer,
            MessagesTransformer,
            LifecycleTransformer,
            SubgraphTransformer,
            _ChildExploder,
        ],
        is_async=False,
    )
    run = GraphRunStream(
        iter(
            [
                _stream_part(
                    "tasks",
                    ("agent:abc",),
                    {
                        "id": "t1",
                        "name": "tool",
                        "input": None,
                        "triggers": [],
                    },
                ),
                _stream_part("values", ("agent:abc",), {"x": 1}),
            ]
        ),
        mux,
    )

    handle = next(iter(run.subgraphs))
    assert handle.path == ("agent:abc",)
    with pytest.raises(RuntimeError, match="child boom"):
        _ = run.output
    assert run._mux._events._error is not None


@pytest.mark.anyio
async def test_child_forwarding_errors_fail_async_run() -> None:
    mux = StreamMux(
        factories=[
            ValuesTransformer,
            MessagesTransformer,
            LifecycleTransformer,
            SubgraphTransformer,
            _ChildExploder,
        ],
        is_async=True,
    )
    run = AsyncGraphRunStream(
        _astream_parts(
            _stream_part(
                "tasks",
                ("agent:abc",),
                {
                    "id": "t1",
                    "name": "tool",
                    "input": None,
                    "triggers": [],
                },
            ),
            _stream_part("values", ("agent:abc",), {"x": 1}),
        ),
        mux,
    )

    handle = await run.subgraphs.__aiter__().__anext__()
    assert handle.path == ("agent:abc",)
    with pytest.raises(RuntimeError, match="child boom"):
        await run.output()
    assert run._mux._events._error is not None


def test_child_finalize_errors_propagate_to_sync_run() -> None:
    mux = StreamMux(
        factories=[
            ValuesTransformer,
            MessagesTransformer,
            LifecycleTransformer,
            SubgraphTransformer,
            _ChildFinalizeExploder,
        ],
        is_async=False,
    )
    run = GraphRunStream(
        iter(
            [
                _stream_part(
                    "tasks",
                    ("agent:abc",),
                    {
                        "id": "t1",
                        "name": "tool",
                        "input": None,
                        "triggers": [],
                    },
                )
            ]
        ),
        mux,
    )

    with pytest.raises(RuntimeError, match="child finalize boom"):
        _ = run.output


@pytest.mark.anyio
async def test_child_finalize_errors_propagate_to_async_run() -> None:
    mux = StreamMux(
        factories=[
            ValuesTransformer,
            MessagesTransformer,
            LifecycleTransformer,
            SubgraphTransformer,
            _ChildFinalizeExploder,
        ],
        is_async=True,
    )
    run = AsyncGraphRunStream(
        _astream_parts(
            _stream_part(
                "tasks",
                ("agent:abc",),
                {
                    "id": "t1",
                    "name": "tool",
                    "input": None,
                    "triggers": [],
                },
            )
        ),
        mux,
    )

    with pytest.raises(RuntimeError, match="child afinalize boom"):
        await run.output()


# ---------------------------------------------------------------------------
# End-to-end real-graph tests
# ---------------------------------------------------------------------------


class _State(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _passthrough(state: _State) -> dict[str, Any]:
    return {"value": state["value"] + "!", "items": ["x"]}


def _make_two_level_nested() -> Any:
    """outer → middle → inner. Three Pregel instances, two nesting levels."""
    inner_b: StateGraph = StateGraph(_State, input_schema=_State)
    inner_b.add_node("inner_node", _passthrough)
    inner_b.add_edge(START, "inner_node")
    inner_b.add_edge("inner_node", END)
    inner = inner_b.compile()

    middle_b: StateGraph = StateGraph(_State, input_schema=_State)
    middle_b.add_node("inner", inner)
    middle_b.add_edge(START, "inner")
    middle_b.add_edge("inner", END)
    middle = middle_b.compile()

    outer_b: StateGraph = StateGraph(_State, input_schema=_State)
    outer_b.add_node("middle", middle)
    outer_b.add_edge(START, "middle")
    outer_b.add_edge("middle", END)
    return outer_b.compile()


def _item_node(item: str):
    def node(state: _State) -> dict[str, Any]:
        return {"items": [item]}

    return node


def _make_two_sibling_subgraphs() -> Any:
    """outer → one → two, where both nodes are compiled subgraphs."""
    one_b: StateGraph = StateGraph(_State, input_schema=_State)
    one_b.add_node("add_one", _item_node("one"))
    one_b.add_edge(START, "add_one")
    one_b.add_edge("add_one", END)
    one = one_b.compile()

    two_b: StateGraph = StateGraph(_State, input_schema=_State)
    two_b.add_node("add_two", _item_node("two"))
    two_b.add_edge(START, "add_two")
    two_b.add_edge("add_two", END)
    two = two_b.compile()

    outer_b: StateGraph = StateGraph(_State, input_schema=_State)
    outer_b.add_node("one", one)
    outer_b.add_node("two", two)
    outer_b.add_edge(START, "one")
    outer_b.add_edge("one", "two")
    outer_b.add_edge("two", END)
    return outer_b.compile()


def _failing_node(state: _State) -> dict[str, Any]:
    raise ValueError("child boom")


def _make_failing_nested() -> Any:
    inner_b: StateGraph = StateGraph(_State, input_schema=_State)
    inner_b.add_node("fail", _failing_node)
    inner_b.add_edge(START, "fail")
    inner_b.add_edge("fail", END)
    inner = inner_b.compile()

    outer_b: StateGraph = StateGraph(_State, input_schema=_State)
    outer_b.add_node("inner", inner)
    outer_b.add_edge(START, "inner")
    outer_b.add_edge("inner", END)
    return outer_b.compile()


def test_stream_v2_real_graph_yields_subgraph_handles() -> None:
    """Iterating `run.subgraphs` yields handles for direct-child subgraphs."""
    graph = _make_two_level_nested()
    run = graph.stream_v2({"value": "x", "items": []})

    handle_paths: list[tuple[str, ...]] = []
    final_status: dict[tuple[str, ...], str] = {}
    for handle in run.subgraphs:
        # Drill into the handle's projections inside the loop body so
        # the mini-mux is subscribed before the next pump cycle.
        list(handle.values)
        handle_paths.append(handle.path)
        final_status[handle.path] = handle.status

    assert len(handle_paths) == 1
    assert handle_paths[0][0].startswith("middle:")
    assert final_status[handle_paths[0]] == "completed"


def test_stream_v2_grandchild_visible_on_child_handle() -> None:
    """Drilling into `handle.subgraphs` surfaces nested grandchildren."""
    graph = _make_two_level_nested()
    run = graph.stream_v2({"value": "x", "items": []})

    grandchild_paths: list[tuple[str, ...]] = []
    middle_path: tuple[str, ...] | None = None
    for middle_handle in run.subgraphs:
        # Subscribe to grandchildren before the next pump cycle.
        for inner_handle in middle_handle.subgraphs:
            # Subscribe to inner.values so its mini-mux drains.
            list(inner_handle.values)
            grandchild_paths.append(inner_handle.path)
        middle_path = middle_handle.path

    assert middle_path is not None
    assert len(grandchild_paths) == 1
    inner_path = grandchild_paths[0]
    assert inner_path[1].startswith("inner:")
    assert inner_path[: len(middle_path)] == middle_path


def test_subgraph_output_stops_at_own_terminal_without_draining_siblings() -> None:
    """A handle's `output` must not pump past its terminal event.

    If it over-pumps the root run, the second sibling handle is yielded
    only after it has already completed, so subscribing to `values`
    inside the loop body misses its events.
    """
    graph = _make_two_sibling_subgraphs()
    run = graph.stream_v2({"value": "x", "items": []})

    paths: list[tuple[str, ...]] = []
    second_values: list[dict[str, Any]] = []
    for handle in run.subgraphs:
        paths.append(handle.path)
        if handle.graph_name == "one":
            assert handle.output is not None
            assert handle.status == "completed"
        elif handle.graph_name == "two":
            second_values = list(handle.values)

    assert [path[0].split(":", 1)[0] for path in paths] == ["one", "two"]
    assert second_values
    assert second_values[-1]["items"] == ["one", "two"]


def test_aborted_subgraph_handle_does_not_fail_parent_forwarding() -> None:
    graph = _make_two_sibling_subgraphs()
    run = graph.stream_v2({"value": "x", "items": []})

    seen: list[str | None] = []
    for handle in run.subgraphs:
        seen.append(handle.graph_name)
        if handle.graph_name == "one":
            # Subscribe before aborting to ensure forwarding into the
            # closed mini-mux would have raised without the closed check.
            iter(handle.values)
            handle.abort()
        elif handle.graph_name == "two":
            assert list(handle.values)

    assert seen == ["one", "two"]


def test_failed_subgraph_output_raises_terminal_error() -> None:
    graph = _make_failing_nested()
    run = graph.stream_v2({"value": "x", "items": []})

    handle = next(iter(run.subgraphs))
    with pytest.raises(RuntimeError, match="child boom"):
        _ = handle.output
    assert handle.status == "failed"
    assert handle.error == "child boom"
