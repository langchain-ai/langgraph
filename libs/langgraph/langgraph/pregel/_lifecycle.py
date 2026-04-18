from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, TypeVar, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from langgraph._internal._constants import NS_SEP
from langgraph.errors import GraphInterrupt
from langgraph.pregel.protocol import StreamChunk

try:
    from langchain_core.tracers._streaming import _StreamingCallbackHandler
except ImportError:
    _StreamingCallbackHandler = object  # type: ignore[assignment,misc]


T = TypeVar("T")


_LANGGRAPH_SENTINEL_NODES = frozenset({"__start__", "__end__"})


def _is_nested_pregel_start(
    name: str | None,
    metadata: dict[str, Any] | None,
    parent_run_id: UUID | None,
    task_run_ids: set[UUID],
) -> bool:
    """Recognize a nested `Pregel` invocation from its `on_chain_start` metadata.

    When a compiled graph is added as a node, pregel fires two
    `on_chain_start` callbacks at that task: first for the node chain
    (whose `name` matches `metadata["langgraph_node"]`) and second for
    the inner `Pregel` chain (whose `name` is the graph's `name`, not
    the node name). Both share the same `langgraph_checkpoint_ns`.

    Primary signal: a `langgraph_checkpoint_ns` is set AND `name`
    differs from the owning task's `langgraph_node`. This covers the
    common case where the compiled subgraph's name differs from the
    node name it was registered under.

    Fallback for name collisions (subgraph compiled with
    `name == node_name`): the inner `Pregel` start's `parent_run_id`
    is the run_id of the node chain's start event, which the handler
    records in `task_run_ids` on the first start. Matching
    `parent_run_id` to that set identifies the second start as the
    nested `Pregel` even when names coincide.

    Regular node chains are skipped; the root `Pregel` (which has no
    `langgraph_node` metadata) isn't observed by this handler because
    the root's start fires before the handler is attached.

    Metadata-based detection is used because `on_chain_start`'s
    `serialized` argument is `None` for compiled graphs in this
    version of langchain-core, so class-based detection via
    `serialized["id"]` isn't available.

    Sentinel nodes (`__start__` / `__end__`) are excluded: conditional
    edges from `START` fire an `on_chain_start` with `lg_node=__start__`
    and the router function's name as `name`, which would otherwise
    match the discriminator without representing an actual nested
    `Pregel`.

    Args:
        name: The `name` kwarg from `on_chain_start`.
        metadata: The `metadata` kwarg from `on_chain_start`.
        parent_run_id: The `parent_run_id` kwarg from `on_chain_start`.
        task_run_ids: The set of run_ids the handler has already seen
            as node-chain starts (i.e. `name == langgraph_node`).
    """
    if not metadata:
        return False
    if not metadata.get("langgraph_checkpoint_ns"):
        return False
    lg_node = metadata.get("langgraph_node")
    if lg_node is None or lg_node in _LANGGRAPH_SENTINEL_NODES:
        return False
    if name != lg_node:
        return True
    # Name collision fallback: the inner Pregel's parent_run_id is
    # the node chain's run_id, which we recorded when that node
    # chain's start fired.
    return parent_run_id is not None and parent_run_id in task_run_ids


class StreamLifecycleHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    """Callback handler that emits subgraph lifecycle events on the stream.

    Pushes `LifecycleData`-shaped payloads onto the pregel stream under
    the `"lifecycle"` mode, keyed by the subgraph's namespace tuple.
    Drives the `started` → `running` → `completed` / `failed` /
    `interrupted` state machine.

    The handler is attached to `run_manager.inheritable_handlers` inside
    a `Pregel.stream` / `astream` call, so it sees callbacks for every
    descendant chain (nodes, nested `Pregel` subgraphs) but *not* for
    the root `Pregel` whose start event has already fired. The root's
    `started` event is emitted eagerly at construction; its terminal
    state is emitted by `SubgraphTransformer.finalize` / `fail`.

    `run_inline = True` keeps event ordering deterministic.
    """

    run_inline = True

    def __init__(
        self,
        stream: Callable[[StreamChunk], None],
        *,
        root_graph_name: str | None = None,
    ) -> None:
        """Initialize the handler and emit the root graph's `started` event.

        Args:
            stream: Callable that accepts a `StreamChunk` tuple
                `(namespace, mode, payload)` and enqueues it.
            root_graph_name: The root `Pregel` instance's `name`, emitted
                with the root's `started` lifecycle payload.
        """
        self.stream = stream
        # Namespaces awaiting the started→running transition.
        self._pending_running: set[tuple[str, ...]] = set()
        # run_id → subgraph namespace; populated only for Pregel chains.
        self._run_to_ns: dict[UUID, tuple[str, ...]] = {}
        # run_ids of node-chain starts (name == langgraph_node); used
        # as the parent_run_id fallback when a subgraph's name equals
        # its node name. Cleared as each chain ends.
        self._task_run_ids: set[UUID] = set()

        root_payload: dict[str, Any] = {"event": "started"}
        if root_graph_name is not None:
            root_payload["graph_name"] = root_graph_name
        self.stream(((), "lifecycle", root_payload))
        self._pending_running.add(())

    @staticmethod
    def _subgraph_ns_from_metadata(metadata: dict[str, Any] | None) -> tuple[str, ...]:
        """Return the running subgraph's own namespace from task metadata.

        For a nested `Pregel` invoked as a node, `langgraph_checkpoint_ns`
        ends at the node segment (no inner task appended yet), so
        splitting on `NS_SEP` gives the subgraph's own namespace.
        """
        if not metadata:
            return ()
        nskey = metadata.get("langgraph_checkpoint_ns")
        if not nskey:
            return ()
        return tuple(cast(str, nskey).split(NS_SEP))

    @staticmethod
    def _containing_ns_from_metadata(
        metadata: dict[str, Any] | None,
    ) -> tuple[str, ...]:
        """Return the namespace of the subgraph that contains this task.

        For an inner task with `langgraph_checkpoint_ns`
        `"seg_a|seg_b"`, the containing subgraph is `("seg_a",)`.
        """
        if not metadata:
            return ()
        nskey = metadata.get("langgraph_checkpoint_ns")
        if not nskey:
            return ()
        return tuple(cast(str, nskey).split(NS_SEP))[:-1]

    @staticmethod
    def _trigger_call_id(metadata: dict[str, Any] | None) -> str | None:
        """Extract `trigger_call_id` from task metadata if present.

        The task that spawned a nested `Pregel` has its task id encoded
        in `langgraph_checkpoint_ns`'s last segment as
        `node_name:task_id`. Returns the `task_id` portion, which
        parents can correlate with their `tools` / `tasks` events.
        """
        if not metadata:
            return None
        nskey = cast(str | None, metadata.get("langgraph_checkpoint_ns"))
        if not nskey:
            return None
        last = nskey.split(NS_SEP)[-1]
        _, sep, task_id = last.rpartition(":")
        return task_id if sep else None

    def _emit(self, ns: tuple[str, ...], payload: dict[str, Any]) -> None:
        self.stream((ns, "lifecycle", payload))

    def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """Pass-through — required by the `_StreamingCallbackHandler` protocol.

        Returns the iterator unchanged. A missing implementation lets
        langchain's default `Protocol` body return `None`, which breaks
        the `_consume_aiter` code path in `_runnable.py:900`.
        """
        return output

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """Pass-through — sync counterpart to `tap_output_aiter`."""
        return output

    def _fire_running_if_pending(self, ns: tuple[str, ...]) -> None:
        if ns in self._pending_running:
            self._pending_running.discard(ns)
            self._emit(ns, {"event": "running"})

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        # Any descendant activity transitions the containing subgraph to running.
        containing = self._containing_ns_from_metadata(metadata)
        self._fire_running_if_pending(containing)

        name = cast(str | None, kwargs.get("name"))
        lg_node = (metadata or {}).get("langgraph_node")

        # Record node-chain starts so the name-collision fallback in
        # `_is_nested_pregel_start` can match the inner Pregel's
        # parent_run_id to them.
        if (
            lg_node is not None
            and lg_node not in _LANGGRAPH_SENTINEL_NODES
            and name == lg_node
        ):
            self._task_run_ids.add(run_id)

        if not _is_nested_pregel_start(
            name, metadata, parent_run_id, self._task_run_ids
        ):
            return

        ns = self._subgraph_ns_from_metadata(metadata)
        if not ns:
            return

        self._run_to_ns[run_id] = ns
        payload: dict[str, Any] = {"event": "started"}
        if name:
            payload["graph_name"] = name
        trigger_call_id = self._trigger_call_id(metadata)
        if trigger_call_id:
            payload["trigger_call_id"] = trigger_call_id
        self._emit(ns, payload)
        self._pending_running.add(ns)

    def on_chain_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._task_run_ids.discard(run_id)
        ns = self._run_to_ns.pop(run_id, None)
        if ns is None:
            return
        # Ensure started→running fired even for empty subgraphs.
        if ns in self._pending_running:
            self._pending_running.discard(ns)
            self._emit(ns, {"event": "running"})
        self._emit(ns, {"event": "completed"})

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._task_run_ids.discard(run_id)
        ns = self._run_to_ns.pop(run_id, None)
        if ns is None:
            return
        self._pending_running.discard(ns)
        if isinstance(error, GraphInterrupt):
            self._emit(ns, {"event": "interrupted"})
        else:
            self._emit(ns, {"event": "failed", "error": str(error)})
