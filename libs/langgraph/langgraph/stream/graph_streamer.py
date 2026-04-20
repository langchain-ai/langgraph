from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, ClassVar, cast

from langchain_core.runnables import RunnableConfig

from langgraph._internal._constants import CONF, CONFIG_KEY_STREAM_MESSAGES_V2
from langgraph.pregel import Pregel
from langgraph.stream._mux import StreamMux, TransformerFactory
from langgraph.stream._types import StreamTransformer
from langgraph.stream.run_stream import AsyncGraphRunStream, GraphRunStream
from langgraph.stream.transformers import (
    MessagesTransformer,
    SubgraphTransformer,
    ValuesTransformer,
)
from langgraph.types import All, StreamMode


def _coerce_factories(
    transformers: list[TransformerFactory] | None,
) -> list[TransformerFactory]:
    """Validate caller-supplied factories.

    Each factory must be callable — a transformer class (which accepts
    a positional `scope` argument) or a callable that returns a fresh
    instance per scope. Already-built instances are rejected because
    they can't be re-instantiated in subgraph mini-muxes, which would
    silently disable per-subagent scoping for that transformer.
    """
    coerced: list[TransformerFactory] = []
    for item in transformers or ():
        if isinstance(item, StreamTransformer):
            raise TypeError(
                "GraphStreamer.transformers takes factories, not "
                "pre-built instances. Pass the transformer class "
                "(e.g. `MyTransformer`) or a callable taking `scope` "
                "(e.g. `lambda scope: MyTransformer(scope, foo=...)`), "
                "so fresh instances can be built for each subgraph."
            )
        if not callable(item):
            raise TypeError(
                f"GraphStreamer.transformers entries must be callable; "
                f"got {type(item).__name__}."
            )
        coerced.append(item)
    return coerced


def _merge_v2_messages_flag(
    config: RunnableConfig | None,
) -> RunnableConfig:
    """Return a config with the v2 messages flag set in `configurable`.

    Signals to pregel that `stream_mode="messages"` should attach
    `StreamMessagesHandlerV2` for this call so invoke-time model runs
    route through the v2 event generator and their protocol events
    reach the messages channel.
    """
    merged: RunnableConfig = dict(config or {})  # type: ignore[assignment]
    configurable = dict(merged.get(CONF) or {})
    configurable[CONFIG_KEY_STREAM_MESSAGES_V2] = True
    merged[CONF] = configurable
    return merged


def _collect_stream_modes(mux: StreamMux) -> list[StreamMode]:
    """Return the union of `required_stream_modes` across registered transformers.

    Transformers declare the stream modes they need to function, and
    `GraphStreamer` asks the graph for exactly that union — no
    hardcoded default set. If zero transformers are registered (or none
    declares a given mode), the graph does not stream events for that
    mode; consumers that want raw `custom` / `updates` / `checkpoints`
    / `tasks` / `debug` visibility must register a transformer that
    declares those modes as required.
    """
    modes: set[str] = set()
    for transformer in mux._transformers:
        modes.update(transformer.required_stream_modes)
    return cast("list[StreamMode]", list(modes))


class GraphStreamer:
    """Wrap a compiled graph with ergonomic streaming projections.

    Example:
        ```python
        streamer = GraphStreamer(graph)

        # Sync
        run = streamer.stream(input_data)
        for state in run.values:
            print(state)
        output = run.output

        # Async — terminal accessors are methods so a missing `await`
        # fails loudly instead of silently yielding a coroutine.
        run = await streamer.astream(input_data)
        async for state in run.values:
            print(state)
        output = await run.output()
        ```

    Subclassing hooks:

    - `builtin_factories`: tuple of transformer factories registered on
      every run. Override to append domain-specific transformers
      without the caller threading them through `transformers=`. An
      `AgentStreamer` subclass, for example, can append
      `ToolCallTransformer` so every agent run exposes `run.tool_calls`
      by default.
    - `_make_run_stream(...)` / `_make_async_run_stream(...)`: factory
      hooks that return the run stream instance. Override to return a
      subclass of `GraphRunStream` / `AsyncGraphRunStream` with typed
      accessors over the projections contributed by the extra
      transformers.

    See the end of this module for a minimal subclassing sketch.
    """

    builtin_factories: ClassVar[tuple[TransformerFactory, ...]] = (
        ValuesTransformer,
        MessagesTransformer,
        SubgraphTransformer,
    )
    """Factories registered on every run before caller-supplied transformers.

    Subclasses append to this tuple to bundle domain-specific
    transformers — for example, an `AgentStreamer` appends
    `ToolCallTransformer` so every agent run exposes `run.tool_calls`
    without the caller opting in.
    """

    def __init__(self, graph: Pregel) -> None:
        """Initialize the streamer.

        Args:
            graph: A compiled LangGraph graph to stream from.
        """
        self._graph = graph

    def _build_factories(
        self,
        transformers: list[TransformerFactory] | None,
    ) -> list[TransformerFactory]:
        """Return `builtin_factories` plus the caller's transformers.

        Subclasses rarely override this directly — extend
        `builtin_factories` instead. Override only when you need to
        inspect or re-order the caller's transformers (e.g. to inject
        a transformer *after* user-supplied ones).
        """
        return [*self.builtin_factories, *_coerce_factories(transformers)]

    def _make_run_stream(
        self,
        graph_iter: Iterator[Any],
        mux: StreamMux,
    ) -> GraphRunStream:
        """Construct the sync run stream returned from `stream()`.

        Override in a subclass to return a `GraphRunStream` subclass
        with typed accessors over the extra projections contributed by
        the subclass's `builtin_factories` (e.g. `AgentRunStream` with
        a typed `.tool_calls`).
        """
        return GraphRunStream(graph_iter, mux)

    def _make_async_run_stream(
        self,
        graph_aiter: AsyncIterator[Any],
        mux: StreamMux,
    ) -> AsyncGraphRunStream:
        """Async counterpart to `_make_run_stream`."""
        return AsyncGraphRunStream(graph_aiter, mux)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        transformers: list[TransformerFactory] | None = None,
    ) -> GraphRunStream:
        """Start a sync streaming run.

        Returns a GraphRunStream immediately. The caller's iteration on
        any projection drives the graph forward — no background thread
        is used. This matches v1's model where the caller's `for` loop
        is the pump.

        Args:
            input: Graph input.
            config: Optional runnable config forwarded to the graph.
            interrupt_before: Nodes to interrupt before, if any.
            interrupt_after: Nodes to interrupt after, if any.
            transformers: User transformers appended after
                `self.builtin_factories`.

        Returns:
            A `GraphRunStream` the caller can iterate to drive the run.
            Subclasses may narrow this to a `GraphRunStream` subtype via
            `_make_run_stream`.
        """
        mux = StreamMux(
            factories=self._build_factories(transformers),
            is_async=False,
        )
        stream_modes = _collect_stream_modes(mux)

        graph_iter = iter(
            self._graph.stream(
                input,
                _merge_v2_messages_flag(config),
                stream_mode=stream_modes,
                subgraphs=True,
                version="v2",
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
            )
        )

        return self._make_run_stream(graph_iter, mux)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        transformers: list[TransformerFactory] | None = None,
    ) -> AsyncGraphRunStream:
        """Start an async streaming run.

        Returns an AsyncGraphRunStream immediately. The caller's
        iteration on any projection drives the graph forward — there
        is no background task. Concurrent consumers share a
        single-flight pump via an internal `asyncio.Lock`.

        Args:
            input: Graph input.
            config: Optional runnable config forwarded to the graph.
            interrupt_before: Nodes to interrupt before, if any.
            interrupt_after: Nodes to interrupt after, if any.
            transformers: User transformers appended after
                `self.builtin_factories`.

        Returns:
            An `AsyncGraphRunStream` whose projections can be awaited
            concurrently; subclasses may narrow this via
            `_make_async_run_stream`.
        """
        mux = StreamMux(
            factories=self._build_factories(transformers),
            is_async=True,
        )
        stream_modes = _collect_stream_modes(mux)

        graph_aiter = self._graph.astream(
            input,
            _merge_v2_messages_flag(config),
            stream_mode=stream_modes,
            subgraphs=True,
            version="v2",
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        ).__aiter__()

        return self._make_async_run_stream(graph_aiter, mux)
