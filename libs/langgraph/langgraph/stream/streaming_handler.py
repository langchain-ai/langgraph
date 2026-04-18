from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    transformers: list[StreamTransformer | TransformerFactory] | None,
) -> list[TransformerFactory]:
    """Normalize caller-supplied transformers into scope-taking factories.

    Accepts already-built instances (wrapped as single-use factories,
    with the caveat that they won't be re-instantiated in subgraph
    mini-muxes) or proper factories (classes / callables taking a
    scope). The built-in root transformers are always factories so
    they propagate into every subgraph scope automatically.
    """

    def _wrap_instance(t: StreamTransformer) -> TransformerFactory:
        # Single-use: only wires at root scope. A user that wants
        # subgraph propagation should pass the class (or a lambda).
        def _factory(_scope: tuple[str, ...]) -> StreamTransformer:
            return t

        return _factory

    coerced: list[TransformerFactory] = []
    for item in transformers or ():
        if isinstance(item, StreamTransformer):
            coerced.append(_wrap_instance(item))
        else:
            coerced.append(item)
    return coerced


_BUILTIN_FACTORIES: list[TransformerFactory] = [
    ValuesTransformer,
    MessagesTransformer,
    SubgraphTransformer,
]


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


# All stream modes to request from the graph.
STREAM_V2_MODES: list[StreamMode] = [
    "values",
    "updates",
    "messages",
    "custom",
    "checkpoints",
    "tasks",
    "debug",
    "lifecycle",
]


class StreamingHandler:
    """Wrap a compiled graph with ergonomic streaming projections.

    Example:
        ```python
        handler = StreamingHandler(graph)

        # Sync
        run = handler.stream(input_data)
        for state in run.values:
            print(state)
        output = run.output

        # Async — terminal accessors are methods so a missing `await`
        # fails loudly instead of silently yielding a coroutine.
        run = await handler.astream(input_data)
        async for state in run.values:
            print(state)
        output = await run.output()
        ```
    """

    def __init__(self, graph: Pregel) -> None:
        """Initialize the handler.

        Args:
            graph: A compiled LangGraph graph to stream from.
        """
        self._graph = graph

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        transformers: list[StreamTransformer | TransformerFactory] | None = None,
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
            transformers: User transformers appended after the built-in
                `ValuesTransformer` and `MessagesTransformer`.

        Returns:
            A GraphRunStream the caller can iterate to drive the run.
        """
        mux = StreamMux(
            factories=_BUILTIN_FACTORIES + _coerce_factories(transformers),
            is_async=False,
        )
        values_t = mux.transformer_by_key("values")
        assert isinstance(values_t, ValuesTransformer)

        graph_iter = iter(
            self._graph.stream(
                input,
                _merge_v2_messages_flag(config),
                stream_mode=STREAM_V2_MODES,
                subgraphs=True,
                version="v2",
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
            )
        )

        return GraphRunStream(graph_iter, mux, values_t)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        transformers: list[StreamTransformer | TransformerFactory] | None = None,
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
            transformers: User transformers appended after the built-in
                `ValuesTransformer` and `MessagesTransformer`.

        Returns:
            An AsyncGraphRunStream whose projections can be awaited
            concurrently; each subscribed cursor drives the pump when
            its buffer is empty.
        """
        mux = StreamMux(
            factories=_BUILTIN_FACTORIES + _coerce_factories(transformers),
            is_async=True,
        )
        values_t = mux.transformer_by_key("values")
        assert isinstance(values_t, ValuesTransformer)

        graph_aiter = self._graph.astream(
            input,
            _merge_v2_messages_flag(config),
            stream_mode=STREAM_V2_MODES,
            subgraphs=True,
            version="v2",
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        ).__aiter__()

        return AsyncGraphRunStream(graph_aiter, mux, values_t)
