"""Experimental streaming wrapper for CompiledGraph.

``StreamingHandler`` wraps a compiled graph and exposes the new streaming
API without adding methods to the ``CompiledGraph`` class itself.

Usage::

    from langgraph.stream import StreamingHandler

    s = StreamingHandler(graph)

    # async
    run = await s.astream(input)
    async for msg in run.messages:
        ...

    # sync
    run = s.stream(input)
    for event in run:
        ...
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from langchain_core.runnables import RunnableConfig

from langgraph._internal._config import patch_configurable
from langgraph.stream._convert import STREAM_V2_MODES
from langgraph.stream._types import StreamTransformer
from langgraph.stream.run_stream import (
    AsyncGraphRunStream,
    GraphRunStream,
    create_async_graph_run_stream,
    create_graph_run_stream,
)
from langgraph.types import All

if TYPE_CHECKING:
    from langgraph.pregel import Pregel

#: Config key that activates the protocol messages handler.
#: Duplicated here to avoid a circular import with ``pregel._messages_v2``.
PROTOCOL_MESSAGES_STREAM_KEY = "__protocol_messages_stream"


class StreamingHandler:
    """Experimental streaming wrapper around a compiled graph.

    Provides ``.stream()`` and ``.astream()`` returning
    :class:`GraphRunStream` / :class:`AsyncGraphRunStream` with
    ergonomic projections (``run.values``, ``run.messages``,
    ``run.subgraphs``, ``run.output``).

    Args:
        graph: A compiled LangGraph (``Pregel`` instance).
    """

    def __init__(self, graph: Pregel) -> None:
        self._graph = graph

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        debug: bool | None = None,
        transformers: list[StreamTransformer] | None = None,
    ) -> AsyncGraphRunStream:
        """Stream graph execution, returning an
        :class:`~langgraph.stream.run_stream.AsyncGraphRunStream`.

        The returned stream provides ergonomic projections:

        - ``await run.output`` -- final state
        - ``async for v in run.values`` -- intermediate state snapshots
        - ``async for msg in run.messages`` -- per-message
          :class:`~langgraph.stream.chat_model_stream.AsyncChatModelStream`
          objects
        - ``async for sub in run.subgraphs`` -- child
          :class:`~langgraph.stream.run_stream.AsyncSubgraphRunStream`
          instances
        - ``async for event in run`` -- raw
          :class:`~langgraph.stream._types.ProtocolEvent` objects

        Args:
            input: The input to the graph.
            config: The configuration to use for the run.
            context: The static context to use for the run.
            interrupt_before: Nodes to interrupt before.
            interrupt_after: Nodes to interrupt after.
            debug: Whether to emit debug events.
            transformers: Optional user-supplied
                :class:`~langgraph.stream._types.StreamTransformer` instances
                for custom projections (available on ``run.extensions``).

        Returns:
            An :class:`~langgraph.stream.run_stream.AsyncGraphRunStream`.
        """
        merged_config = patch_configurable(config, {PROTOCOL_MESSAGES_STREAM_KEY: True})

        source = cast(
            AsyncIterator[tuple[tuple[str, ...], str, Any]],
            self._graph.astream(
                input,
                merged_config,
                context=context,
                stream_mode=STREAM_V2_MODES,
                subgraphs=True,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
                version="v1",
            ),
        )

        return await create_async_graph_run_stream(
            source,
            transformers=transformers,
            output_mapper=self._graph._output_mapper,
        )

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        debug: bool | None = None,
        transformers: list[StreamTransformer] | None = None,
    ) -> GraphRunStream:
        """Synchronous variant of :meth:`astream`.

        Returns a :class:`~langgraph.stream.run_stream.GraphRunStream`
        immediately.  The underlying source is consumed lazily as
        projections are iterated.

        See :meth:`astream` for full documentation.
        """
        merged_config = patch_configurable(config, {PROTOCOL_MESSAGES_STREAM_KEY: True})

        source = cast(
            Iterator[tuple[tuple[str, ...], str, Any]],
            self._graph.stream(
                input,
                merged_config,
                context=context,
                stream_mode=STREAM_V2_MODES,
                subgraphs=True,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
                version="v1",
            ),
        )

        return create_graph_run_stream(
            source,
            transformers=transformers,
            output_mapper=self._graph._output_mapper,
        )
