from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig

from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import StreamTransformer
from langgraph.stream.run_stream import AsyncGraphRunStream, GraphRunStream
from langgraph.stream.transformers import MessagesTransformer, ValuesTransformer
from langgraph.types import All, StreamMode

# All stream modes to request from the graph.
STREAM_V2_MODES: list[StreamMode] = [
    "values",
    "updates",
    "messages",
    "custom",
    "checkpoints",
    "tasks",
    "debug",
]


class StreamingHandler:
    """Wraps a compiled graph and provides ergonomic streaming projections.

    Usage::

        handler = StreamingHandler(graph)

        # Sync
        run = handler.stream(input_data)
        for state in run.values:
            print(state)
        output = run.output

        # Async
        run = await handler.astream(input_data)
        async for state in run.values:
            print(state)
        output = await run.output
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        transformers: list[StreamTransformer] | None = None,
    ) -> GraphRunStream:
        """Start a sync streaming run.

        Returns a `GraphRunStream` immediately. The caller's iteration on
        any projection drives the graph forward — no background thread is
        used. This matches v1's model where the caller's ``for`` loop is
        the pump.
        """
        values_t = ValuesTransformer()
        mux = StreamMux(
            [values_t, MessagesTransformer(), *(transformers or ())],
            is_async=False,
        )

        graph_iter = iter(
            self._graph.stream(
                input,
                config,
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
        transformers: list[StreamTransformer] | None = None,
    ) -> AsyncGraphRunStream:
        """Start an async streaming run.

        Returns an `AsyncGraphRunStream` immediately. A background asyncio
        task pumps events from the graph into the transformer pipeline.
        """
        values_t = ValuesTransformer()
        mux = StreamMux(
            [values_t, MessagesTransformer(), *(transformers or ())],
            is_async=True,
        )

        async def pump() -> None:
            try:
                async for part in self._graph.astream(
                    input,
                    config,
                    stream_mode=STREAM_V2_MODES,
                    subgraphs=True,
                    version="v2",
                    interrupt_before=interrupt_before,
                    interrupt_after=interrupt_after,
                ):
                    await mux.apush(convert_to_protocol_event(part))
                await mux.aclose()
            except BaseException as e:
                await mux.afail(e)

        task = asyncio.create_task(pump())

        return AsyncGraphRunStream(mux, values_t, task)
