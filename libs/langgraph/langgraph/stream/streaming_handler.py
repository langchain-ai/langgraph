from __future__ import annotations

import asyncio
import threading
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

        Returns a `GraphRunStream` immediately. A background daemon thread
        pumps events from the graph into the transformer pipeline.
        """
        mux, extensions, native_keys, values_t = self._setup(
            transformers, is_async=False
        )

        def pump() -> None:
            try:
                for part in self._graph.stream(
                    input,
                    config,
                    stream_mode=STREAM_V2_MODES,
                    subgraphs=True,
                    version="v2",
                    interrupt_before=interrupt_before,
                    interrupt_after=interrupt_after,
                ):
                    mux.push(convert_to_protocol_event(part))
                mux.close()
            except BaseException as e:
                mux.fail(e)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()

        run = GraphRunStream(mux, extensions, values_t, thread)
        for key in native_keys:
            setattr(run, key, extensions[key])
        return run

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
        mux, extensions, native_keys, values_t = self._setup(
            transformers, is_async=True
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
                    mux.push(convert_to_protocol_event(part))
                mux.close()
            except BaseException as e:
                mux.fail(e)

        task = asyncio.create_task(pump())

        run = AsyncGraphRunStream(mux, extensions, values_t, task)
        for key in native_keys:
            setattr(run, key, extensions[key])
        return run

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _setup(
        user_transformers: list[StreamTransformer] | None,
        *,
        is_async: bool = False,
    ) -> tuple[StreamMux, dict[str, Any], set[str], ValuesTransformer]:
        """Create the mux, register all transformers.

        Returns (mux, extensions, native_keys, values_transformer).
        """
        mux = StreamMux(is_async=is_async)

        values_t = ValuesTransformer(is_async=is_async)
        messages_t = MessagesTransformer(is_async=is_async)

        all_transformers: list[StreamTransformer] = [values_t, messages_t]
        if user_transformers:
            all_transformers.extend(user_transformers)

        extensions: dict[str, Any] = {}
        native_keys: set[str] = set()

        for t in all_transformers:
            projection = mux.register(t)
            extensions.update(projection)
            if getattr(t, "_native", False):
                native_keys.update(projection.keys())

        return mux, extensions, native_keys, values_t
