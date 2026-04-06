from __future__ import annotations

import asyncio
import inspect
from collections.abc import Coroutine, Sequence
from typing import Any, TypeAlias, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.runnables import RunnableConfig

from langgraph._internal._constants import CONF
from langgraph.types import Interrupt

__all__ = (
    "GRAPH_CALLBACKS_KEY",
    "GraphCallbackHandler",
    "GraphCallbackManager",
    "GraphCallbacks",
    "get_graph_callback_manager_for_config",
)

GRAPH_CALLBACKS_KEY = "graph_callbacks"


class GraphCallbackHandler(BaseCallbackHandler):
    """Base class for graph-level lifecycle callbacks."""

    def on_interrupt(
        self,
        interrupts: Sequence[Interrupt],
        *,
        run_id: UUID | None,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> Any:
        """Run when graph execution pauses due to interrupts."""

    def on_resume(
        self,
        *,
        run_id: UUID | None,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> Any:
        """Run when graph execution resumes from a checkpoint."""


_MISSING = object()


class GraphCallbackManager(BaseCallbackManager):
    """Dispatches graph lifecycle events to GraphCallbackHandler handlers."""

    def __init__(
        self,
        handlers: Sequence[GraphCallbackHandler] | None = None,
        inheritable_handlers: Sequence[GraphCallbackHandler] | None = None,
        parent_run_id: UUID | None = None,
        *,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
        run_id: UUID | None = None,
    ) -> None:
        super().__init__(
            handlers=cast(list[BaseCallbackHandler], list(handlers or [])),
            inheritable_handlers=cast(
                list[BaseCallbackHandler], list(inheritable_handlers or [])
            ),
            parent_run_id=parent_run_id,
            tags=tags,
            inheritable_tags=inheritable_tags,
            metadata=metadata,
            inheritable_metadata=inheritable_metadata,
        )
        self.run_id = run_id

    def add_handler(
        self,
        handler: BaseCallbackHandler,
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        if not isinstance(handler, GraphCallbackHandler):
            raise TypeError("graph_callbacks entries must inherit GraphCallbackHandler")
        super().add_handler(handler, inherit=inherit)

    def copy(
        self,
        *,
        run_id: UUID | None | object = _MISSING,
    ) -> GraphCallbackManager:
        return self.__class__(
            handlers=cast(list[GraphCallbackHandler], self.handlers.copy()),
            inheritable_handlers=cast(
                list[GraphCallbackHandler], self.inheritable_handlers.copy()
            ),
            parent_run_id=self.parent_run_id,
            tags=self.tags.copy(),
            inheritable_tags=self.inheritable_tags.copy(),
            metadata=self.metadata.copy(),
            inheritable_metadata=self.inheritable_metadata.copy(),
            run_id=self.run_id if run_id is _MISSING else cast(UUID | None, run_id),
        )

    @classmethod
    def configure(
        cls,
        callbacks: GraphCallbacks = None,
        *,
        run_id: UUID | None = None,
    ) -> GraphCallbackManager:
        if callbacks is None:
            return cls(run_id=run_id)
        if isinstance(callbacks, GraphCallbackManager):
            return callbacks.copy(run_id=run_id)
        if isinstance(callbacks, GraphCallbackHandler):
            return cls((callbacks,), run_id=run_id)
        if not isinstance(callbacks, Sequence):
            raise TypeError("graph_callbacks must be a handler, sequence, or manager")
        handlers = list(callbacks)
        for handler in handlers:
            if not isinstance(handler, GraphCallbackHandler):
                raise TypeError(
                    "graph_callbacks entries must inherit GraphCallbackHandler"
                )
        return cls(handlers, run_id=run_id)

    def on_interrupt(
        self,
        interrupts: Sequence[Interrupt],
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> None:
        for handler in self.handlers:
            graph_handler = cast(GraphCallbackHandler, handler)
            result = graph_handler.on_interrupt(
                interrupts,
                run_id=self.run_id,
                status=status,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
                is_nested=is_nested,
            )
            if inspect.iscoroutine(result):
                self._drain_coroutine(result)

    def on_resume(
        self,
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> None:
        for handler in self.handlers:
            graph_handler = cast(GraphCallbackHandler, handler)
            result = graph_handler.on_resume(
                run_id=self.run_id,
                status=status,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
                is_nested=is_nested,
            )
            if inspect.iscoroutine(result):
                self._drain_coroutine(result)

    async def aon_interrupt(
        self,
        interrupts: Sequence[Interrupt],
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> None:
        for handler in self.handlers:
            graph_handler = cast(GraphCallbackHandler, handler)
            result = graph_handler.on_interrupt(
                interrupts,
                run_id=self.run_id,
                status=status,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
                is_nested=is_nested,
            )
            if inspect.isawaitable(result):
                await result

    async def aon_resume(
        self,
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> None:
        for handler in self.handlers:
            graph_handler = cast(GraphCallbackHandler, handler)
            result = graph_handler.on_resume(
                run_id=self.run_id,
                status=status,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
                is_nested=is_nested,
            )
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _drain_coroutine(coro: Coroutine[Any, Any, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)


GraphCallbacks: TypeAlias = (
    GraphCallbackManager | GraphCallbackHandler | Sequence[GraphCallbackHandler] | None
)


def get_graph_callback_manager_for_config(
    config: RunnableConfig,
    *,
    run_id: UUID | None = None,
) -> GraphCallbackManager:
    configured = cast(
        GraphCallbacks,
        config.get(CONF, {}).get(GRAPH_CALLBACKS_KEY)
        if CONF in config
        else config.get(GRAPH_CALLBACKS_KEY),
    )
    return GraphCallbackManager.configure(configured, run_id=run_id)
