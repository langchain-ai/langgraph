from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.callbacks.manager import ahandle_event, handle_event
from langchain_core.runnables import RunnableConfig

from langgraph.types import Interrupt

__all__ = (
    "AsyncGraphCallbackManager",
    "GraphCallbackHandler",
    "GraphCallbackManager",
    "GraphCallbacks",
    "get_graph_callback_manager_for_config",
)


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
    ) -> Any:
        """Run when graph execution pauses due to interrupts."""

    def on_resume(
        self,
        *,
        run_id: UUID | None,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
    ) -> Any:
        """Run when graph execution resumes from a checkpoint."""


_MISSING = object()


def _filter_graph_handlers(
    handlers: list[BaseCallbackHandler],
) -> list[GraphCallbackHandler]:
    return [h for h in handlers if isinstance(h, GraphCallbackHandler)]


def _init_base_manager(
    manager: BaseCallbackManager,
    handlers: Sequence[GraphCallbackHandler] | None,
    inheritable_handlers: Sequence[GraphCallbackHandler] | None,
    parent_run_id: UUID | None,
    *,
    tags: list[str] | None,
    inheritable_tags: list[str] | None,
    metadata: dict[str, Any] | None,
    inheritable_metadata: dict[str, Any] | None,
    run_id: UUID | None,
) -> None:
    base_handlers: list[BaseCallbackHandler] = []
    base_inheritable_handlers: list[BaseCallbackHandler] = []
    if handlers is not None:
        base_handlers.extend(handlers)
    if inheritable_handlers is not None:
        base_inheritable_handlers.extend(inheritable_handlers)
    BaseCallbackManager.__init__(
        manager,
        handlers=base_handlers,
        inheritable_handlers=base_inheritable_handlers,
        parent_run_id=parent_run_id,
        tags=tags,
        inheritable_tags=inheritable_tags,
        metadata=metadata,
        inheritable_metadata=inheritable_metadata,
    )
    manager.run_id = run_id  # type: ignore[attr-defined]


def _configure_graph_callbacks(
    cls: type[_GraphManagerT],
    callbacks: object | None,
    *,
    run_id: UUID | None,
) -> _GraphManagerT:
    if callbacks is None:
        return cls(run_id=run_id)
    if isinstance(callbacks, cls):
        return callbacks.copy(run_id=run_id)
    if isinstance(callbacks, (GraphCallbackManager, AsyncGraphCallbackManager)):
        # Cross-type: extract handlers into the requested cls.
        return cls(
            handlers=_filter_graph_handlers(callbacks.handlers),
            inheritable_handlers=_filter_graph_handlers(callbacks.inheritable_handlers),
            parent_run_id=callbacks.parent_run_id,
            tags=callbacks.tags.copy(),
            inheritable_tags=callbacks.inheritable_tags.copy(),
            metadata=callbacks.metadata.copy(),
            inheritable_metadata=callbacks.inheritable_metadata.copy(),
            run_id=run_id,
        )
    if isinstance(callbacks, BaseCallbackManager):
        return cls(
            handlers=_filter_graph_handlers(callbacks.handlers),
            inheritable_handlers=_filter_graph_handlers(callbacks.inheritable_handlers),
            parent_run_id=callbacks.parent_run_id,
            tags=callbacks.tags.copy(),
            inheritable_tags=callbacks.inheritable_tags.copy(),
            metadata=callbacks.metadata.copy(),
            inheritable_metadata=callbacks.inheritable_metadata.copy(),
            run_id=run_id,
        )
    if isinstance(callbacks, GraphCallbackHandler):
        return cls((callbacks,), run_id=run_id)
    if isinstance(callbacks, (str, bytes)) or not isinstance(callbacks, Sequence):
        raise TypeError("callbacks must be a handler, sequence, or manager")
    return cls(_filter_graph_handlers(list(callbacks)), run_id=run_id)


def _copy_graph_manager(
    manager: GraphCallbackManager | AsyncGraphCallbackManager,
    cls: type[_GraphManagerT],
    run_id: UUID | None | object,
) -> _GraphManagerT:
    resolved_run_id: UUID | None
    if run_id is _MISSING:
        resolved_run_id = manager.run_id
    else:
        if run_id is not None and not isinstance(run_id, UUID):
            raise TypeError("run_id must be a UUID or None")
        resolved_run_id = run_id

    return cls(
        handlers=_filter_graph_handlers(manager.handlers),
        inheritable_handlers=_filter_graph_handlers(manager.inheritable_handlers),
        parent_run_id=manager.parent_run_id,
        tags=manager.tags.copy(),
        inheritable_tags=manager.inheritable_tags.copy(),
        metadata=manager.metadata.copy(),
        inheritable_metadata=manager.inheritable_metadata.copy(),
        run_id=resolved_run_id,
    )


class GraphCallbackManager(BaseCallbackManager):
    """Sync dispatcher for graph lifecycle events."""

    run_id: UUID | None

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
        _init_base_manager(
            self,
            handlers,
            inheritable_handlers,
            parent_run_id,
            tags=tags,
            inheritable_tags=inheritable_tags,
            metadata=metadata,
            inheritable_metadata=inheritable_metadata,
            run_id=run_id,
        )

    def add_handler(
        self,
        handler: BaseCallbackHandler,
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        if not isinstance(handler, GraphCallbackHandler):
            raise TypeError("handlers must inherit GraphCallbackHandler")
        super().add_handler(handler, inherit=inherit)

    def copy(
        self,
        *,
        run_id: UUID | None | object = _MISSING,
    ) -> GraphCallbackManager:
        return _copy_graph_manager(self, GraphCallbackManager, run_id)

    @classmethod
    def configure(
        cls,
        callbacks: object | None = None,
        *,
        run_id: UUID | None = None,
    ) -> GraphCallbackManager:
        return _configure_graph_callbacks(cls, callbacks, run_id=run_id)

    def on_interrupt(
        self,
        interrupts: Sequence[Interrupt],
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
    ) -> None:
        handle_event(
            self.handlers,
            "on_interrupt",
            None,
            interrupts,
            run_id=self.run_id,
            status=status,
            checkpoint_id=checkpoint_id,
            checkpoint_ns=checkpoint_ns,
        )

    def on_resume(
        self,
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
    ) -> None:
        handle_event(
            self.handlers,
            "on_resume",
            None,
            run_id=self.run_id,
            status=status,
            checkpoint_id=checkpoint_id,
            checkpoint_ns=checkpoint_ns,
        )


class AsyncGraphCallbackManager(BaseCallbackManager):
    """Async dispatcher for graph lifecycle events."""

    run_id: UUID | None

    @property
    def is_async(self) -> bool:
        """Return whether the manager is async."""
        return True

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
        _init_base_manager(
            self,
            handlers,
            inheritable_handlers,
            parent_run_id,
            tags=tags,
            inheritable_tags=inheritable_tags,
            metadata=metadata,
            inheritable_metadata=inheritable_metadata,
            run_id=run_id,
        )

    def add_handler(
        self,
        handler: BaseCallbackHandler,
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        if not isinstance(handler, GraphCallbackHandler):
            raise TypeError("handlers must inherit GraphCallbackHandler")
        super().add_handler(handler, inherit=inherit)

    def copy(
        self,
        *,
        run_id: UUID | None | object = _MISSING,
    ) -> AsyncGraphCallbackManager:
        return _copy_graph_manager(self, AsyncGraphCallbackManager, run_id)

    @classmethod
    def configure(
        cls,
        callbacks: object | None = None,
        *,
        run_id: UUID | None = None,
    ) -> AsyncGraphCallbackManager:
        return _configure_graph_callbacks(cls, callbacks, run_id=run_id)

    async def on_interrupt(
        self,
        interrupts: Sequence[Interrupt],
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
    ) -> None:
        await ahandle_event(
            self.handlers,
            "on_interrupt",
            None,
            interrupts,
            run_id=self.run_id,
            status=status,
            checkpoint_id=checkpoint_id,
            checkpoint_ns=checkpoint_ns,
        )

    async def on_resume(
        self,
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
    ) -> None:
        await ahandle_event(
            self.handlers,
            "on_resume",
            None,
            run_id=self.run_id,
            status=status,
            checkpoint_id=checkpoint_id,
            checkpoint_ns=checkpoint_ns,
        )


# TypeVar for shared configure/copy helpers.
from typing import TypeVar  # noqa: E402

_GraphManagerT = TypeVar(
    "_GraphManagerT", GraphCallbackManager, AsyncGraphCallbackManager
)

GraphCallbacks: TypeAlias = (
    GraphCallbackManager
    | AsyncGraphCallbackManager
    | BaseCallbackManager
    | GraphCallbackHandler
    | Sequence[BaseCallbackHandler]
    | Sequence[GraphCallbackHandler]
    | None
)


def get_graph_callback_manager_for_config(
    config: RunnableConfig,
    *,
    run_id: UUID | None = None,
) -> GraphCallbackManager:
    return GraphCallbackManager.configure(
        config.get("callbacks"),
        run_id=run_id,
    )


def get_async_graph_callback_manager_for_config(
    config: RunnableConfig,
    *,
    run_id: UUID | None = None,
) -> AsyncGraphCallbackManager:
    return AsyncGraphCallbackManager.configure(
        config.get("callbacks"),
        run_id=run_id,
    )
