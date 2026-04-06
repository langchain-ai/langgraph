from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.callbacks.manager import ahandle_event, handle_event
from langchain_core.runnables import RunnableConfig

from langgraph.types import Interrupt

__all__ = (
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
        base_handlers: list[BaseCallbackHandler] = []
        base_inheritable_handlers: list[BaseCallbackHandler] = []
        if handlers is not None:
            base_handlers.extend(handlers)
        if inheritable_handlers is not None:
            base_inheritable_handlers.extend(inheritable_handlers)
        super().__init__(
            handlers=base_handlers,
            inheritable_handlers=base_inheritable_handlers,
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
            raise TypeError("handlers must inherit GraphCallbackHandler")
        super().add_handler(handler, inherit=inherit)

    def copy(
        self,
        *,
        run_id: UUID | None | object = _MISSING,
    ) -> GraphCallbackManager:
        resolved_run_id: UUID | None
        if run_id is _MISSING:
            resolved_run_id = self.run_id
        else:
            if run_id is not None and not isinstance(run_id, UUID):
                raise TypeError("run_id must be a UUID or None")
            resolved_run_id = run_id

        return self.__class__(
            handlers=[
                handler
                for handler in self.handlers
                if isinstance(handler, GraphCallbackHandler)
            ],
            inheritable_handlers=[
                handler
                for handler in self.inheritable_handlers
                if isinstance(handler, GraphCallbackHandler)
            ],
            parent_run_id=self.parent_run_id,
            tags=self.tags.copy(),
            inheritable_tags=self.inheritable_tags.copy(),
            metadata=self.metadata.copy(),
            inheritable_metadata=self.inheritable_metadata.copy(),
            run_id=resolved_run_id,
        )

    @classmethod
    def configure(
        cls,
        callbacks: object | None = None,
        *,
        run_id: UUID | None = None,
    ) -> GraphCallbackManager:
        if callbacks is None:
            return cls(run_id=run_id)
        if isinstance(callbacks, GraphCallbackManager):
            return callbacks.copy(run_id=run_id)
        if isinstance(callbacks, BaseCallbackManager):
            handlers = [
                handler
                for handler in callbacks.handlers
                if isinstance(handler, GraphCallbackHandler)
            ]
            inheritable_handlers = [
                handler
                for handler in callbacks.inheritable_handlers
                if isinstance(handler, GraphCallbackHandler)
            ]
            return cls(
                handlers=handlers,
                inheritable_handlers=inheritable_handlers,
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
        handlers = [
            handler
            for handler in callbacks
            if isinstance(handler, GraphCallbackHandler)
        ]
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
        handle_event(
            self.handlers,
            "on_interrupt",
            None,
            interrupts,
            run_id=self.run_id,
            status=status,
            checkpoint_id=checkpoint_id,
            checkpoint_ns=checkpoint_ns,
            is_nested=is_nested,
        )

    def on_resume(
        self,
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> None:
        handle_event(
            self.handlers,
            "on_resume",
            None,
            run_id=self.run_id,
            status=status,
            checkpoint_id=checkpoint_id,
            checkpoint_ns=checkpoint_ns,
            is_nested=is_nested,
        )

    async def aon_interrupt(
        self,
        interrupts: Sequence[Interrupt],
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
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
            is_nested=is_nested,
        )

    async def aon_resume(
        self,
        *,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> None:
        await ahandle_event(
            self.handlers,
            "on_resume",
            None,
            run_id=self.run_id,
            status=status,
            checkpoint_id=checkpoint_id,
            checkpoint_ns=checkpoint_ns,
            is_nested=is_nested,
        )


GraphCallbacks: TypeAlias = (
    GraphCallbackManager
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
