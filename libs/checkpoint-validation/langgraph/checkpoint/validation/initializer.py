"""Checkpointer test registration and factory management."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

# Module-level registry of decorated checkpointer factories.
_REGISTRY: dict[str, RegisteredCheckpointer] = {}


@dataclass
class RegisteredCheckpointer:
    """A registered checkpointer test factory."""

    name: str
    factory: Callable[[], AsyncGenerator[BaseCheckpointSaver, None]]
    skip_capabilities: set[str] = field(default_factory=set)

    @asynccontextmanager
    async def create(self) -> AsyncGenerator[BaseCheckpointSaver, None]:
        """Create a fresh checkpointer instance via the async generator."""
        gen = self.factory()
        try:
            saver = await gen.__anext__()
            yield saver
        finally:
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass


def checkpointer_test(
    name: str,
    *,
    skip_capabilities: set[str] | None = None,
) -> Callable[[Any], RegisteredCheckpointer]:
    """Register an async generator as a checkpointer test factory.

    The generator should: setup -> yield checkpointer -> cleanup.

    Example::

        @checkpointer_test(name="InMemorySaver")
        async def memory_checkpointer():
            yield InMemorySaver()
    """

    def decorator(fn: Any) -> RegisteredCheckpointer:
        registered = RegisteredCheckpointer(
            name=name,
            factory=fn,
            skip_capabilities=skip_capabilities or set(),
        )
        _REGISTRY[name] = registered
        return registered

    return decorator


def get_registry() -> dict[str, RegisteredCheckpointer]:
    """Return the module-level registry of registered checkpointers."""
    return _REGISTRY
