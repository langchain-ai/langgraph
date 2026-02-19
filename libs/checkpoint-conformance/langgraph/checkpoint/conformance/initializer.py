"""Checkpointer test registration and factory management."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

# Type for the lifespan async context manager factory.
LifespanFactory = Callable[[], AsyncGenerator[None, None]]

# Module-level registry of decorated checkpointer factories.
_REGISTRY: dict[str, RegisteredCheckpointer] = {}


async def _noop_lifespan() -> AsyncGenerator[None, None]:
    yield


@dataclass
class RegisteredCheckpointer:
    """A registered checkpointer test factory."""

    name: str
    factory: Callable[[], AsyncGenerator[BaseCheckpointSaver, None]]
    skip_capabilities: set[str] = field(default_factory=set)
    lifespan: LifespanFactory = _noop_lifespan

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

    @asynccontextmanager
    async def enter_lifespan(self) -> AsyncGenerator[None, None]:
        """Enter the lifespan context (once per validation run)."""
        gen = self.lifespan()
        try:
            await gen.__anext__()
            yield
        finally:
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass


def checkpointer_test(
    name: str,
    *,
    skip_capabilities: set[str] | None = None,
    lifespan: LifespanFactory | None = None,
) -> Callable[[Any], RegisteredCheckpointer]:
    """Register an async generator as a checkpointer test factory.

    The factory is called once per capability suite to create a fresh
    checkpointer.  The optional `lifespan` is an async generator that
    runs once for the entire validation run (e.g. to create/destroy a
    database).

    Example::

        @checkpointer_test(name="InMemorySaver")
        async def memory_checkpointer():
            yield InMemorySaver()

    With lifespan::

        async def pg_lifespan():
            await create_database()
            yield
            await drop_database()

        @checkpointer_test(name="PostgresSaver", lifespan=pg_lifespan)
        async def pg_checkpointer():
            yield PostgresSaver(conn_string="...")
    """

    def decorator(fn: Any) -> RegisteredCheckpointer:
        registered = RegisteredCheckpointer(
            name=name,
            factory=fn,
            skip_capabilities=skip_capabilities or set(),
            lifespan=lifespan or _noop_lifespan,
        )
        _REGISTRY[name] = registered
        return registered

    return decorator
