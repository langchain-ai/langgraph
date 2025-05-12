from collections.abc import AsyncIterator, Iterator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from inspect import signature
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Union,
)

from typing_extensions import Self

from langgraph.managed.base import ConfiguredManagedValue, ManagedValue, V
from langgraph.types import LoopProtocol


class Context(ManagedValue[V], Generic[V]):
    runtime = True

    value: V

    @staticmethod
    def of(
        ctx: Union[
            None,
            Callable[..., AbstractContextManager[V]],
            type[AbstractContextManager[V]],
            Callable[..., AbstractAsyncContextManager[V]],
            type[AbstractAsyncContextManager[V]],
        ] = None,
        actx: Optional[
            Union[
                Callable[..., AbstractAsyncContextManager[V]],
                type[AbstractAsyncContextManager[V]],
            ]
        ] = None,
    ) -> ConfiguredManagedValue:
        if ctx is None and actx is None:
            raise ValueError("Must provide either sync or async context manager.")
        return ConfiguredManagedValue(Context, {"ctx": ctx, "actx": actx})

    @classmethod
    @contextmanager
    def enter(cls, loop: LoopProtocol, **kwargs: Any) -> Iterator[Self]:
        with super().enter(loop, **kwargs) as self:
            if self.ctx is None:
                raise ValueError(
                    "Synchronous context manager not found. Please initialize Context value with a sync context manager, or invoke your graph asynchronously."
                )
            ctx = (
                self.ctx(loop.config)  # type: ignore[call-arg]
                if signature(self.ctx).parameters.get("config")
                else self.ctx()
            )
            with ctx as v:  # type: ignore[union-attr]
                self.value = v
                yield self

    @classmethod
    @asynccontextmanager
    async def aenter(cls, loop: LoopProtocol, **kwargs: Any) -> AsyncIterator[Self]:
        async with super().aenter(loop, **kwargs) as self:
            if self.actx is not None:
                ctx = (
                    self.actx(loop.config)  # type: ignore[call-arg]
                    if signature(self.actx).parameters.get("config")
                    else self.actx()
                )
            elif self.ctx is not None:
                ctx = (
                    self.ctx(loop.config)  # type: ignore
                    if signature(self.ctx).parameters.get("config")
                    else self.ctx()
                )
            else:
                raise ValueError(
                    "Asynchronous context manager not found. Please initialize Context value with an async context manager, or invoke your graph synchronously."
                )
            if hasattr(ctx, "__aenter__"):
                async with ctx as v:
                    self.value = v
                    yield self
            elif hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__"):
                with ctx as v:
                    self.value = v
                    yield self
            else:
                raise ValueError(
                    "Context manager must have either __enter__ or __aenter__ method."
                )

    def __init__(
        self,
        loop: LoopProtocol,
        *,
        ctx: Union[
            None, type[AbstractContextManager[V]], type[AbstractAsyncContextManager[V]]
        ] = None,
        actx: Optional[type[AbstractAsyncContextManager[V]]] = None,
    ) -> None:
        self.ctx = ctx
        self.actx = actx

    def __call__(self) -> V:
        return self.value
