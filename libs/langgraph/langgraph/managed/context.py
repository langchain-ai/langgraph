from contextlib import asynccontextmanager, contextmanager
from inspect import signature
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Callable,
    ContextManager,
    Generic,
    Iterator,
    Optional,
    Type,
    Union,
)

from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.managed.base import ConfiguredManagedValue, ManagedValue, V


class Context(ManagedValue[V], Generic[V]):
    runtime = True

    value: V

    @staticmethod
    def of(
        ctx: Union[
            None,
            Callable[..., ContextManager[V]],
            Type[ContextManager[V]],
            Callable[..., AsyncContextManager[V]],
            Type[AsyncContextManager[V]],
        ] = None,
        actx: Optional[
            Union[
                Callable[..., AsyncContextManager[V]],
                Type[AsyncContextManager[V]],
            ]
        ] = None,
    ) -> ConfiguredManagedValue:
        if ctx is None and actx is None:
            raise ValueError("Must provide either sync or async context manager.")
        return ConfiguredManagedValue(Context, {"ctx": ctx, "actx": actx})

    @classmethod
    @contextmanager
    def enter(cls, config: RunnableConfig, **kwargs: Any) -> Iterator[Self]:
        with super().enter(config, **kwargs) as self:
            if self.ctx is None:
                raise ValueError(
                    "Synchronous context manager not found. Please initialize Context value with a sync context manager, or invoke your graph asynchronously."
                )
            ctx = (
                self.ctx(config)  # type: ignore[call-arg]
                if signature(self.ctx).parameters.get("config")
                else self.ctx()
            )
            with ctx as v:  # type: ignore[union-attr]
                self.value = v
                yield self

    @classmethod
    @asynccontextmanager
    async def aenter(cls, config: RunnableConfig, **kwargs: Any) -> AsyncIterator[Self]:
        async with super().aenter(config, **kwargs) as self:
            if self.actx is not None:
                ctx = (
                    self.actx(config)  # type: ignore[call-arg]
                    if signature(self.actx).parameters.get("config")
                    else self.actx()
                )
            elif self.ctx is not None:
                ctx = (
                    self.ctx(config)  # type: ignore
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
        config: RunnableConfig,
        *,
        ctx: Union[None, Type[ContextManager[V]], Type[AsyncContextManager[V]]] = None,
        actx: Optional[Type[AsyncContextManager[V]]] = None,
    ) -> None:
        self.ctx = ctx
        self.actx = actx

    def __call__(self, step: int) -> V:
        return self.value
