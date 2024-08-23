from contextlib import asynccontextmanager, contextmanager
from inspect import signature
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    ContextManager,
    Iterator,
    Optional,
    Self,
    Type,
    Union,
)

from langchain_core.runnables import RunnableConfig

from langgraph.managed.base import ConfiguredManagedValue, ManagedValue, V


class Context(ManagedValue):
    runtime = True

    value: V

    @staticmethod
    def of(
        ctx: Union[None, Type[ContextManager[V]], Type[AsyncContextManager[V]]] = None,
        actx: Optional[Type[AsyncContextManager[V]]] = None,
    ) -> ConfiguredManagedValue:
        if ctx is None and actx is None:
            raise ValueError("Must provide either sync or async context manager.")
        return ConfiguredManagedValue(Context, {"ctx": ctx, "actx": actx})

    @classmethod
    @contextmanager
    def enter(cls, config: RunnableConfig, **kwargs: Any) -> Iterator[Self]:
        with super().enter(config, **kwargs) as self:
            if self.ctx is None:
                raise ValueError("Cannot enter sync context manager.")
            ctx = (
                self.ctx(config)
                if signature(self.ctx).parameters.get("config")
                else self.ctx()
            )
            with ctx as v:
                self.value = v
                yield self

    @classmethod
    @asynccontextmanager
    async def aenter(cls, config: RunnableConfig, **kwargs: Any) -> AsyncIterator[Self]:
        async with super().aenter(config, **kwargs) as self:
            if self.actx is not None:
                ctx = (
                    self.actx(config)
                    if signature(self.actx).parameters.get("config")
                    else self.actx()
                )
            else:
                ctx = (
                    self.ctx(config)
                    if signature(self.ctx).parameters.get("config")
                    else self.ctx()
                )
            if hasattr(ctx, "__aenter__"):
                async with ctx as v:
                    self.value = v
                    yield self
            else:
                with ctx as v:
                    self.value = v
                    yield self

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
