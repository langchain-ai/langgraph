import enum
from typing import Any, Awaitable, Callable, Optional

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import merge_configs


# Before Python 3.11 native StrEnum is not available
class StrEnum(str, enum.Enum):
    """A string enum."""

    pass


class RunnableCallable(Runnable):
    """A much simpler version of RunnableLambda that requires sync and async functions."""

    def __init__(
        self,
        func: Callable[..., Optional[Runnable]],
        afunc: Optional[Callable[..., Awaitable[Optional[Runnable]]]] = None,
        *,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        trace: bool = True,
        **kwargs: Any,
    ) -> None:
        self.name = name or func.__name__
        self.func = func
        self.afunc = afunc
        self.config: Optional[RunnableConfig] = {"tags": tags} if tags else None
        self.kwargs = kwargs
        self.trace = trace

    def __repr__(self) -> str:
        repr_args = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"name", "func", "afunc", "config", "kwargs", "trace"}
        }
        return f"{self.get_name()}({', '.join(f'{k}={v!r}' for k, v in repr_args.items())})"

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        kwargs = {**self.kwargs, **kwargs}
        if self.trace:
            ret = self._call_with_config(
                self.func, input, merge_configs(self.config, config), **kwargs
            )
        else:
            ret = self.func(input, merge_configs(self.config, config), **kwargs)
        if isinstance(ret, Runnable):
            return ret.invoke(input, config)
        return ret

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        if not self.afunc:
            return self.invoke(input, config, **kwargs)
        if self.trace:
            kwargs = {**self.kwargs, **kwargs}
            ret = await self._acall_with_config(
                self.afunc, input, merge_configs(self.config, config), **kwargs
            )
        else:
            kwargs = {**self.kwargs, **kwargs}
            ret = await self.afunc(input, merge_configs(self.config, config), **kwargs)
        if isinstance(ret, Runnable):
            return await ret.ainvoke(input, config)
        return ret
