import asyncio
import enum
import inspect
import sys
from contextvars import copy_context
from functools import partial, wraps
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from langchain_core.load.serializable import to_json_not_implemented
from langchain_core.runnables.base import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableLike,
    RunnableParallel,
)
from langchain_core.runnables.config import (
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    run_in_executor,
    var_child_runnable_config,
)
from langchain_core.runnables.utils import accepts_config, accepts_run_manager
from typing_extensions import TypeGuard

from langgraph.utils.config import merge_configs, patch_config

try:
    from langchain_core.runnables.config import _set_config_context
except ImportError:
    # For forwards compatibility
    def _set_config_context(context: RunnableConfig) -> None:  # type: ignore
        """Set the context for the current thread."""
        var_child_runnable_config.set(context)


# Before Python 3.11 native StrEnum is not available
class StrEnum(str, enum.Enum):
    """A string enum."""


ASYNCIO_ACCEPTS_CONTEXT = sys.version_info >= (3, 11)


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
        recurse: bool = True,
        **kwargs: Any,
    ) -> None:
        if name is not None:
            self.name = name
        elif func:
            try:
                if func.__name__ != "<lambda>":
                    self.name = func.__name__
            except AttributeError:
                pass
        elif afunc:
            try:
                self.name = afunc.__name__
            except AttributeError:
                pass
        self.func = func
        if func is not None:
            self.func_accepts_config = accepts_config(func)
            self.func_accepts_run_manager = accepts_run_manager(func)
        self.afunc = afunc
        if afunc is not None:
            self.afunc_accepts_config = accepts_config(afunc)
            self.afunc_accepts_run_manager = accepts_run_manager(afunc)
        self.config: Optional[RunnableConfig] = {"tags": tags} if tags else None
        self.kwargs = kwargs
        self.trace = trace
        self.recurse = recurse
        self.serialized = to_json_not_implemented(self)

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
        if self.func is None:
            raise TypeError(
                f'No synchronous function provided to "{self.name}".'
                "\nEither initialize with a synchronous function or invoke"
                " via the async API (ainvoke, astream, etc.)"
            )
        kwargs = {**self.kwargs, **kwargs}
        config = ensure_config(merge_configs(self.config, config))
        context = copy_context()
        if self.trace:
            config = ensure_config(config)
            callback_manager = get_callback_manager_for_config(config)
            run_manager = callback_manager.on_chain_start(
                self.serialized,
                input,
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            try:
                child_config = patch_config(config, callbacks=run_manager.get_child())
                context = copy_context()
                context.run(_set_config_context, child_config)
                if self.func_accepts_config:
                    kwargs["config"] = config
                if self.func_accepts_run_manager:
                    kwargs["run_manager"] = run_manager
                ret = context.run(self.func, input, **kwargs)
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise
            else:
                run_manager.on_chain_end(ret)
        else:
            context.run(_set_config_context, config)
            if self.func_accepts_config:
                kwargs["config"] = config
            ret = context.run(self.func, input, **kwargs)
        if isinstance(ret, Runnable) and self.recurse:
            return ret.invoke(input, config)
        return ret

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        if not self.afunc:
            return self.invoke(input, config)
        kwargs = {**self.kwargs, **kwargs}
        config = ensure_config(merge_configs(self.config, config))
        context = copy_context()
        if self.trace:
            callback_manager = get_async_callback_manager_for_config(config)
            run_manager = await callback_manager.on_chain_start(
                self.serialized,
                input,
                name=config.get("run_name") or self.name,
                run_id=config.pop("run_id", None),
            )
            try:
                child_config = patch_config(config, callbacks=run_manager.get_child())
                context.run(_set_config_context, child_config)
                if self.afunc_accepts_config:
                    kwargs["config"] = config
                if self.afunc_accepts_run_manager:
                    kwargs["run_manager"] = run_manager
                coro = self.afunc(input, **kwargs)
                if ASYNCIO_ACCEPTS_CONTEXT:
                    ret = await asyncio.create_task(coro, context=context)
                else:
                    ret = await coro
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise
            else:
                await run_manager.on_chain_end(ret)
        else:
            context.run(_set_config_context, config)
            if self.afunc_accepts_config:
                kwargs["config"] = config
            if ASYNCIO_ACCEPTS_CONTEXT:
                ret = await asyncio.create_task(
                    self.afunc(input, **kwargs), context=context
                )
            else:
                ret = await self.afunc(input, **kwargs)
        if isinstance(ret, Runnable) and self.recurse:
            return await ret.ainvoke(input, config)
        return ret


def is_async_callable(
    func: Any,
) -> TypeGuard[Callable[..., Awaitable]]:
    """Check if a function is async."""
    return (
        asyncio.iscoroutinefunction(func)
        or hasattr(func, "__call__")
        and asyncio.iscoroutinefunction(func.__call__)
    )


def is_async_generator(
    func: Any,
) -> TypeGuard[Callable[..., AsyncIterator]]:
    """Check if a function is an async generator."""
    return (
        inspect.isasyncgenfunction(func)
        or hasattr(func, "__call__")
        and inspect.isasyncgenfunction(func.__call__)
    )


def coerce_to_runnable(thing: RunnableLike, *, name: str, trace: bool) -> Runnable:
    """Coerce a runnable-like object into a Runnable.

    Args:
        thing: A runnable-like object.

    Returns:
        A Runnable.
    """
    if isinstance(thing, Runnable):
        return thing
    elif is_async_generator(thing) or inspect.isgeneratorfunction(thing):
        return RunnableLambda(thing, name=name)
    elif callable(thing):
        if is_async_callable(thing):
            return RunnableCallable(None, thing, name=name, trace=trace)
        else:
            return RunnableCallable(
                thing,
                wraps(thing)(partial(run_in_executor, None, thing)),
                name=name,
                trace=trace,
            )
    elif isinstance(thing, dict):
        return RunnableParallel(thing)
    else:
        raise TypeError(
            f"Expected a Runnable, callable or dict."
            f"Instead got an unsupported type: {type(thing)}"
        )
