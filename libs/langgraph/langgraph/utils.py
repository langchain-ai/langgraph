import asyncio
import enum
import inspect
import sys
from contextvars import copy_context
from functools import partial, wraps
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Type, Union

from langchain_core.runnables.base import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableLike,
    RunnableParallel,
)
from langchain_core.runnables.config import (
    merge_configs,
    run_in_executor,
    var_child_runnable_config,
)
from langchain_core.runnables.utils import accepts_config
from typing_extensions import (
    Annotated,
    NotRequired,
    ReadOnly,
    Required,
    TypeGuard,
    get_origin,
)

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
        self.afunc = afunc
        self.config: Optional[RunnableConfig] = {"tags": tags} if tags else None
        self.kwargs = kwargs
        self.trace = trace
        self.recurse = recurse

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
        if self.trace:
            ret = self._call_with_config(
                self.func, input, merge_configs(self.config, config), **kwargs
            )
        else:
            config = merge_configs(self.config, config)
            context = copy_context()
            context.run(_set_config_context, config)
            if accepts_config(self.func):
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
        if self.trace:
            ret = await self._acall_with_config(
                self.afunc, input, merge_configs(self.config, config), **kwargs
            )
        else:
            config = merge_configs(self.config, config)
            context = copy_context()
            context.run(_set_config_context, config)
            if accepts_config(self.afunc):
                kwargs["config"] = config
            if sys.version_info >= (3, 11):
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


def _is_optional_type(type_: Any) -> bool:
    """Check if a type is Optional."""

    if hasattr(type_, "__origin__") and hasattr(type_, "__args__"):
        origin = get_origin(type_)
        if origin is Optional:
            return True
        if origin is Union:
            return any(
                arg is type(None) or _is_optional_type(arg) for arg in type_.__args__
            )
        if origin is Annotated:
            return _is_optional_type(type_.__args__[0])
        return origin is None
    if hasattr(type_, "__bound__") and type_.__bound__ is not None:
        return _is_optional_type(type_.__bound__)
    return type_ is None


def _is_required_type(type_: Any) -> Optional[bool]:
    """Check if an annotation is marked as Required/NotRequired.

    Returns:
        - True if required
        - False if not required
        - None if not annotated with either
    """
    origin = get_origin(type_)
    if origin is Required:
        return True
    if origin is NotRequired:
        return False
    if origin is Annotated or getattr(origin, "__args__", None):
        # See https://typing.readthedocs.io/en/latest/spec/typeddict.html#interaction-with-annotated
        return _is_required_type(type_.__args__[0])
    return None


def _is_readonly_type(type_: Any) -> bool:
    """Check if an annotation is marked as ReadOnly.

    Returns:
        - True if is read only
        - False if not read only
    """

    # See: https://typing.readthedocs.io/en/latest/spec/typeddict.html#typing-readonly-type-qualifier
    origin = get_origin(type_)
    if origin is Annotated:
        return _is_readonly_type(type_.__args__[0])
    if origin is ReadOnly:
        return True
    return False


_DEFAULT_KEYS = frozenset()


def get_field_default(name: str, type_: Any, schema: Type[Any]) -> Any:
    """Determine the default value for a field in a state schema.

    This is based on:
        If TypedDict:
            - Required/NotRequired
            - total=False -> everything optional
        - Type annotation (Optional/Union[None])
    """
    optional_keys = getattr(schema, "__optional_keys__", _DEFAULT_KEYS)
    irq = _is_required_type(type_)
    if name in optional_keys:
        # Either total=False or explicit NotRequired.
        # No type annotation trumps this.
        if irq:
            # Unless it's earlier versions of python & explicit Required
            return ...
        return None
    if irq is not None:
        if irq:
            # Handle Required[<type>]
            # (we already handled NotRequired and total=False)
            return ...
        # Handle NotRequired[<type>] for earlier versions of python
        return None
    # Note, we ignore ReadOnly attributes,
    # as they don't make much sense. (we don't care if you mutate the state in your node)
    # and mutating state in your node has no effect on our graph state.
    # Base case is the annotation
    if _is_optional_type(type_):
        return None
    return ...
