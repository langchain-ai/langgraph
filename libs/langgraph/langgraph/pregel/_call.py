"""Utility to convert a user provided function into a Runnable with a ChannelWrite."""

from __future__ import annotations

import concurrent.futures
import functools
import inspect
import sys
import types
from collections.abc import Awaitable, Generator, Sequence
from typing import Any, Callable, Generic, TypeVar, cast

from langchain_core.runnables import Runnable
from typing_extensions import ParamSpec

from langgraph._internal._constants import CONF, CONFIG_KEY_CALL, RETURN
from langgraph._internal._runnable import (
    RunnableCallable,
    RunnableSeq,
    is_async_callable,
    run_in_executor,
)
from langgraph.config import get_config
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langgraph.types import CachePolicy, RetryPolicy

##
# Utilities borrowed from cloudpickle.
# https://github.com/cloudpipe/cloudpickle/blob/6220b0ce83ffee5e47e06770a1ee38ca9e47c850/cloudpickle/cloudpickle.py#L265


def _getattribute(obj: Any, name: str) -> Any:
    parent = None
    for subpath in name.split("."):
        if subpath == "<locals>":
            raise AttributeError(f"Can't get local attribute {name!r} on {obj!r}")
        try:
            parent = obj
            obj = getattr(obj, subpath)
        except AttributeError:
            raise AttributeError(f"Can't get attribute {name!r} on {obj!r}") from None
    return obj, parent


def _whichmodule(obj: Any, name: str) -> str | None:
    """Find the module an object belongs to.

    This function differs from ``pickle.whichmodule`` in two ways:
    - it does not mangle the cases where obj's module is __main__ and obj was
      not found in any module.
    - Errors arising during module introspection are ignored, as those errors
      are considered unwanted side effects.
    """
    module_name = getattr(obj, "__module__", None)

    if module_name is not None:
        return module_name
    # Protect the iteration by using a copy of sys.modules against dynamic
    # modules that trigger imports of other modules upon calls to getattr or
    # other threads importing at the same time.
    for module_name, module in sys.modules.copy().items():
        # Some modules such as coverage can inject non-module objects inside
        # sys.modules
        if (
            module_name == "__main__"
            or module_name == "__mp_main__"
            or module is None
            or not isinstance(module, types.ModuleType)
        ):
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except Exception:
            pass
    return None


def identifier(obj: Any, name: str | None = None) -> str | None:
    """Return the module and name of an object."""
    from langgraph._internal._runnable import RunnableCallable, RunnableSeq
    from langgraph.pregel._read import PregelNode

    if isinstance(obj, PregelNode):
        obj = obj.bound
    if isinstance(obj, RunnableSeq):
        obj = obj.steps[0]
    if isinstance(obj, RunnableCallable):
        obj = obj.func
    if name is None:
        name = getattr(obj, "__qualname__", None)
    if name is None:  # pragma: no cover
        # This used to be needed for Python 2.7 support but is probably not
        # needed anymore. However we keep the __name__ introspection in case
        # users of cloudpickle rely on this old behavior for unknown reasons.
        name = getattr(obj, "__name__", None)
    if name is None:
        return None

    module_name = getattr(obj, "__module__", None)
    if module_name is None:
        # In this case, obj.__module__ is None. obj is thus treated as dynamic.
        return None

    return f"{module_name}.{name}"


def _lookup_module_and_qualname(
    obj: Any, name: str | None = None
) -> tuple[types.ModuleType, str] | None:
    if name is None:
        name = getattr(obj, "__qualname__", None)
    if name is None:  # pragma: no cover
        # This used to be needed for Python 2.7 support but is probably not
        # needed anymore. However we keep the __name__ introspection in case
        # users of cloudpickle rely on this old behavior for unknown reasons.
        name = getattr(obj, "__name__", None)
    if name is None:
        return None

    module_name = _whichmodule(obj, name)

    if module_name is None:
        # In this case, obj.__module__ is None AND obj was not found in any
        # imported module. obj is thus treated as dynamic.
        return None

    if module_name == "__main__":
        return None

    # Note: if module_name is in sys.modules, the corresponding module is
    # assumed importable at unpickling time. See #357
    module = sys.modules.get(module_name, None)
    if module is None:
        # The main reason why obj's module would not be imported is that this
        # module has been dynamically created, using for example
        # types.ModuleType. The other possibility is that module was removed
        # from sys.modules after obj was created/imported. But this case is not
        # supported, as the standard pickle does not support it either.
        return None

    try:
        obj2, parent = _getattribute(module, name)
    except AttributeError:
        # obj was not found inside the module it points to
        return None
    if obj2 is not obj:
        return None
    return module, name


def _explode_args_trace_inputs(
    sig: inspect.Signature, input: tuple[tuple[Any, ...], dict[str, Any]]
) -> dict[str, Any]:
    args, kwargs = input
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop("self", None)
    arguments.pop("cls", None)
    for param_name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # Update with the **kwargs, and remove the original entry
            # This is to help flatten out keyword arguments
            if param_name in arguments:
                arguments.update(arguments.pop(param_name))
    return arguments


def get_runnable_for_entrypoint(func: Callable[..., Any]) -> Runnable:
    key = (func, False)
    if key in CACHE:
        return CACHE[key]
    else:
        if is_async_callable(func):
            run = RunnableCallable(
                None, func, name=func.__name__, trace=False, recurse=False
            )
        else:
            afunc = functools.update_wrapper(
                functools.partial(run_in_executor, None, func), func
            )
            run = RunnableCallable(
                func,
                afunc,
                name=func.__name__,
                trace=False,
                recurse=False,
            )
        if not _lookup_module_and_qualname(func):
            return run
        return CACHE.setdefault(key, run)


def get_runnable_for_task(func: Callable[..., Any]) -> Runnable:
    key = (func, True)
    if key in CACHE:
        return CACHE[key]
    else:
        if hasattr(func, "__name__"):
            name = func.__name__
        elif hasattr(func, "func"):
            name = func.func.__name__
        elif hasattr(func, "__class__"):
            name = func.__class__.__name__
        else:
            name = str(func)

        if is_async_callable(func):
            run = RunnableCallable(
                None,
                func,
                explode_args=True,
                name=name,
                trace=False,
                recurse=False,
            )
        else:
            run = RunnableCallable(
                func,
                functools.wraps(func)(functools.partial(run_in_executor, None, func)),
                explode_args=True,
                name=name,
                trace=False,
                recurse=False,
            )
        seq = RunnableSeq(
            run,
            ChannelWrite([ChannelWriteEntry(RETURN)]),
            name=name,
            trace_inputs=functools.partial(
                _explode_args_trace_inputs, inspect.signature(func)
            ),
        )
        if not _lookup_module_and_qualname(func):
            return seq
        return CACHE.setdefault(key, seq)


CACHE: dict[tuple[Callable[..., Any], bool], Runnable] = {}


P = ParamSpec("P")
P1 = TypeVar("P1")
T = TypeVar("T")


class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self) -> Generator[T, None, T]:
        yield cast(T, ...)


def call(
    func: Callable[P, Awaitable[T]] | Callable[P, T],
    *args: Any,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    **kwargs: Any,
) -> SyncAsyncFuture[T]:
    config = get_config()
    impl = config[CONF][CONFIG_KEY_CALL]
    fut = impl(
        func,
        (args, kwargs),
        retry_policy=retry_policy,
        cache_policy=cache_policy,
        callbacks=config["callbacks"],
    )
    return fut
