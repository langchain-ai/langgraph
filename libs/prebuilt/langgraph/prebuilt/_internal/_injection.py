from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Optional, Callable

from langchain_core.runnables import RunnableConfig

from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter


# Special type to denote any type is accepted
ANY_TYPE = object()

VALID_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)



# List of keyword arguments that can be injected into nodes / tasks / tools at runtime.
# A named argument may appear multiple times if it appears with distinct types.
KWARGS_CONFIG_KEYS: tuple[tuple[str, tuple[Any, ...], str, Any], ...] = (
    (
        "config",
        (
            RunnableConfig,
            "RunnableConfig",
            Optional[RunnableConfig],
            "Optional[RunnableConfig]",
            inspect.Parameter.empty,
        ),
        # for now, use config directly, eventually, will pop off of Runtime
        "N/A",
        inspect.Parameter.empty,
    ),
    (
        "writer",
        (StreamWriter, "StreamWriter", inspect.Parameter.empty),
        "stream_writer",
        lambda _: None,
    ),
    (
        "store",
        (
            BaseStore,
            "BaseStore",
            inspect.Parameter.empty,
        ),
        "store",
        inspect.Parameter.empty,
    ),
    (
        "store",
        (
            Optional[BaseStore],
            "Optional[BaseStore]",
        ),
        "store",
        None,
    ),
    (
        "previous",
        (ANY_TYPE,),
        "previous",
        inspect.Parameter.empty,
    ),
    (
        "runtime",
        (ANY_TYPE,),
        # we never hit this block, we just inject runtime directly
        "N/A",
        inspect.Parameter.empty,
    ),
)


@dataclass
class InjectionInfo:
    """Information about which injected arguments a function supports.

    Attributes:
        func_accepts: Dictionary mapping argument names to tuples of (runtime_key, default_value)
        supported_args: Set of argument names that the function accepts for injection
        has_config: Whether the function accepts a 'config' argument
        has_writer: Whether the function accepts a 'writer' argument
        has_store: Whether the function accepts a 'store' argument
        has_previous: Whether the function accepts a 'previous' argument
        has_runtime: Whether the function accepts a 'runtime' argument
    """

    func_accepts: dict[str, tuple[str, Any]]
    supported_args: set[str]
    has_config: bool
    has_writer: bool
    has_store: bool
    has_previous: bool
    has_runtime: bool


def get_function_injection_info(func: Callable) -> InjectionInfo:
    """Determine which injected arguments are supported by a function.

    This function analyzes a function's signature to determine which runtime arguments
    it can accept for injection. It uses the same logic as RunnableCallable to check
    parameter names, types, and annotations against the supported injection types.

    Args:
        func: The function to analyze for injection support

    Returns:
        InjectionInfo containing details about which arguments the function supports

    Example:
        ```python
        def my_tool(x: int, config: RunnableConfig, store: BaseStore) -> str:
            return f"x={x}, config={config is not None}, store={store is not None}"

        info = get_function_injection_info(my_tool)
        print(info.has_config)  # True
        print(info.has_store)   # True
        print(info.has_writer)  # False
        print(info.supported_args)  # {'config', 'store'}
        ```
    """
    func_accepts: dict[str, tuple[str, Any]] = {}
    params = inspect.signature(func).parameters

    for kw, typ, runtime_key, default in KWARGS_CONFIG_KEYS:
        p = params.get(kw)

        if p is None or p.kind not in VALID_KINDS:
            # If parameter is not found or is not a valid kind, skip
            continue

        if typ != (ANY_TYPE,) and p.annotation not in typ:
            # A specific type is required, but the function annotation does
            # not match the expected type.

            # If this is a config parameter with incorrect typing, we still accept it
            # but could emit a warning (following RunnableCallable behavior)
            if kw == "config" and p.annotation != inspect.Parameter.empty:
                # Could add warning here if needed
                pass
            else:
                continue

        # If the kwarg is accepted by the function, store the key / runtime attribute to inject
        func_accepts[kw] = (runtime_key, default)

    supported_args = set(func_accepts.keys())

    return InjectionInfo(
        func_accepts=func_accepts,
        supported_args=supported_args,
        has_config="config" in supported_args,
        has_writer="writer" in supported_args,
        has_store="store" in supported_args,
        has_previous="previous" in supported_args,
        has_runtime="runtime" in supported_args,
    )
