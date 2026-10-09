from __future__ import annotations

import dis
import re
from collections.abc import Callable, Sequence
from functools import partial
from types import CodeType, FunctionType
from typing import Any

from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
)
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.config import run_in_executor
from langgraph.checkpoint.base import ChannelVersions

from langgraph._internal._runnable import RunnableCallable, RunnableSeq
from langgraph._internal._timeout import sync_timeout_unsupported
from langgraph.pregel.protocol import PregelProtocol

_SEQUENCE_TYPES = (RunnableSeq, RunnableSequence)


def get_new_channel_versions(
    previous_versions: ChannelVersions, current_versions: ChannelVersions
) -> ChannelVersions:
    """Get subset of current_versions that are newer than previous_versions."""
    if previous_versions:
        version_type = type(next(iter(current_versions.values()), None))
        null_version = version_type()  # type: ignore[misc]
        new_versions = {
            k: v
            for k, v in current_versions.items()
            if v > previous_versions.get(k, null_version)  # type: ignore[operator]
        }
    else:
        new_versions = current_versions

    return new_versions


def find_subgraph_pregel(candidate: Runnable) -> PregelProtocol | None:
    from langgraph.pregel import Pregel

    candidates: list[Runnable] = [candidate]

    for c in candidates:
        if (
            isinstance(c, PregelProtocol)
            # subgraphs that disabled checkpointing are not considered
            and (not isinstance(c, Pregel) or c.checkpointer is not False)
        ):
            return c
        elif isinstance(c, RunnableSequence) or isinstance(c, RunnableSeq):
            candidates.extend(c.steps)
        elif isinstance(c, RunnableLambda):
            candidates.extend(c.deps)
        elif isinstance(c, RunnableCallable):
            if c.func is not None:
                candidates.extend(
                    nl.__self__ if hasattr(nl, "__self__") else nl
                    for nl in get_function_nonlocals(c.func)
                )
            elif c.afunc is not None:
                candidates.extend(
                    nl.__self__ if hasattr(nl, "__self__") else nl
                    for nl in get_function_nonlocals(c.afunc)
                )

    return None


def _sequence_steps(runnable: Runnable) -> Sequence[Runnable] | None:
    if isinstance(runnable, _SEQUENCE_TYPES):
        return runnable.steps
    return None


def _parallel_steps(runnable: Runnable) -> Sequence[Runnable] | None:
    if isinstance(runnable, RunnableParallel):
        return tuple(runnable.steps__.values())
    return None


def _has_method_override(runnable: Runnable, method_name: str) -> bool:
    method = getattr(type(runnable), method_name, None)
    return method is not None and method is not getattr(Runnable, method_name)


def _is_executor_backed_afunc(afunc: Callable[..., Any] | None) -> bool:
    return isinstance(afunc, partial) and afunc.func is run_in_executor


def _has_native_async(runnable: Runnable) -> bool:
    if isinstance(runnable, RunnableCallable):
        return runnable.afunc is not None and not _is_executor_backed_afunc(
            runnable.afunc
        )
    if isinstance(runnable, RunnableLambda):
        return bool(getattr(runnable, "afunc", False))
    return _has_method_override(runnable, "ainvoke")


def _runnable_has_native_async(runnable: Runnable) -> bool:
    """Return whether a runnable can be idle-timed without known sync code.

    For custom runnable subclasses, an `ainvoke` override is treated as the
    async contract. We do not introspect whether that implementation delegates
    to blocking work internally — e.g. a subclass whose `ainvoke` calls
    `asyncio.to_thread(self.invoke, ...)` will pass this check but the wrapped
    sync work is still uncancellable. Idle-timeout enforcement on such a
    runnable will fire `NodeTimeoutError` correctly, but the background thread
    will keep running until its sync work returns.
    """

    while isinstance(runnable, RunnableBindingBase):
        runnable = runnable.bound
    steps = _sequence_steps(runnable)
    if steps is None:
        steps = _parallel_steps(runnable)
    if steps is not None:
        return all(_runnable_has_native_async(step) for step in steps)
    # Raw callables and the common composition wrappers created by graph
    # builders fall through here. We do not exhaustively unwrap every Runnable
    # wrapper — wrappers that provide `ainvoke` are treated as owning the async
    # contract.
    return _has_native_async(runnable)


def validate_timeout_supported(runnable: Runnable, *, name: str) -> None:
    if not _runnable_has_native_async(runnable):
        raise sync_timeout_unsupported(name)


# Values treated as dead ends when deciding whether to walk a function's
# bytecode. A container can hold a graph, but `find_subgraph_pregel` does not
# look inside one, so skipping it costs nothing while that holds. Matched by
# exact type, since a subclass of a builtin can carry attributes.
_LEAF_TYPES = frozenset(
    {
        int,
        float,
        complex,
        bool,
        str,
        bytes,
        bytearray,
        list,
        tuple,
        dict,
        set,
        frozenset,
        type(None),
    }
)


def get_function_nonlocals(func: Callable) -> list[Any]:
    """Get the values a function reaches from outside its own scope.

    Args:
        func: The function to check.

    Returns:
        Every captured cell value, the globals the function names, and each
        value along an attribute path it loads. Over-approximates: a value can
        come back without the function reaching it at runtime.
    """
    func = getattr(func, "__func__", func)  # bound method -> function
    wrapped = getattr(func, "__wrapped__", None)
    if callable(wrapped):
        func = getattr(wrapped, "__func__", wrapped)
    if not isinstance(func, FunctionType):
        return []
    code = func.__code__

    cells: dict[str, Any] = {}
    for name, cell in zip(code.co_freevars, func.__closure__ or ()):
        try:
            cells[name] = cell.cell_contents
        except ValueError:
            continue  # empty cell: a recursive def not yet bound

    # Every captured value counts, referenced or not: over-declaring costs an
    # introspection entry, under-declaring drops the subgraph's checkpoints and
    # stream events. Checking each cell against the bytecode would cost more and
    # only trade the cheap error for the expensive one.
    values: list[Any] = list(cells.values())
    global_ns = func.__globals__
    globals_ = {name: global_ns[name] for name in code.co_names if name in global_ns}
    if all(type(v) in _LEAF_TYPES for v in (*cells.values(), *globals_.values())):
        return values

    # Nested code objects hold the references made by inner defs, lambdas and
    # comprehensions, which resolve against the namespaces gathered above.
    codes = [code]
    for c in codes:
        codes.extend(k for k in c.co_consts if isinstance(k, CodeType))
        value: Any = None
        for instruction in dis.get_instructions(c):
            opname = instruction.opname
            if opname == "LOAD_GLOBAL":
                value = globals_.get(instruction.argval)
            elif opname == "LOAD_DEREF":
                value = cells.get(instruction.argval)
            elif opname in ("LOAD_ATTR", "LOAD_METHOD"):
                value = getattr(value, instruction.argval, None)
            else:
                value = None  # anything else ends the chain: `a, b.c` is not `a.c`
                continue
            if value is not None:
                values.append(value)
    return values


def is_xxh3_128_hexdigest(value: str) -> bool:
    """Check if the given string matches the format of xxh3_128_hexdigest."""
    return bool(re.fullmatch(r"[0-9a-f]{32}", value))
