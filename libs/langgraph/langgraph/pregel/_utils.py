from __future__ import annotations

import ast
import inspect
import re
import textwrap
from collections.abc import Callable, Sequence
from functools import partial
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
from typing_extensions import override

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


def get_function_nonlocals(func: Callable) -> list[Any]:
    """Get the nonlocal variables accessed by a function.

    Args:
        func: The function to check.

    Returns:
        List[Any]: The nonlocal variables accessed by the function.
    """
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = FunctionNonLocals()
        visitor.visit(tree)
        values: list[Any] = []
        closure = (
            inspect.getclosurevars(func.__wrapped__)
            if hasattr(func, "__wrapped__") and callable(func.__wrapped__)
            else inspect.getclosurevars(func)
        )
        candidates = {**closure.globals, **closure.nonlocals}
        for k, v in candidates.items():
            if k in visitor.nonlocals:
                values.append(v)
            for kk in visitor.nonlocals:
                if "." in kk and kk.startswith(k):
                    vv = v
                    for part in kk.split(".")[1:]:
                        if vv is None:
                            break
                        else:
                            try:
                                vv = getattr(vv, part)
                            except AttributeError:
                                break
                    else:
                        values.append(vv)
    except (SyntaxError, TypeError, OSError, SystemError):
        return []

    return values


class FunctionNonLocals(ast.NodeVisitor):
    """Get the nonlocal variables accessed of a function."""

    def __init__(self) -> None:
        self.nonlocals: set[str] = set()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visit a function definition.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Visit an async function definition.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    @override
    def visit_Lambda(self, node: ast.Lambda) -> Any:
        """Visit a lambda function.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)


class NonLocals(ast.NodeVisitor):
    """Get nonlocal variables accessed."""

    def __init__(self) -> None:
        self.loads: set[str] = set()
        self.stores: set[str] = set()

    @override
    def visit_Name(self, node: ast.Name) -> Any:
        """Visit a name node.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.stores.add(node.id)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Visit an attribute node.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        if isinstance(node.ctx, ast.Load):
            parent = node.value
            attr_expr = node.attr
            while isinstance(parent, ast.Attribute):
                attr_expr = parent.attr + "." + attr_expr
                parent = parent.value
            if isinstance(parent, ast.Name):
                self.loads.add(parent.id + "." + attr_expr)
                self.loads.discard(parent.id)
            elif isinstance(parent, ast.Call):
                if isinstance(parent.func, ast.Name):
                    self.loads.add(parent.func.id)
                else:
                    parent = parent.func
                    attr_expr = ""
                    while isinstance(parent, ast.Attribute):
                        if attr_expr:
                            attr_expr = parent.attr + "." + attr_expr
                        else:
                            attr_expr = parent.attr
                        parent = parent.value
                    if isinstance(parent, ast.Name):
                        self.loads.add(parent.id + "." + attr_expr)


def is_xxh3_128_hexdigest(value: str) -> bool:
    """Check if the given string matches the format of xxh3_128_hexdigest."""
    return bool(re.fullmatch(r"[0-9a-f]{32}", value))
