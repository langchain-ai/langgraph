from __future__ import annotations

import ast
import functools
import inspect
import re
import textwrap
import types
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence
from langgraph.checkpoint.base import ChannelVersions
from typing_extensions import override

from langgraph._internal._runnable import RunnableCallable, RunnableSeq
from langgraph.pregel.protocol import PregelProtocol


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


@functools.lru_cache(maxsize=256)
def _get_nonlocal_names(code: types.CodeType) -> frozenset[str]:
    """Return the set of nonlocal variable names referenced by a function.

    Cached by code object so the expensive source fetch + AST parse only
    happens once per unique function definition across repeated graph compiles.

    Args:
        code: The code object of the function to analyse.

    Returns:
        Frozenset of variable names that the function reads from its enclosing
        scope (free variables and globals referenced in function bodies).
    """
    try:
        source = inspect.getsource(code)
        tree = ast.parse(textwrap.dedent(source))
        visitor = FunctionNonLocals()
        visitor.visit(tree)
        return frozenset(visitor.nonlocals)
    except (SyntaxError, TypeError, OSError, SystemError):
        return frozenset()


def get_function_nonlocals(func: Callable) -> list[Any]:
    """Get the nonlocal variables accessed by a function.

    The expensive source-parsing step is cached by code object; only the
    cheap closure-variable lookup runs on every call.

    Args:
        func: The function to check.

    Returns:
        List[Any]: The nonlocal variables accessed by the function.
    """
    actual_func = (
        func.__wrapped__
        if hasattr(func, "__wrapped__") and callable(func.__wrapped__)
        else func
    )
    # Fast path: no free variables means nothing to scan.
    if not actual_func.__code__.co_freevars:
        return []

    nonlocal_names = _get_nonlocal_names(actual_func.__code__)
    if not nonlocal_names:
        return []

    closure = inspect.getclosurevars(actual_func)
    candidates = {**closure.globals, **closure.nonlocals}
    values: list[Any] = []
    for k, v in candidates.items():
        if k in nonlocal_names:
            values.append(v)
        for kk in nonlocal_names:
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
