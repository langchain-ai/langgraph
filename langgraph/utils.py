import asyncio
import enum
import inspect
import sys
from contextvars import copy_context
from functools import partial, wraps
from typing import Any, Awaitable, Callable, Optional

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
from langchain_core.runnables.graph import Edge, Graph, Node, is_uuid
from langchain_core.runnables.utils import accepts_config


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
        self.config = {"tags": tags} if tags else None
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

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        if self.trace:
            ret = self._call_with_config(
                self.func, input, merge_configs(self.config, config), **self.kwargs
            )
        else:
            config = merge_configs(self.config, config)
            context = copy_context()
            context.run(var_child_runnable_config.set, config)
            kwargs = (
                {**self.kwargs, "config": config}
                if accepts_config(self.func)
                else self.kwargs
            )
            ret = context.run(self.func, input, **kwargs)
        if isinstance(ret, Runnable) and self.recurse:
            return ret.invoke(input, config)
        return ret

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        if not self.afunc:
            return self.invoke(input, config)
        if self.trace:
            ret = await self._acall_with_config(
                self.afunc, input, merge_configs(self.config, config), **self.kwargs
            )
        else:
            config = merge_configs(self.config, config)
            context = copy_context()
            context.run(var_child_runnable_config.set, config)
            kwargs = (
                {**self.kwargs, "config": config}
                if accepts_config(self.afunc)
                else self.kwargs
            )
            if sys.version_info >= (3, 11):
                ret = await asyncio.create_task(
                    self.afunc(input, **kwargs), context=context
                )
            else:
                ret = await self.afunc(input, **kwargs)
        if isinstance(ret, Runnable) and self.recurse:
            return await ret.ainvoke(input, config)
        return ret


class DrawableGraph(Graph):
    def extend(
        self, graph: Graph, prefix: str = ""
    ) -> tuple[Optional[Node], Optional[Node]]:
        if all(is_uuid(node.id) for node in graph.nodes.values()):
            super().extend(graph)
            return graph.first_node(), graph.last_node()

        new_nodes = {
            f"{prefix}:{k}": Node(f"{prefix}:{k}", v.data)
            for k, v in graph.nodes.items()
        }
        new_edges = [
            Edge(
                f"{prefix}:{edge.source}",
                f"{prefix}:{edge.target}",
                edge.data,
                edge.conditional,
            )
            for edge in graph.edges
        ]
        self.nodes.update(new_nodes)
        self.edges.extend(new_edges)
        first = graph.first_node()
        last = graph.last_node()
        return (
            Node(f"{prefix}:{first.id}", first.data) if first else None,
            Node(f"{prefix}:{last.id}", last.data) if last else None,
        )


def _isgencheck(thing: RunnableLike) -> bool:
    return (
        inspect.isasyncgenfunction(thing)
        or inspect.isgeneratorfunction(thing)
    )


def _isgenerator(thing: RunnableLike) -> bool:
    return (
        _isgencheck(thing)
        or hasattr(thing, "__call__")
        and _isgencheck(thing.__call__)
    )


def _iscoroutinefunction(thing: RunnableLike) -> bool:
    return (
        asyncio.iscoroutinefunction(thing)
        or hasattr(thing, "__call__")
        and asyncio.iscoroutinefunction(thing.__call__)
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
    elif _isgenerator(thing):
        return RunnableLambda(thing, name=name)
    elif callable(thing):
        if _iscoroutinefunction(thing):
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
