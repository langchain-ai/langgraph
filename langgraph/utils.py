import enum
from typing import Any, Awaitable, Callable, Optional

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import merge_configs
from langchain_core.runnables.graph import Edge, Graph, Node, is_uuid


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
        self.name = name or func.__name__
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
            ret = self.func(input, merge_configs(self.config, config), **self.kwargs)
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
            ret = await self.afunc(
                input, merge_configs(self.config, config), **self.kwargs
            )
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
