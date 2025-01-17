import asyncio
import concurrent
import concurrent.futures
import functools
import inspect
import types
from collections.abc import Iterator
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    TypeVar,
    Union,
    overload,
)

from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import Graph, Node
from typing_extensions import ParamSpec

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import CONF, END, START, TAG_HIDDEN
from langgraph.pregel import Pregel
from langgraph.pregel.call import get_runnable_for_func
from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.read import PregelNode
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy, StreamMode, StreamWriter

P = ParamSpec("P")
P1 = TypeVar("P1")
T = TypeVar("T")


def call(
    func: Callable[P, T],
    *args: Any,
    retry: Optional[RetryPolicy] = None,
    **kwargs: Any,
) -> concurrent.futures.Future[T]:
    from langgraph.constants import CONFIG_KEY_CALL
    from langgraph.utils.config import get_config

    config = get_config()
    impl = config[CONF][CONFIG_KEY_CALL]
    fut = impl(func, (args, kwargs), retry=retry, callbacks=config["callbacks"])
    return fut


@overload
def task(
    *, retry: Optional[RetryPolicy] = None
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, asyncio.Future[T]]]: ...


@overload
def task(  # type: ignore[overload-cannot-match]
    *, retry: Optional[RetryPolicy] = None
) -> Callable[[Callable[P, T]], Callable[P, concurrent.futures.Future[T]]]: ...


@overload
def task(
    __func_or_none__: Callable[P, T],
) -> Callable[P, concurrent.futures.Future[T]]: ...


@overload
def task(
    __func_or_none__: Callable[P, Awaitable[T]],
) -> Callable[P, asyncio.Future[T]]: ...


def task(
    __func_or_none__: Optional[Union[Callable[P, T], Callable[P, Awaitable[T]]]] = None,
    *,
    retry: Optional[RetryPolicy] = None,
) -> Union[
    Callable[[Callable[P, Awaitable[T]]], Callable[P, asyncio.Future[T]]],
    Callable[[Callable[P, T]], Callable[P, concurrent.futures.Future[T]]],
    Callable[P, asyncio.Future[T]],
    Callable[P, concurrent.futures.Future[T]],
]:
    def decorator(
        func: Union[Callable[P, Awaitable[T]], Callable[P, T]],
    ) -> Callable[P, concurrent.futures.Future[T]]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _tick(__allargs__: tuple) -> T:
                return await func(*__allargs__[0], **__allargs__[1])

        else:

            @functools.wraps(func)
            def _tick(__allargs__: tuple) -> T:
                return func(*__allargs__[0], **__allargs__[1])

        wrapper = functools.partial(call, _tick, retry=retry)
        object.__setattr__(wrapper, "_is_pregel_task", True)
        return functools.update_wrapper(wrapper, func)

    if __func_or_none__ is not None:
        return decorator(__func_or_none__)

    return decorator


def entrypoint(
    *,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    config_schema: Optional[type[Any]] = None,
) -> Callable[[types.FunctionType], Pregel]:
    def _imp(func: types.FunctionType) -> Pregel:
        # wrap generators in a function that writes to StreamWriter
        if inspect.isgeneratorfunction(func):

            def gen_wrapper(*args: Any, writer: StreamWriter, **kwargs: Any) -> Any:
                for chunk in func(*args, **kwargs):
                    writer(chunk)

            bound = get_runnable_for_func(gen_wrapper)
            stream_mode: StreamMode = "custom"
        elif inspect.isasyncgenfunction(func):

            async def agen_wrapper(
                *args: Any, writer: StreamWriter, **kwargs: Any
            ) -> Any:
                async for chunk in func(*args, **kwargs):
                    writer(chunk)

            bound = get_runnable_for_func(agen_wrapper)
            stream_mode = "custom"
        else:
            bound = get_runnable_for_func(func)
            stream_mode = "updates"

        # get input and output types
        sig = inspect.signature(func)
        first_parameter_name = next(iter(sig.parameters.keys()), None)
        if not first_parameter_name:
            raise ValueError("Entrypoint function must have at least one parameter")
        input_type = (
            sig.parameters[first_parameter_name].annotation
            if sig.parameters[first_parameter_name].annotation
            is not inspect.Signature.empty
            else Any
        )
        output_type = (
            sig.return_annotation
            if sig.return_annotation is not inspect.Signature.empty
            else Any
        )

        return EntrypointPregel(
            nodes={
                func.__name__: PregelNode(
                    bound=bound,
                    triggers=[START],
                    channels=[START],
                    writers=[ChannelWrite([ChannelWriteEntry(END)], tags=[TAG_HIDDEN])],
                )
            },
            channels={
                START: EphemeralValue(input_type),
                END: LastValue(output_type, END),
            },
            input_channels=START,
            output_channels=END,
            stream_channels=END,
            stream_mode=stream_mode,
            stream_eager=True,
            checkpointer=checkpointer,
            store=store,
            config_type=config_schema,
        )

    return _imp


class EntrypointPregel(Pregel):
    def get_graph(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        xray: int | bool = False,
    ) -> Graph:
        name, entrypoint = next(iter(self.nodes.items()))
        graph = Graph()
        node = Node(f"__{name}", name, entrypoint.bound, None)
        graph.nodes[node.id] = node
        candidates: list[tuple[Node, Union[Callable, PregelProtocol]]] = [
            *_find_children(entrypoint.bound, node)
        ]
        seen: set[Union[Callable, PregelProtocol]] = set()
        for parent, child in candidates:
            if child in seen:
                continue
            else:
                seen.add(child)
            if callable(child):
                node = Node(f"__{child.__name__}", child.__name__, child, None)  # type: ignore[arg-type]
                graph.nodes[node.id] = node
                graph.add_edge(parent, node, conditional=True)
                graph.add_edge(node, parent)
                candidates.extend(_find_children(child, node))
            elif isinstance(child, Runnable):
                if xray > 0:
                    graph = child.get_graph(config, xray=xray - 1 if xray else 0)
                    graph.trim_first_node()
                    graph.trim_last_node()
                    s, e = graph.extend(graph, prefix=child.name or "")
                    if s is None:
                        raise ValueError(
                            f"Could not extend subgraph '{child.name}' due to missing entrypoint"
                        )
                    else:
                        graph.add_edge(parent, s, conditional=True)
                    if e is not None:
                        graph.add_edge(e, parent)
                else:
                    node = graph.add_node(child, child.name)
                    graph.add_edge(parent, node, conditional=True)
                    graph.add_edge(node, parent)
        return graph


def _find_children(
    candidate: Union[Callable, Runnable], parent: Node
) -> Iterator[tuple[Node, Union[Callable, PregelProtocol]]]:
    from langchain_core.runnables.utils import get_function_nonlocals

    from langgraph.utils.runnable import (
        RunnableCallable,
        RunnableLambda,
        RunnableSeq,
        RunnableSequence,
    )

    candidates: list[Union[Callable, Runnable]] = []
    if callable(candidate) and getattr(candidate, "_is_pregel_task", False) is True:
        candidates.extend(
            nl.__self__ if hasattr(nl, "__self__") else nl
            for nl in get_function_nonlocals(
                candidate.__wrapped__
                if hasattr(candidate, "__wrapped__") and callable(candidate.__wrapped__)
                else candidate
            )
        )
    else:
        candidates.append(candidate)

    for c in candidates:
        if callable(c) and getattr(c, "_is_pregel_task", False) is True:
            yield (parent, c)
        elif isinstance(c, PregelProtocol):
            yield (parent, c)
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
