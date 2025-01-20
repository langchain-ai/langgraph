import asyncio
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
    Union,
    overload,
)

from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import Graph, Node

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START, TAG_HIDDEN
from langgraph.pregel import Pregel
from langgraph.pregel.call import P, T, call, get_runnable_for_entrypoint
from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.read import PregelNode
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy, StreamMode, StreamWriter


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
    """Define a LangGraph task using the `task` decorator.

    !!! warning "Experimental"
        This is an experimental API that is subject to change.
        Do not use for production code.

    !!! important "Requires python 3.11 or higher for async functions"
        The `task` decorator supports both sync and async functions. To use async
        functions, ensure that you are using Python 3.11 or higher.

    Tasks can only be called from within an [entrypoint][langgraph.func.entrypoint] or
    from within a StateGraph. A task can be called like a regular function with the
    following differences:

    - When a checkpointer is enabled, the function inputs and outputs must be serializable.
    - The decorated function can only be called from within an entrypoint or StateGraph.
    - Calling the function produces a future. This makes it easy to parallelize tasks.

    Args:
        retry: An optional retry policy to use for the task in case of a failure.

    Returns:
        A callable function when used as a decorator.

    Example: Sync Task
        ```python
        from langgraph.func import entrypoint, task

        @task
        def add_one(a: int) -> int:
            return a + 1

        @entrypoint()
        def add_one(numbers: list[int]) -> list[int]:
            futures = [add_one(n) for n in numbers]
            results = [f.result() for f in futures]
            return results

        # Call the entrypoint
        add_one.invoke([1, 2, 3])  # Returns [2, 3, 4]
        ```

    Example: Async Task
        ```python
        import asyncio
        from langgraph.func import entrypoint, task

        @task
        async def add_one(a: int) -> int:
            return a + 1

        @entrypoint()
        async def add_one(numbers: list[int]) -> list[int]:
            futures = [add_one(n) for n in numbers]
            return asyncio.gather(*futures)

        # Call the entrypoint
        await add_one.ainvoke([1, 2, 3])  # Returns [2, 3, 4]
        ```
    """

    def decorator(
        func: Union[Callable[P, Awaitable[T]], Callable[P, T]],
    ) -> Union[
        Callable[P, concurrent.futures.Future[T]], Callable[P, asyncio.Future[T]]
    ]:
        call_func = functools.partial(call, func, retry=retry)
        object.__setattr__(call_func, "_is_pregel_task", True)
        return functools.update_wrapper(call_func, func)

    if __func_or_none__ is not None:
        return decorator(__func_or_none__)

    return decorator


def entrypoint(
    *,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    config_schema: Optional[type[Any]] = None,
) -> Callable[[types.FunctionType], Pregel]:
    """Define a LangGraph workflow using the `entrypoint` decorator.

    !!! warning "Experimental"
        This is an experimental API that is subject to change.
        Do not use for production code.

    The decorated function must accept a single parameter, which serves as the input
    to the function. This input parameter can be of any type. Use a dictionary
    to pass multiple parameters to the function.

    The decorated function also has access to these optional parameters:

    - `writer`: A `StreamWriter` instance for writing data to a stream.
    - `config`: A configuration object for accessing workflow settings.
    - `previous`: The previous return value for the given thread (available only when
        a checkpointer is provided).

    The entrypoint decorator can be applied to sync functions, async functions,
    generator functions, and async generator functions.

    For generator functions, the `previous` parameter will represent a list of
    the values previously yielded by the generator. During a run any values yielded
    by the generator, will be written to the `custom` stream.

    Args:
        checkpointer: Specify a checkpointer to create a workflow that can persist
            its state across runs.
        store: A generalized key-value store. Some implementations may support
            semantic search capabilities through an optional `index` configuration.
        config_schema: Specifies the schema for the configuration object that will be
            passed to the workflow.

    Returns:
        A decorator that converts a function into a Pregel graph.

    Example: Using entrypoint and tasks
        ```python
        import time

        from langgraph.func import entrypoint, task
        from langgraph.types import interrupt, Command
        from langgraph.checkpoint.memory import MemorySaver

        @task
        def compose_essay(topic: str) -> str:
            time.sleep(1.0)  # Simulate slow operation
            return f"An essay about {topic}"

        @entrypoint(checkpointer=MemorySaver())
        def review_workflow(topic: str) -> dict:
            \"\"\"Manages the workflow for generating and reviewing an essay.

            The workflow includes:
            1. Generating an essay about the given topic.
            2. Interrupting the workflow for human review of the generated essay.

            Upon resuming the workflow, compose_essay task will not be re-executed
            as its result is cached by the checkpointer.

            Args:
                topic (str): The subject of the essay.

            Returns:
                dict: A dictionary containing the generated essay and the human review.
            \"\"\"
            essay_future = compose_essay(topic)
            essay = essay_future.result()
            human_review = interrupt({
                \"question\": \"Please provide a review\",
                \"essay\": essay
            })
            return {
                \"essay\": essay,
                \"review\": human_review,
            }

        # Example configuration for the workflow
        config = {
            \"configurable\": {
                \"thread_id\": \"some_thread\"
            }
        }

        # Topic for the essay
        topic = \"cats\"

        # Stream the workflow to generate the essay and await human review
        for result in review_workflow.stream(topic, config):
            print(result)

        # Example human review provided after the interrupt
        human_review = \"This essay is great.\"

        # Resume the workflow with the provided human review
        for result in review_workflow.stream(Command(resume=human_review), config):
            print(result)
        ```

    Example: Accessing the previous return value
        When a checkpointer is enabled the function can access the previous return value
        of the previous invocation on the same thread id.

        ```python
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.func import entrypoint, task

        @entrypoint(checkpointer=MemorySaver())
        def my_workflow(input_data: str, previous: Optional[str] = None) -> str:
            return "world"

        # highlight-next-line
        config = {
            "configurable": {
                "thread_id":
            }
        }
        my_workflow.invoke("hello")
        ```
    """

    def _imp(func: types.FunctionType) -> Pregel:
        """Convert a function into a Pregel graph.

        Args:
            func: The function to convert. Support both sync and async functions, as well
                   as generator and async generator functions.

        Returns:
            A Pregel graph.
        """
        # wrap generators in a function that writes to StreamWriter
        if inspect.isgeneratorfunction(func):
            original_sig = inspect.signature(func)
            # Check if original signature has a writer argument with a matching type.
            # If not, we'll inject it into the decorator, but not pass it
            # to the wrapped function.
            if "writer" in original_sig.parameters:

                @functools.wraps(func)
                def gen_wrapper(*args: Any, writer: StreamWriter, **kwargs: Any) -> Any:
                    chunks = []
                    for chunk in func(*args, writer=writer, **kwargs):
                        writer(chunk)
                        chunks.append(chunk)
                    return chunks
            else:

                @functools.wraps(func)
                def gen_wrapper(*args: Any, writer: StreamWriter, **kwargs: Any) -> Any:
                    chunks = []
                    # Do not pass the writer argument to the wrapped function
                    # as it does not have a matching parameter
                    for chunk in func(*args, **kwargs):
                        writer(chunk)
                        chunks.append(chunk)
                    return chunks

                # Create a new parameter for the writer argument
                extra_param = inspect.Parameter(
                    "writer",
                    inspect.Parameter.KEYWORD_ONLY,
                    # The extra argument is a keyword-only argument
                    default=lambda _: None,
                )
                # Update the function's signature to include the extra argument
                new_params = list(original_sig.parameters.values()) + [extra_param]
                new_sig = original_sig.replace(parameters=new_params)
                # Update the signature of the wrapper function
                gen_wrapper.__signature__ = new_sig  # type: ignore

            bound = get_runnable_for_entrypoint(gen_wrapper)
            stream_mode: StreamMode = "custom"
        elif inspect.isasyncgenfunction(func):
            original_sig = inspect.signature(func)
            # Check if original signature has a writer argument with a matching type.
            # If not, we'll inject it into the decorator, but not pass it
            # to the wrapped function.
            if "writer" in original_sig.parameters:

                @functools.wraps(func)
                async def agen_wrapper(
                    *args: Any, writer: StreamWriter, **kwargs: Any
                ) -> Any:
                    chunks = []
                    async for chunk in func(*args, writer=writer, **kwargs):
                        writer(chunk)
                        chunks.append(chunk)
                    return chunks
            else:

                @functools.wraps(func)
                async def agen_wrapper(
                    *args: Any, writer: StreamWriter, **kwargs: Any
                ) -> Any:
                    chunks = []
                    async for chunk in func(*args, **kwargs):
                        writer(chunk)
                        chunks.append(chunk)
                    return chunks

                # Create a new parameter for the writer argument
                extra_param = inspect.Parameter(
                    "writer",
                    inspect.Parameter.KEYWORD_ONLY,
                    # The extra argument is a keyword-only argument
                    default=lambda _: None,
                )
                # Update the function's signature to include the extra argument
                new_params = list(original_sig.parameters.values()) + [extra_param]
                new_sig = original_sig.replace(parameters=new_params)
                # Update the signature of the wrapper function
                agen_wrapper.__signature__ = new_sig  # type: ignore

            bound = get_runnable_for_entrypoint(agen_wrapper)
            stream_mode = "custom"
        else:
            bound = get_runnable_for_entrypoint(func)
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
        xray: Union[int, bool] = False,
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
