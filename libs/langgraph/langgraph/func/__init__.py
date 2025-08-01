from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import warnings
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    cast,
    get_args,
    get_origin,
    overload,
)

from typing_extensions import Unpack

from langgraph._internal._constants import CACHE_NS_WRITES, PREVIOUS
from langgraph._internal._typing import MISSING, DeprecatedKwargs
from langgraph.cache.base import BaseCache
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.pregel import Pregel
from langgraph.pregel._call import (
    P,
    SyncAsyncFuture,
    T,
    call,
    get_runnable_for_entrypoint,
    identifier,
)
from langgraph.pregel._read import PregelNode
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langgraph.store.base import BaseStore
from langgraph.types import _DC_KWARGS, CachePolicy, RetryPolicy, StreamMode
from langgraph.typing import ContextT
from langgraph.warnings import LangGraphDeprecatedSinceV05, LangGraphDeprecatedSinceV10

__all__ = ("task", "entrypoint")


class _TaskFunction(Generic[P, T]):
    def __init__(
        self,
        func: Callable[P, T],
        *,
        retry_policy: Sequence[RetryPolicy],
        cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
        name: str | None = None,
    ) -> None:
        if name is not None:
            if hasattr(func, "__func__"):
                # handle class methods
                # NOTE: we're modifying the instance method to avoid modifying
                # the original class method in case it's shared across multiple tasks
                instance_method = functools.partial(func.__func__, func.__self__)  # type: ignore [attr-defined]
                instance_method.__name__ = name  # type: ignore [attr-defined]
                func = instance_method
            else:
                # handle regular functions / partials / callable classes, etc.
                func.__name__ = name
        self.func = func
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        functools.update_wrapper(self, func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]:
        return call(
            self.func,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
            *args,
            **kwargs,
        )

    def clear_cache(self, cache: BaseCache) -> None:
        """Clear the cache for this task."""
        if self.cache_policy is not None:
            cache.clear(((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),))

    async def aclear_cache(self, cache: BaseCache) -> None:
        """Clear the cache for this task."""
        if self.cache_policy is not None:
            await cache.aclear(
                ((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),)
            )


@overload
def task(
    *,
    name: str | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
    **kwargs: Unpack[DeprecatedKwargs],
) -> Callable[
    [Callable[P, Awaitable[T]] | Callable[P, T]],
    _TaskFunction[P, T],
]: ...


@overload
def task(
    __func_or_none__: Callable[P, Awaitable[T]] | Callable[P, T],
) -> _TaskFunction[P, T]: ...


def task(
    __func_or_none__: Callable[P, Awaitable[T]] | Callable[P, T] | None = None,
    *,
    name: str | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
    **kwargs: Unpack[DeprecatedKwargs],
) -> (
    Callable[[Callable[P, Awaitable[T]] | Callable[P, T]], _TaskFunction[P, T]]
    | _TaskFunction[P, T]
):
    """Define a LangGraph task using the `task` decorator.

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
        name: An optional name for the task. If not provided, the function name will be used.
        retry_policy: An optional retry policy (or list of policies) to use for the task in case of a failure.
        cache_policy: An optional cache policy to use for the task. This allows caching of the task results.

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
    if (retry := kwargs.get("retry", MISSING)) is not MISSING:
        warnings.warn(
            "`retry` is deprecated and will be removed. Please use `retry_policy` instead.",
            category=LangGraphDeprecatedSinceV05,
            stacklevel=2,
        )
        if retry_policy is None:
            retry_policy = retry  # type: ignore[assignment]

    retry_policies: Sequence[RetryPolicy] = (
        ()
        if retry_policy is None
        else (retry_policy,)
        if isinstance(retry_policy, RetryPolicy)
        else retry_policy
    )

    def decorator(
        func: Callable[P, Awaitable[T]] | Callable[P, T],
    ) -> Callable[P, concurrent.futures.Future[T]] | Callable[P, asyncio.Future[T]]:
        return _TaskFunction(
            func, retry_policy=retry_policies, cache_policy=cache_policy, name=name
        )

    if __func_or_none__ is not None:
        return decorator(__func_or_none__)

    return decorator


R = TypeVar("R")
S = TypeVar("S")


# The decorator was wrapped in a class to support the `final` attribute.
# In this form, the `final` attribute should play nicely with IDE autocompletion,
# and type checking tools.
# In addition, we'll be able to surface this information in the API Reference.
class entrypoint(Generic[ContextT]):
    """Define a LangGraph workflow using the `entrypoint` decorator.

    ### Function signature

    The decorated function must accept a **single parameter**, which serves as the input
    to the function. This input parameter can be of any type. Use a dictionary
    to pass **multiple parameters** to the function.

    ### Injectable parameters

    The decorated function can request access to additional parameters
    that will be injected automatically at run time. These parameters include:

    | Parameter        | Description                                                                                        |
    |------------------|----------------------------------------------------------------------------------------------------|
    | **`config`**     | A configuration object (aka RunnableConfig) that holds run-time configuration values.              |
    | **`previous`**   | The previous return value for the given thread (available only when a checkpointer is provided).   |
    | **`runtime`**    | A Runtime object that contains information about the current run, including context, store, writer |                                |

    The entrypoint decorator can be applied to sync functions or async functions.

    ### State management

    The **`previous`** parameter can be used to access the return value of the previous
    invocation of the entrypoint on the same thread id. This value is only available
    when a checkpointer is provided.

    If you want **`previous`** to be different from the return value, you can use the
    `entrypoint.final` object to return a value while saving a different value to the
    checkpoint.

    Args:
        checkpointer: Specify a checkpointer to create a workflow that can persist
            its state across runs.
        store: A generalized key-value store. Some implementations may support
            semantic search capabilities through an optional `index` configuration.
        cache: A cache to use for caching the results of the workflow.
        context_schema: Specifies the schema for the context object that will be
            passed to the workflow.
        cache_policy: A cache policy to use for caching the results of the workflow.
        retry_policy: A retry policy (or list of policies) to use for the workflow in case of a failure.

    !!! warning "`config_schema` Deprecated"
        The `config_schema` parameter is deprecated in v0.6.0 and support will be removed in v2.0.0.
        Please use `context_schema` instead to specify the schema for run-scoped context.


    Example: Using entrypoint and tasks
        ```python
        import time

        from langgraph.func import entrypoint, task
        from langgraph.types import interrupt, Command
        from langgraph.checkpoint.memory import InMemorySaver

        @task
        def compose_essay(topic: str) -> str:
            time.sleep(1.0)  # Simulate slow operation
            return f"An essay about {topic}"

        @entrypoint(checkpointer=InMemorySaver())
        def review_workflow(topic: str) -> dict:
            \"\"\"Manages the workflow for generating and reviewing an essay.

            The workflow includes:
            1. Generating an essay about the given topic.
            2. Interrupting the workflow for human review of the generated essay.

            Upon resuming the workflow, compose_essay task will not be re-executed
            as its result is cached by the checkpointer.

            Args:
                topic: The subject of the essay.

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
        from typing import Optional

        from langgraph.checkpoint.memory import MemorySaver

        from langgraph.func import entrypoint

        @entrypoint(checkpointer=InMemorySaver())
        def my_workflow(input_data: str, previous: Optional[str] = None) -> str:
            return "world"

        config = {
            "configurable": {
                "thread_id": "some_thread"
            }
        }
        my_workflow.invoke("hello", config)
        ```

    Example: Using entrypoint.final to save a value
        The `entrypoint.final` object allows you to return a value while saving
        a different value to the checkpoint. This value will be accessible
        in the next invocation of the entrypoint via the `previous` parameter, as
        long as the same thread id is used.

        ```python
        from typing import Any

        from langgraph.checkpoint.memory import MemorySaver

        from langgraph.func import entrypoint

        @entrypoint(checkpointer=InMemorySaver())
        def my_workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
            previous = previous or 0
            # This will return the previous value to the caller, saving
            # 2 * number to the checkpoint, which will be used in the next invocation
            # for the `previous` parameter.
            return entrypoint.final(value=previous, save=2 * number)

        config = {
            "configurable": {
                "thread_id": "some_thread"
            }
        }

        my_workflow.invoke(3, config)  # 0 (previous was None)
        my_workflow.invoke(1, config)  # 6 (previous was 3 * 2 from the previous invocation)
        ```
    """

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        cache: BaseCache | None = None,
        context_schema: type[ContextT] | None = None,
        cache_policy: CachePolicy | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> None:
        """Initialize the entrypoint decorator."""
        if (config_schema := kwargs.get("config_schema", MISSING)) is not MISSING:
            warnings.warn(
                "`config_schema` is deprecated and will be removed. Please use `context_schema` instead.",
                category=LangGraphDeprecatedSinceV10,
                stacklevel=2,
            )
            if context_schema is None:
                context_schema = cast(type[ContextT], config_schema)

        if (retry := kwargs.get("retry", MISSING)) is not MISSING:
            warnings.warn(
                "`retry` is deprecated and will be removed. Please use `retry_policy` instead.",
                category=LangGraphDeprecatedSinceV05,
                stacklevel=2,
            )
            if retry_policy is None:
                retry_policy = cast("RetryPolicy | Sequence[RetryPolicy]", retry)

        self.checkpointer = checkpointer
        self.store = store
        self.cache = cache
        self.cache_policy = cache_policy
        self.retry_policy = retry_policy
        self.context_schema = context_schema

    @dataclass(**_DC_KWARGS)
    class final(Generic[R, S]):
        """A primitive that can be returned from an entrypoint.

        This primitive allows to save a value to the checkpointer distinct from the
        return value from the entrypoint.

        Example: Decoupling the return value and the save value
            ```python
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.func import entrypoint

            @entrypoint(checkpointer=InMemorySaver())
            def my_workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
                previous = previous or 0
                # This will return the previous value to the caller, saving
                # 2 * number to the checkpoint, which will be used in the next invocation
                # for the `previous` parameter.
                return entrypoint.final(value=previous, save=2 * number)

            config = {
                "configurable": {
                    "thread_id": "1"
                }
            }

            my_workflow.invoke(3, config)  # 0 (previous was None)
            my_workflow.invoke(1, config)  # 6 (previous was 3 * 2 from the previous invocation)
            ```
        """

        value: R
        """Value to return. A value will always be returned even if it is None."""
        save: S
        """The value for the state for the next checkpoint.

        A value will always be saved even if it is None.
        """

    def __call__(self, func: Callable[..., Any]) -> Pregel:
        """Convert a function into a Pregel graph.

        Args:
            func: The function to convert. Support both sync and async functions.

        Returns:
            A Pregel graph.
        """
        # wrap generators in a function that writes to StreamWriter
        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
            raise NotImplementedError(
                "Generators are not supported in the Functional API."
            )

        bound = get_runnable_for_entrypoint(func)
        stream_mode: StreamMode = "updates"

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

        def _pluck_return_value(value: Any) -> Any:
            """Extract the return_ value the entrypoint.final object or passthrough."""
            return value.value if isinstance(value, entrypoint.final) else value

        def _pluck_save_value(value: Any) -> Any:
            """Get save value from the entrypoint.final object or passthrough."""
            return value.save if isinstance(value, entrypoint.final) else value

        output_type, save_type = Any, Any
        if sig.return_annotation is not inspect.Signature.empty:
            # User does not parameterize entrypoint.final properly
            if (
                sig.return_annotation is entrypoint.final
            ):  # Un-parameterized entrypoint.final
                output_type = save_type = Any
            else:
                origin = get_origin(sig.return_annotation)
                if origin is entrypoint.final:
                    type_annotations = get_args(sig.return_annotation)
                    if len(type_annotations) != 2:
                        raise TypeError(
                            "Please an annotation for both the return_ and "
                            "the save values."
                            "For example, `-> entrypoint.final[int, str]` would assign a "
                            "return_ a type of `int` and save the type `str`."
                        )
                    output_type, save_type = get_args(sig.return_annotation)
                else:
                    output_type = save_type = sig.return_annotation

        return Pregel(
            nodes={
                func.__name__: PregelNode(
                    bound=bound,
                    triggers=[START],
                    channels=START,
                    writers=[
                        ChannelWrite(
                            [
                                ChannelWriteEntry(END, mapper=_pluck_return_value),
                                ChannelWriteEntry(PREVIOUS, mapper=_pluck_save_value),
                            ]
                        )
                    ],
                )
            },
            channels={
                START: EphemeralValue(input_type),
                END: LastValue(output_type, END),
                PREVIOUS: LastValue(save_type, PREVIOUS),
            },
            input_channels=START,
            output_channels=END,
            stream_channels=END,
            stream_mode=stream_mode,
            stream_eager=True,
            checkpointer=self.checkpointer,
            store=self.store,
            cache=self.cache,
            cache_policy=self.cache_policy,
            retry_policy=self.retry_policy or (),
            context_schema=self.context_schema,  # type: ignore[arg-type]
        )
