import asyncio
import sys
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.store.base import BaseStore

from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
from langgraph.types import StreamWriter


def _no_op_stream_writer(c: Any) -> None:
    pass


def get_config() -> RunnableConfig:
    if sys.version_info < (3, 11):
        try:
            if asyncio.current_task():
                raise RuntimeError(
                    "Python 3.11 or later required to use this in an async context"
                )
        except RuntimeError:
            pass
    if var_config := var_child_runnable_config.get():
        return var_config
    else:
        raise RuntimeError("Called get_config outside of a runnable context")


def get_store() -> BaseStore:
    """Access LangGraph store from inside a graph node or entrypoint task at runtime.

    Can be called from inside any [StateGraph][langgraph.graph.StateGraph] node or
    functional API [task][langgraph.func.task], as long as the StateGraph or the [entrypoint][langgraph.func.entrypoint]
    was initialized with a store, e.g.:

    ```python
    # with StateGraph
    graph = (
        StateGraph(...)
        ...
        .compile(store=store)
    )

    # or with entrypoint
    @entrypoint(store=store)
    def workflow(inputs):
        ...
    ```

    !!! warning "Async with Python < 3.11"

        If you are using Python < 3.11 and are running LangGraph asynchronously,
        `get_store()` won't work since it uses [contextvar](https://docs.python.org/3/library/contextvars.html) propagation (only available in [Python >= 3.11](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)).


    Example: Using with StateGraph
        ```python
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START
        from langgraph.store.memory import InMemoryStore
        from langgraph.config import get_store

        store = InMemoryStore()
        store.put(("values",), "foo", {"bar": 2})


        class State(TypedDict):
            foo: int


        def my_node(state: State):
            my_store = get_store()
            stored_value = my_store.get(("values",), "foo").value["bar"]
            return {"foo": stored_value + 1}


        graph = (
            StateGraph(State)
            .add_node(my_node)
            .add_edge(START, "my_node")
            .compile(store=store)
        )

        graph.invoke({"foo": 1})
        ```

        ```pycon
        {"foo": 3}
        ```

    Example: Using with functional API
        ```python
        from langgraph.func import entrypoint, task
        from langgraph.store.memory import InMemoryStore
        from langgraph.config import get_store

        store = InMemoryStore()
        store.put(("values",), "foo", {"bar": 2})


        @task
        def my_task(value: int):
            my_store = get_store()
            stored_value = my_store.get(("values",), "foo").value["bar"]
            return stored_value + 1


        @entrypoint(store=store)
        def workflow(value: int):
            return my_task(value).result()


        workflow.invoke(1)
        ```

        ```pycon
        3
        ```
    """
    return get_config()[CONF][CONFIG_KEY_RUNTIME].store


def get_stream_writer() -> StreamWriter:
    """Access LangGraph [StreamWriter][langgraph.types.StreamWriter] from inside a graph node or entrypoint task at runtime.

    Can be called from inside any [StateGraph][langgraph.graph.StateGraph] node or
    functional API [task][langgraph.func.task].

    !!! warning "Async with Python < 3.11"

        If you are using Python < 3.11 and are running LangGraph asynchronously,
        `get_stream_writer()` won't work since it uses [contextvar](https://docs.python.org/3/library/contextvars.html) propagation (only available in [Python >= 3.11](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)).

    Example: Using with StateGraph
        ```python
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START
        from langgraph.config import get_stream_writer


        class State(TypedDict):
            foo: int


        def my_node(state: State):
            my_stream_writer = get_stream_writer()
            my_stream_writer({"custom_data": "Hello!"})
            return {"foo": state["foo"] + 1}


        graph = (
            StateGraph(State)
            .add_node(my_node)
            .add_edge(START, "my_node")
            .compile(store=store)
        )

        for chunk in graph.stream({"foo": 1}, stream_mode="custom"):
            print(chunk)
        ```

        ```pycon
        {"custom_data": "Hello!"}
        ```

    Example: Using with functional API
        ```python
        from langgraph.func import entrypoint, task
        from langgraph.config import get_stream_writer


        @task
        def my_task(value: int):
            my_stream_writer = get_stream_writer()
            my_stream_writer({"custom_data": "Hello!"})
            return value + 1


        @entrypoint(store=store)
        def workflow(value: int):
            return my_task(value).result()


        for chunk in workflow.stream(1, stream_mode="custom"):
            print(chunk)
        ```

        ```pycon
        {"custom_data": "Hello!"}
        ```
    """
    runtime = get_config()[CONF][CONFIG_KEY_RUNTIME]
    return runtime.stream_writer
