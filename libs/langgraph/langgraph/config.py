import asyncio
import sys
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.store.base import BaseStore

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_READ,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SEND,
    CONFIG_KEY_STATE_KEYS,
)
from langgraph.errors import InvalidUpdateError, NotInTaskError
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

    Can be called from inside any [`StateGraph`][langgraph.graph.StateGraph] node or
    functional API [`task`][langgraph.func.task], as long as the `StateGraph` or the [`entrypoint`][langgraph.func.entrypoint]
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
        `get_store()` won't work since it uses [`contextvar`](https://docs.python.org/3/library/contextvars.html) propagation (only available in [Python >= 3.11](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)).


    Example: Using with `StateGraph`
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
    """Access LangGraph [`StreamWriter`][langgraph.types.StreamWriter] from inside a graph node or entrypoint task at runtime.

    Can be called from inside any [`StateGraph`][langgraph.graph.StateGraph] node or
    functional API [`task`][langgraph.func.task].

    !!! warning "Async with Python < 3.11"

        If you are using Python < 3.11 and are running LangGraph asynchronously,
        `get_stream_writer()` won't work since it uses [`contextvar`](https://docs.python.org/3/library/contextvars.html) propagation (only available in [Python >= 3.11](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)).

    Example: Using with `StateGraph`
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


def _task_conf(caller: str) -> dict[str, Any]:
    conf = (var_child_runnable_config.get() or {}).get(CONF, {})
    if CONFIG_KEY_READ not in conf:
        msg = f"Called {caller} outside of a graph node or task"
        raise NotInTaskError(msg)
    return conf


def read_state(*keys: str) -> dict[str, Any]:
    """Read graph state from inside a node or tool, with its own updates applied.

    A node receives state as it was when the step started. This also applies
    the updates the node has queued since, such as with
    [`write_state`][langgraph.config.write_state]. A key that holds its value
    for one step, such as an `EphemeralValue`, reads as unset unless the node
    wrote it. Like `get_store`, this does not work in async code on
    Python < 3.11.

    !!! version-added "Added in version 1.2.23"

    Args:
        *keys: The state keys to read. Reads every state key when omitted.

    Returns:
        The values by key. A key with no value is left out.

    Raises:
        KeyError: When a key is not a state key.
        NotInTaskError: When called outside of a graph node or task.
    """
    conf = _task_conf("read_state")
    state_keys = conf.get(CONFIG_KEY_STATE_KEYS, ())
    for key in keys:
        if key not in state_keys:
            msg = f"'{key}' is not a state key"
            raise KeyError(msg)
    return conf[CONFIG_KEY_READ](list(keys or state_keys), True)


def write_state(update: Mapping[str, Any]) -> None:
    """Queue a state update from inside a node or tool.

    The update is applied when the step ends, and
    [`read_state`][langgraph.config.read_state] sees it at once. Like
    `get_store`, this does not work in async code on Python < 3.11.

    !!! version-added "Added in version 1.2.23"

    Args:
        update: The values to write, by state key.

    Raises:
        InvalidUpdateError: When a key is not a state key, or takes one value
            per step and the node already wrote it. Nothing is queued.
        NotInTaskError: When called outside of a graph node or task.
    """
    conf = _task_conf("write_state")
    state_keys = conf.get(CONFIG_KEY_STATE_KEYS, ())
    for key in update:
        if key not in state_keys:
            msg = f"Cannot write to '{key}': it is not a state key"
            raise InvalidUpdateError(msg)
    writes = list(update.items())
    conf[CONFIG_KEY_READ](list(update), True, proposed=writes)
    conf[CONFIG_KEY_SEND](writes)
