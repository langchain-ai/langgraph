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
)
from langgraph.errors import NotInTaskError
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


def _task_conf(caller: str, config: RunnableConfig | None) -> dict[str, Any]:
    if config is None:
        if sys.version_info < (3, 11) and _in_asyncio_task():
            msg = f"Pass `config` to {caller} in async code on Python < 3.11"
            raise RuntimeError(msg)
        config = var_child_runnable_config.get()
    conf = config.get(CONF, {}) if config else {}
    if CONFIG_KEY_READ not in conf:
        msg = f"Called {caller} outside of a graph node or task"
        raise NotInTaskError(msg)
    return conf


def _in_asyncio_task() -> bool:
    try:
        return asyncio.current_task() is not None
    except RuntimeError:
        return False


def read_state(*keys: str, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Read graph state from inside a node or tool, with its own updates applied.

    A node receives state as it was when the step started. This returns the
    current values: that state plus the updates the running node has queued,
    for example with [`write_state`][langgraph.config.write_state], each applied
    through the key's reducer. Updates from other nodes in the same step are
    not visible until the step ends.

    !!! version-added "Added in version 1.2.23"

    Args:
        *keys: The state keys to read. Reads every state key when omitted.
        config: The config the node received. Defaults to the node running in
            the current context. Required in async code on Python < 3.11.

    Returns:
        The channel values by key, as a dict whatever the state schema is. A
            key with no value is left out.

    Raises:
        NotInTaskError: When called outside of a graph node or task.

    Example:
        ```python
        from langgraph.config import read_state, write_state


        def add_note(state: State) -> None:
            write_state({"notes": ["first"]})
            assert read_state("notes") == {"notes": [*state["notes"], "first"]}
        ```
    """
    conf = _task_conf("read_state", config)
    return conf[CONFIG_KEY_READ](list(keys) or None, True)


def write_state(
    update: Mapping[str, Any], *, config: RunnableConfig | None = None
) -> None:
    """Queue a state update from inside a node or tool.

    The update is applied when the step ends, as an update the node returns is,
    and [`read_state`][langgraph.config.read_state] in the same node sees it at
    once. It is checked before it is queued. If the node already wrote a key
    that takes one value per step, this raises and queues nothing.

    !!! version-added "Added in version 1.2.23"

    Args:
        update: The values to write, by state key.
        config: The config the node received. Defaults to the node running in
            the current context. Required in async code on Python < 3.11.

    Raises:
        InvalidUpdateError: When a key is not a state key, or takes one value
            per step and the node already wrote it.
        NotInTaskError: When called outside of a graph node or task.
    """
    conf = _task_conf("write_state", config)
    writes = list(update.items())
    conf[CONFIG_KEY_READ]([key for key, _ in writes], True, proposed=writes)
    conf[CONFIG_KEY_SEND](writes)
