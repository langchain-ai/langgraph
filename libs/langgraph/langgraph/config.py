import asyncio
import sys
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.store.base import BaseStore

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_READ,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SCRATCHPAD,
    CONFIG_KEY_SEND,
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
    config = var_child_runnable_config.get()
    conf = config.get(CONF, {}) if config else {}
    if CONFIG_KEY_READ not in conf:
        msg = f"Called {caller} outside of a graph node or task"
        raise NotInTaskError(msg)
    return conf


def _state_lock(conf: dict[str, Any]) -> AbstractContextManager[Any]:
    # `update_state` runs a node's writers on one thread, with no scratchpad
    scratchpad = conf.get(CONFIG_KEY_SCRATCHPAD)
    return scratchpad.state_lock if scratchpad is not None else nullcontext()


def read_state(keys: str | Sequence[str] | None = None) -> dict[str, Any]:
    """Read graph state from inside a node, tool or task, with the task's own writes applied.

    The state a node receives is a snapshot from the start of the step. This reads
    the current value of each key instead: the snapshot, plus every update the
    running task has queued so far with [`write_state`][langgraph.config.write_state],
    `push_message` or a state-backed tool, each passed through the key's reducer.
    Updates from other tasks in the same step are not visible until the step ends.

    Args:
        keys: The state keys to read. Reads every key when omitted.

    Returns:
        The values by key. A key with no value is left out.

    Raises:
        KeyError: When a key is not a state key.
        NotInTaskError: When called outside of a graph node or task.

    Example:
        ```python
        from langchain_core.tools import tool
        from langgraph.config import read_state, write_state


        @tool
        def add_note(note: str) -> str:
            "Save a note."
            notes = read_state("notes").get("notes", [])
            write_state({"notes": [*notes, note]})
            return f"Saved note {len(notes) + 1}"
        ```
    """
    conf = _task_conf("read_state")
    if isinstance(keys, str):
        keys = [keys]
    with _state_lock(conf):
        return conf[CONFIG_KEY_READ](keys if keys is None else list(keys), True)


def write_state(update: Mapping[str, Any] | Sequence[tuple[str, Any]]) -> None:
    """Queue a state update from inside a node, tool or task.

    The update is applied when the step ends, the same way as an update the node
    returns, and [`read_state`][langgraph.config.read_state] in the same task sees it
    at once. It is checked against the task's earlier updates before it is queued.
    If a key would reject it, nothing in `update` is queued.

    The check covers the running task only. Two tasks that write the same
    one-value key in one step still fail when the step ends.

    Args:
        update: The values to write by state key. Use a sequence of
            `(key, value)` tuples to write one key more than once.

    Raises:
        InvalidUpdateError: When a key is not a state key, or when the key accepts
            one value per step and the task already wrote it.
        NotInTaskError: When called outside of a graph node or task.
    """
    conf = _task_conf("write_state")
    writes = list(update.items() if isinstance(update, Mapping) else update)
    for write in writes:
        if not (
            isinstance(write, tuple) and len(write) == 2 and isinstance(write[0], str)
        ):
            msg = (
                f"Expected a dict or a sequence of (key, value) tuples, got {update!r}"
            )
            raise InvalidUpdateError(msg)
    with _state_lock(conf):
        conf[CONFIG_KEY_READ]([key for key, _ in writes], True, writes)
        conf[CONFIG_KEY_SEND](writes)
