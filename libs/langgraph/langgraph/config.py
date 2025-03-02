from typing import Any

from langgraph.constants import CONF, CONFIG_KEY_STORE, CONFIG_KEY_STREAM_WRITER
from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter
from langgraph.utils.config import RunnableConfig, var_child_runnable_config


def _no_op_stream_writer(c: Any) -> None:
    pass


def get_config() -> RunnableConfig:
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

    ```

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
        {'foo': 3}
        ```
    """
    config = get_config()
    return config[CONF][CONFIG_KEY_STORE]


def get_stream_writer() -> StreamWriter:
    """Access LangGraph [StreamWriter][langgraph.types.StreamWriter] from inside a graph node or entrypoint task at runtime.

    Can be called from inside any [StateGraph][langgraph.graph.StateGraph] node.

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
        {'custom_data': 'Hello!'}
        ```
    """
    config = get_config()
    return config[CONF].get(CONFIG_KEY_STREAM_WRITER, _no_op_stream_writer)
