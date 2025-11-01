from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Generic, cast

from langgraph.store.base import BaseStore
from typing_extensions import TypedDict, Unpack

from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
from langgraph.config import get_config
from langgraph.types import _DC_KWARGS, StreamWriter
from langgraph.typing import ContextT

__all__ = ("Runtime", "get_runtime")


def _no_op_stream_writer(_: Any) -> None: ...


class _RuntimeOverrides(TypedDict, Generic[ContextT], total=False):
    context: ContextT
    store: BaseStore | None
    stream_writer: StreamWriter
    previous: Any


@dataclass(**_DC_KWARGS)
class Runtime(Generic[ContextT]):
    """Convenience class that bundles run-scoped context and other runtime utilities.

    !!! version-added "Added in version v0.6.0"

    Example:

    ```python
    from typing import TypedDict
    from langgraph.graph import StateGraph
    from dataclasses import dataclass
    from langgraph.runtime import Runtime
    from langgraph.store.memory import InMemoryStore


    @dataclass
    class Context:  # (1)!
        user_id: str


    class State(TypedDict, total=False):
        response: str


    store = InMemoryStore()  # (2)!
    store.put(("users",), "user_123", {"name": "Alice"})


    def personalized_greeting(state: State, runtime: Runtime[Context]) -> State:
        '''Generate personalized greeting using runtime context and store.'''
        user_id = runtime.context.user_id  # (3)!
        name = "unknown_user"
        if runtime.store:
            if memory := runtime.store.get(("users",), user_id):
                name = memory.value["name"]

        response = f"Hello {name}! Nice to see you again."
        return {"response": response}


    graph = (
        StateGraph(state_schema=State, context_schema=Context)
        .add_node("personalized_greeting", personalized_greeting)
        .set_entry_point("personalized_greeting")
        .set_finish_point("personalized_greeting")
        .compile(store=store)
    )

    result = graph.invoke({}, context=Context(user_id="user_123"))
    print(result)
    # > {'response': 'Hello Alice! Nice to see you again.'}
    ```

    1. Define a schema for the runtime context.
    2. Create a store to persist memories and other information.
    3. Use the runtime context to access the `user_id`.
    """

    context: ContextT = field(default=None)  # type: ignore[assignment]
    """Static context for the graph run, like `user_id`, `db_conn`, etc.
    
    Can also be thought of as 'run dependencies'."""

    store: BaseStore | None = field(default=None)
    """Store for the graph run, enabling persistence and memory."""

    stream_writer: StreamWriter = field(default=_no_op_stream_writer)
    """Function that writes to the custom stream."""

    previous: Any = field(default=None)
    """The previous return value for the given thread.
    
    Only available with the functional API when a checkpointer is provided.
    """

    def merge(self, other: Runtime[ContextT]) -> Runtime[ContextT]:
        """Merge two runtimes together.

        If a value is not provided in the other runtime, the value from the current runtime is used.
        """
        return Runtime(
            context=other.context or self.context,
            store=other.store or self.store,
            stream_writer=other.stream_writer
            if other.stream_writer is not _no_op_stream_writer
            else self.stream_writer,
            previous=self.previous if other.previous is None else other.previous,
        )

    def override(
        self, **overrides: Unpack[_RuntimeOverrides[ContextT]]
    ) -> Runtime[ContextT]:
        """Replace the runtime with a new runtime with the given overrides."""
        return replace(self, **overrides)


DEFAULT_RUNTIME = Runtime(
    context=None,
    store=None,
    stream_writer=_no_op_stream_writer,
    previous=None,
)


def get_runtime(context_schema: type[ContextT] | None = None) -> Runtime[ContextT]:
    """Get the runtime for the current graph run.

    Args:
        context_schema: Optional schema used for type hinting the return type of the runtime.

    Returns:
        The runtime for the current graph run.
    """

    # TODO: in an ideal world, we would have a context manager for
    # the runtime that's independent of the config. this will follow
    # from the removal of the configurable packing
    runtime = cast(Runtime[ContextT], get_config()[CONF].get(CONFIG_KEY_RUNTIME))
    return runtime
