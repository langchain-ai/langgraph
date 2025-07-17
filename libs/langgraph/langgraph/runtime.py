from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Generic, cast

from typing_extensions import TypedDict, Unpack

from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
from langgraph.config import get_config
from langgraph.store.base import BaseStore
from langgraph.types import _DC_KWARGS, StreamWriter
from langgraph.typing import ContextT


def _no_op_stream_writer(_: Any) -> None: ...


class _RuntimeOverrides(TypedDict, Generic[ContextT], total=False):
    context: ContextT
    store: BaseStore | None
    stream_writer: StreamWriter
    previous: Any


@dataclass(**_DC_KWARGS)
class Runtime(Generic[ContextT]):
    """Convenience class that bundles run-scoped context and graph configuration.

    !!! version-added "Added in version 1.0.0."
    """

    context: ContextT = field(default=None)  # type: ignore[assignment]
    """Static context for the graph run, like user_id, db_conn, etc.
    
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
            previous=other.previous or self.previous,
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
    """Get the runtime for the current graph run."""

    # TODO: in an ideal world, we would have a context manager for
    # the runtime that's independent of the config. this will follow
    # from the removal of the configurable packing
    runtime = cast(Runtime[ContextT], get_config()[CONF].get(CONFIG_KEY_RUNTIME))
    return runtime
