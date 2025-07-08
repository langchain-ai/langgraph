from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

from langgraph.store.base import BaseStore
from langgraph.types import _DC_KWARGS, StreamWriter
from langgraph.typing import ContextT

_no_op_stream_writer = lambda _: None


@dataclass(**_DC_KWARGS)
class Runtime(Generic[ContextT]):
    """Convenience class that bundles run-scoped context and graph configuration.

    !!! version-added "Added in version 1.0.0."
    """

    context: ContextT
    """Static context for the graph run, like user_id, db_conn, etc.
    
    Can also be thought of as 'run dependencies'."""

    store: BaseStore | None
    """Store for the graph run, enabling persistence and memory."""

    stream_writer: StreamWriter
    """Function to write to the stream."""

    previous: Any | None
    """The previous return value for the given thread (available only when a checkpointer is provided)."""


DEFAULT_RUNTIME = Runtime(
    context=None,
    store=None,
    stream_writer=_no_op_stream_writer,
    previous=None,
)
