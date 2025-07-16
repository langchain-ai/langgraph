from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, cast

from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
from langgraph.config import get_config
from langgraph.store.base import BaseStore
from langgraph.types import _DC_KWARGS, StreamWriter
from langgraph.typing import ContextT


def _no_op_stream_writer(_: Any) -> None: ...


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
    """Function that writes to the custom stream."""

    previous: Any | None
    """The previous return value for the given thread.
    
    Only available with the functional API when a checkpointer is provided."""


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
