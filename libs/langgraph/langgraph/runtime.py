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

    !!! version-added "Added in version 1.0.0.
    """

    context: ContextT

    store: BaseStore | None

    stream_writer: StreamWriter

    previous: Any | None


DEFAULT_RUNTIME = Runtime(
    context=None,
    store=None,
    stream_writer=_no_op_stream_writer,
    previous=None,
)
