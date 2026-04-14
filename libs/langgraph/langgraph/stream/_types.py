"""Protocol types for StreamingHandler.

Re-exports CDDL-derived types from ``langchain-protocol`` and defines
in-process-only types needed by the LangGraph streaming infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Re-exports from langchain-protocol (CDDL-derived)
# ---------------------------------------------------------------------------
# Primitives
# Content blocks
# Messages data
# Tools data
from langchain_protocol import (
    Annotation,
    Citation,
    ContentBlock,
    ContentBlockDeltaData,
    ContentBlockFinishData,
    ContentBlockStartData,
    FinalizedContentBlock,
    FinishReason,
    InvalidToolCallBlock,
    MessageErrorData,
    MessageFinishData,
    MessageMetadata,
    MessageRole,
    MessagesData,
    MessageStartData,
    MetadataScalar,
    Namespace,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolCallChunkBlock,
    ToolErrorData,
    ToolFinishedData,
    ToolOutputDeltaData,
    ToolsData,
    ToolStartedData,
    UsageInfo,
)
from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# In-process types (not in the CDDL spec)
# ---------------------------------------------------------------------------


class _ProtocolEventParams(TypedDict):
    """Payload envelope for a :class:`ProtocolEvent`."""

    namespace: Namespace
    node: NotRequired[str]
    data: Any


class ProtocolEvent(TypedDict):
    """A single protocol event emitted by the StreamingHandler infrastructure.

    ``method`` corresponds to a
    :pydata:`~langgraph.types.StreamMode` value (``"messages"``,
    ``"updates"``, etc.).
    """

    type: str  # always "event"
    seq: NotRequired[int]  # assigned by StreamMux.push(); absent before push()
    method: str  # StreamMode value
    params: _ProtocolEventParams


class StreamTransformer(ABC):
    """Extension point for custom stream projections.

    Implementations are registered with ``StreamingHandler`` and receive every
    :class:`ProtocolEvent` before it is appended to the event log.

    Any :class:`~langgraph.stream.stream_channel.StreamChannel` instances
    returned by ``init()`` are automatically wired to the protocol event
    stream by the mux.

    """

    @abstractmethod
    def init(self) -> Any:
        """Return the initial projection value.

        Called once before the run.  Any
        :class:`~langgraph.stream.stream_channel.StreamChannel` instances
        in the return value are automatically wired by the mux.
        """
        ...

    @abstractmethod
    def process(self, event: ProtocolEvent) -> bool:
        """Process an event.

        Return ``True`` to keep the event in the log, ``False`` to suppress
        it.
        """
        ...

    def finalize(self) -> None:
        """Called once when the run completes successfully.

        Optional — the mux auto-closes any :class:`StreamChannel` instances,
        so transformers that only use channels can omit this.
        """

    def fail(self, err: BaseException) -> None:
        """Called once when the run fails.

        Optional — the mux auto-fails any :class:`StreamChannel` instances,
        so transformers that only use channels can omit this.
        """


class InterruptPayload(TypedDict):
    """An interrupt produced during a StreamingHandler run."""

    interrupt_id: str
    payload: Any


__all__ = [
    # Primitives (re-exported)
    "Namespace",
    "MessageRole",
    "MessageMetadata",
    "MetadataScalar",
    # Content blocks (re-exported)
    "TextBlock",
    "ReasoningBlock",
    "ToolCallBlock",
    "ToolCallChunkBlock",
    "InvalidToolCallBlock",
    "ContentBlock",
    "FinalizedContentBlock",
    "Annotation",
    "Citation",
    # Messages data (re-exported)
    "MessagesData",
    "MessageStartData",
    "ContentBlockStartData",
    "ContentBlockDeltaData",
    "ContentBlockFinishData",
    "MessageFinishData",
    "MessageErrorData",
    "FinishReason",
    "UsageInfo",
    # Tools data (re-exported)
    "ToolsData",
    "ToolStartedData",
    "ToolOutputDeltaData",
    "ToolFinishedData",
    "ToolErrorData",
    # In-process types
    "ProtocolEvent",
    "StreamTransformer",
    "InterruptPayload",
]
