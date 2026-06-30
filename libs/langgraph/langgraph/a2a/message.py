"""
A2A Protocol Message types.

This module defines the message types used for communication between
agents using the A2A protocol, based on JSON-RPC 2.0.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class A2AMessageType(str, Enum):
    """Types of A2A protocol messages."""

    # Task lifecycle messages
    TASK_CREATE = "task.create"
    TASK_STATUS = "task.status"
    TASK_CANCEL = "task.cancel"
    TASK_RESULT = "task.result"

    # Discovery messages
    AGENT_DISCOVER = "agent.discover"
    AGENT_CARD = "agent.card"

    # Capability messages
    CAPABILITY_QUERY = "capability.query"
    CAPABILITY_RESPONSE = "capability.response"

    # Error messages
    ERROR = "error"


class A2ATaskStatus(str, Enum):
    """Status of an A2A task."""

    PENDING = "pending"
    """Task has been received but not yet started."""

    RUNNING = "running"
    """Task is currently being executed."""

    COMPLETED = "completed"
    """Task has completed successfully."""

    FAILED = "failed"
    """Task has failed with an error."""

    CANCELLED = "cancelled"
    """Task was cancelled before completion."""

    INTERRUPTED = "interrupted"
    """Task is paused awaiting human input."""


@dataclass(slots=True)
class A2AMessage:
    """
    Base class for A2A protocol messages.

    All A2A messages follow JSON-RPC 2.0 format with additional
    A2A-specific fields.

    Attributes:
        message_id: Unique identifier for this message.
        message_type: The type of this message.
        sender_id: Identifier of the sending agent.
        receiver_id: Identifier of the receiving agent (optional for broadcasts).
        timestamp: ISO 8601 timestamp of message creation.
        payload: The message payload.
    """

    message_type: A2AMessageType
    """The type of this message."""

    payload: dict[str, Any] = field(default_factory=dict)
    """The message payload."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this message."""

    sender_id: str | None = None
    """Identifier of the sending agent."""

    receiver_id: str | None = None
    """Identifier of the receiving agent (optional for broadcasts)."""

    correlation_id: str | None = None
    """ID of the message this is responding to."""

    timestamp: str | None = None
    """ISO 8601 timestamp of message creation."""

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary representation."""
        result: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self.message_id,
            "method": self.message_type.value,
            "params": self.payload,
        }
        if self.sender_id:
            result["sender_id"] = self.sender_id
        if self.receiver_id:
            result["receiver_id"] = self.receiver_id
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.timestamp:
            result["timestamp"] = self.timestamp
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AMessage:
        """Create message from dictionary representation."""
        return cls(
            message_type=A2AMessageType(data.get("method", "error")),
            payload=data.get("params", {}),
            message_id=data.get("id", str(uuid.uuid4())),
            sender_id=data.get("sender_id"),
            receiver_id=data.get("receiver_id"),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp"),
        )


@dataclass(slots=True)
class A2ARequest:
    """
    An A2A protocol request for task execution.

    Attributes:
        task_id: Unique identifier for this task.
        skill: The skill/capability being requested.
        input_data: Input data for the task.
        context: Additional context for task execution.
        timeout_seconds: Optional timeout for task execution.
        priority: Task priority (higher = more urgent).
    """

    skill: str
    """The skill/capability being requested."""

    input_data: dict[str, Any] = field(default_factory=dict)
    """Input data for the task."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this task."""

    context: dict[str, Any] = field(default_factory=dict)
    """Additional context for task execution."""

    timeout_seconds: float | None = None
    """Optional timeout for task execution."""

    priority: int = 0
    """Task priority (higher = more urgent)."""

    sender_agent_id: str | None = None
    """ID of the agent making this request."""

    def to_message(self) -> A2AMessage:
        """Convert to an A2A message for transmission."""
        return A2AMessage(
            message_type=A2AMessageType.TASK_CREATE,
            payload={
                "task_id": self.task_id,
                "skill": self.skill,
                "input": self.input_data,
                "context": self.context,
                "timeout_seconds": self.timeout_seconds,
                "priority": self.priority,
            },
            sender_id=self.sender_agent_id,
        )

    @classmethod
    def from_message(cls, message: A2AMessage) -> A2ARequest:
        """Create a request from an A2A message."""
        payload = message.payload
        return cls(
            skill=payload.get("skill", ""),
            input_data=payload.get("input", {}),
            task_id=payload.get("task_id", str(uuid.uuid4())),
            context=payload.get("context", {}),
            timeout_seconds=payload.get("timeout_seconds"),
            priority=payload.get("priority", 0),
            sender_agent_id=message.sender_id,
        )


@dataclass(slots=True)
class A2AResponse:
    """
    An A2A protocol response for a completed task.

    Attributes:
        task_id: The ID of the task this responds to.
        status: Current status of the task.
        result: The result data (if completed successfully).
        error: Error information (if failed).
        progress: Progress percentage (0-100) for running tasks.
    """

    task_id: str
    """The ID of the task this responds to."""

    status: A2ATaskStatus
    """Current status of the task."""

    result: dict[str, Any] | None = None
    """The result data (if completed successfully)."""

    error: dict[str, Any] | None = None
    """Error information (if failed)."""

    progress: float | None = None
    """Progress percentage (0-100) for running tasks."""

    responder_agent_id: str | None = None
    """ID of the agent sending this response."""

    def to_message(self, correlation_id: str | None = None) -> A2AMessage:
        """Convert to an A2A message for transmission."""
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status.value,
        }
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        if self.progress is not None:
            payload["progress"] = self.progress

        message_type = (
            A2AMessageType.TASK_RESULT
            if self.status in (A2ATaskStatus.COMPLETED, A2ATaskStatus.FAILED)
            else A2AMessageType.TASK_STATUS
        )

        return A2AMessage(
            message_type=message_type,
            payload=payload,
            sender_id=self.responder_agent_id,
            correlation_id=correlation_id,
        )

    @classmethod
    def from_message(cls, message: A2AMessage) -> A2AResponse:
        """Create a response from an A2A message."""
        payload = message.payload
        return cls(
            task_id=payload.get("task_id", ""),
            status=A2ATaskStatus(payload.get("status", "pending")),
            result=payload.get("result"),
            error=payload.get("error"),
            progress=payload.get("progress"),
            responder_agent_id=message.sender_id,
        )

    @classmethod
    def success(
        cls,
        task_id: str,
        result: dict[str, Any],
        responder_agent_id: str | None = None,
    ) -> A2AResponse:
        """Create a successful response.

        Args:
            task_id: The task this responds to.
            result: The successful result data.
            responder_agent_id: ID of the responding agent.

        Returns:
            A2AResponse with completed status.
        """
        return cls(
            task_id=task_id,
            status=A2ATaskStatus.COMPLETED,
            result=result,
            responder_agent_id=responder_agent_id,
        )

    @classmethod
    def create_error(
        cls,
        task_id: str,
        error_message: str,
        error_code: str | None = None,
        responder_agent_id: str | None = None,
    ) -> A2AResponse:
        """Create an error response.

        Args:
            task_id: The task this responds to.
            error_message: Human-readable error message.
            error_code: Optional error code.
            responder_agent_id: ID of the responding agent.

        Returns:
            A2AResponse with failed status.
        """
        error_data: dict[str, Any] = {"message": error_message}
        if error_code:
            error_data["code"] = error_code
        return cls(
            task_id=task_id,
            status=A2ATaskStatus.FAILED,
            error=error_data,
            responder_agent_id=responder_agent_id,
        )
