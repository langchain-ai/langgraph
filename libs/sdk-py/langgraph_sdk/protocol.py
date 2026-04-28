"""Typed protocol messages for thread-centric remote streaming.

These shapes mirror the JSON protocol used by the JavaScript SDK's
``ThreadStream`` layer.  The SDK keeps them lightweight and dependency-free so
``langgraph`` can build higher-level projections on top without creating a
package cycle.
"""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict

Channel = Literal[
    "values",
    "updates",
    "messages",
    "tools",
    "custom",
    "lifecycle",
    "input",
    "debug",
    "checkpoints",
    "tasks",
]
"""Built-in subscribable protocol channels."""

CommandMethod = Literal[
    "run.input",
    "subscription.subscribe",
    "subscription.unsubscribe",
    "agent.getTree",
    "input.respond",
    "input.inject",
    "state.get",
    "state.listCheckpoints",
    "state.fork",
]
"""Command methods understood by the thread stream protocol."""


class SubscribeParams(TypedDict, total=False):
    """Filter used when subscribing to thread protocol events."""

    channels: list[Channel | str]
    namespaces: NotRequired[list[list[str]]]
    depth: NotRequired[int]


class Command(TypedDict):
    """Command sent to a thread protocol transport."""

    id: str
    method: CommandMethod
    params: dict[str, Any]


class CommandResponse(TypedDict):
    """Successful command response."""

    id: str
    result: Any


class ErrorResponse(TypedDict):
    """Error command response."""

    id: str
    error: dict[str, Any]


class EventParams(TypedDict):
    """Protocol event parameters."""

    namespace: list[str]
    timestamp: int
    data: Any
    node: NotRequired[str]
    run_id: NotRequired[str]
    interrupts: NotRequired[list[Any]]


class Event(TypedDict):
    """Protocol event envelope."""

    type: Literal["event"]
    method: str
    params: EventParams
    event_id: NotRequired[str]
    seq: NotRequired[int]


class RunInputResult(TypedDict, total=False):
    """Result returned by ``run.input`` commands."""

    run_id: str
    thread_id: str


class SubscribeResult(TypedDict):
    """Result returned by ``subscription.subscribe`` commands."""

    subscription_id: str

