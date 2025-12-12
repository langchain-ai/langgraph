"""Data models for interacting with the LangGraph API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import Field
from datetime import datetime
from typing import (
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    Union,
)

from typing_extensions import TypedDict

Json = dict[str, Any] | None
"""Represents a JSON-like structure, which can be None or a dictionary with string keys and any values."""

RunStatus = Literal["pending", "running", "error", "success", "timeout", "interrupted"]
"""
Represents the status of a run:
- "pending": The run is waiting to start.
- "running": The run is currently executing.
- "error": The run encountered an error and stopped.
- "success": The run completed successfully.
- "timeout": The run exceeded its time limit.
- "interrupted": The run was manually stopped or interrupted.
"""

ThreadStatus = Literal["idle", "busy", "interrupted", "error"]
"""
Represents the status of a thread:
- "idle": The thread is not currently processing any task.
- "busy": The thread is actively processing a task.
- "interrupted": The thread's execution was interrupted.
- "error": An exception occurred during task processing.
"""

ThreadStreamMode = Literal["run_modes", "lifecycle", "state_update"]
"""
Defines the mode of streaming:
- "run_modes": Stream the same events as the runs on thread, as well as run_done events.
- "lifecycle": Stream only run start/end events.
- "state_update": Stream state updates on the thread.
"""

StreamMode = Literal[
    "values",
    "messages",
    "updates",
    "events",
    "tasks",
    "checkpoints",
    "debug",
    "custom",
    "messages-tuple",
]
"""
Defines the mode of streaming:
- "values": Stream only the values.
- "messages": Stream complete messages.
- "updates": Stream updates to the state.
- "events": Stream events occurring during execution.
- "checkpoints": Stream checkpoints as they are created.
- "tasks": Stream task start and finish events.
- "debug": Stream detailed debug information.
- "custom": Stream custom events.
"""

DisconnectMode = Literal["cancel", "continue"]
"""
Specifies behavior on disconnection:
- "cancel": Cancel the operation on disconnection.
- "continue": Continue the operation even if disconnected.
"""

MultitaskStrategy = Literal["reject", "interrupt", "rollback", "enqueue"]
"""
Defines how to handle multiple tasks:
- "reject": Reject new tasks when busy.
- "interrupt": Interrupt current task for new ones.
- "rollback": Roll back current task and start new one.
- "enqueue": Queue new tasks for later execution.
"""

OnConflictBehavior = Literal["raise", "do_nothing"]
"""
Specifies behavior on conflict:
- "raise": Raise an exception when a conflict occurs.
- "do_nothing": Ignore conflicts and proceed.
"""

OnCompletionBehavior = Literal["delete", "keep"]
"""
Defines action after completion:
- "delete": Delete resources after completion.
- "keep": Retain resources after completion.
"""

Durability = Literal["sync", "async", "exit"]
"""Durability mode for the graph execution.
- `"sync"`: Changes are persisted synchronously before the next step starts.
- `"async"`: Changes are persisted asynchronously while the next step executes.
- `"exit"`: Changes are persisted only when the graph exits."""

All = Literal["*"]
"""Represents a wildcard or 'all' selector."""

IfNotExists = Literal["create", "reject"]
"""
Specifies behavior if the thread doesn't exist:
- "create": Create a new thread if it doesn't exist.
- "reject": Reject the operation if the thread doesn't exist.
"""

CancelAction = Literal["interrupt", "rollback"]
"""
Action to take when cancelling the run.
- "interrupt": Simply cancel the run.
- "rollback": Cancel the run. Then delete the run and associated checkpoints.
"""

AssistantSortBy = Literal[
    "assistant_id", "graph_id", "name", "created_at", "updated_at"
]
"""
The field to sort by.
"""

ThreadSortBy = Literal["thread_id", "status", "created_at", "updated_at"]
"""
The field to sort by.
"""

CronSortBy = Literal[
    "cron_id", "assistant_id", "thread_id", "created_at", "updated_at", "next_run_date"
]
"""
The field to sort by.
"""

SortOrder = Literal["asc", "desc"]
"""
The order to sort by.
"""


class Config(TypedDict, total=False):
    """Configuration options for a call."""

    tags: list[str]
    """
    Tags for this call and any sub-calls (eg. a Chain calling an LLM).
    You can use these to filter calls.
    """

    recursion_limit: int
    """
    Maximum number of times a call can recurse. If not provided, defaults to 25.
    """

    configurable: dict[str, Any]
    """
    Runtime values for attributes previously made configurable on this Runnable,
    or sub-Runnables, through .configurable_fields() or .configurable_alternatives().
    Check .output_schema() for a description of the attributes that have been made
    configurable.
    """


class Checkpoint(TypedDict):
    """Represents a checkpoint in the execution process."""

    thread_id: str
    """Unique identifier for the thread associated with this checkpoint."""
    checkpoint_ns: str
    """Namespace for the checkpoint; used internally to manage subgraph state."""
    checkpoint_id: str | None
    """Optional unique identifier for the checkpoint itself."""
    checkpoint_map: dict[str, Any] | None
    """Optional dictionary containing checkpoint-specific data."""


class GraphSchema(TypedDict):
    """Defines the structure and properties of a graph."""

    graph_id: str
    """The ID of the graph."""
    input_schema: dict | None
    """The schema for the graph input.
    Missing if unable to generate JSON schema from graph."""
    output_schema: dict | None
    """The schema for the graph output.
    Missing if unable to generate JSON schema from graph."""
    state_schema: dict | None
    """The schema for the graph state.
    Missing if unable to generate JSON schema from graph."""
    config_schema: dict | None
    """The schema for the graph config.
    Missing if unable to generate JSON schema from graph."""
    context_schema: dict | None
    """The schema for the graph context.
    Missing if unable to generate JSON schema from graph."""


Subgraphs = dict[str, GraphSchema]


class AssistantBase(TypedDict):
    """Base model for an assistant."""

    assistant_id: str
    """The ID of the assistant."""
    graph_id: str
    """The ID of the graph."""
    config: Config
    """The assistant config."""
    context: Context
    """The static context of the assistant."""
    created_at: datetime
    """The time the assistant was created."""
    metadata: Json
    """The assistant metadata."""
    version: int
    """The version of the assistant"""
    name: str
    """The name of the assistant"""
    description: str | None
    """The description of the assistant"""


class AssistantVersion(AssistantBase):
    """Represents a specific version of an assistant."""

    pass


class Assistant(AssistantBase):
    """Represents an assistant with additional properties."""

    updated_at: datetime
    """The last time the assistant was updated."""


class AssistantsSearchResponse(TypedDict):
    """Paginated response for assistant search results."""

    assistants: list[Assistant]
    """The assistants returned for the current search page."""
    next: str | None
    """Pagination cursor from the ``X-Pagination-Next`` response header."""


class Interrupt(TypedDict):
    """Represents an interruption in the execution flow."""

    value: Any
    """The value associated with the interrupt."""
    id: str
    """The ID of the interrupt. Can be used to resume the interrupt."""


class Thread(TypedDict):
    """Represents a conversation thread."""

    thread_id: str
    """The ID of the thread."""
    created_at: datetime
    """The time the thread was created."""
    updated_at: datetime
    """The last time the thread was updated."""
    metadata: Json
    """The thread metadata."""
    status: ThreadStatus
    """The status of the thread, one of 'idle', 'busy', 'interrupted'."""
    values: Json
    """The current state of the thread."""
    interrupts: dict[str, list[Interrupt]]
    """Mapping of task ids to interrupts that were raised in that task."""


class ThreadTask(TypedDict):
    """Represents a task within a thread."""

    id: str
    name: str
    error: str | None
    interrupts: list[Interrupt]
    checkpoint: Checkpoint | None
    state: ThreadState | None
    result: dict[str, Any] | None


class ThreadState(TypedDict):
    """Represents the state of a thread."""

    values: list[dict] | dict[str, Any]
    """The state values."""
    next: Sequence[str]
    """The next nodes to execute. If empty, the thread is done until new input is
    received."""
    checkpoint: Checkpoint
    """The ID of the checkpoint."""
    metadata: Json
    """Metadata for this state"""
    created_at: str | None
    """Timestamp of state creation"""
    parent_checkpoint: Checkpoint | None
    """The ID of the parent checkpoint. If missing, this is the root checkpoint."""
    tasks: Sequence[ThreadTask]
    """Tasks to execute in this step. If already attempted, may contain an error."""
    interrupts: list[Interrupt]
    """Interrupts which were thrown in this thread."""


class ThreadUpdateStateResponse(TypedDict):
    """Represents the response from updating a thread's state."""

    checkpoint: Checkpoint
    """Checkpoint of the latest state."""


class Run(TypedDict):
    """Represents a single execution run."""

    run_id: str
    """The ID of the run."""
    thread_id: str
    """The ID of the thread."""
    assistant_id: str
    """The assistant that was used for this run."""
    created_at: datetime
    """The time the run was created."""
    updated_at: datetime
    """The last time the run was updated."""
    status: RunStatus
    """The status of the run. One of 'pending', 'running', "error", 'success', "timeout", "interrupted"."""
    metadata: Json
    """The run metadata."""
    multitask_strategy: MultitaskStrategy
    """Strategy to handle concurrent runs on the same thread."""


class Cron(TypedDict):
    """Represents a scheduled task."""

    cron_id: str
    """The ID of the cron."""
    assistant_id: str
    """The ID of the assistant."""
    thread_id: str | None
    """The ID of the thread."""
    end_time: datetime | None
    """The end date to stop running the cron."""
    schedule: str
    """The schedule to run, cron format."""
    created_at: datetime
    """The time the cron was created."""
    updated_at: datetime
    """The last time the cron was updated."""
    payload: dict
    """The run payload to use for creating new run."""
    user_id: str | None
    """The user ID of the cron."""
    next_run_date: datetime | None
    """The next run date of the cron."""
    metadata: dict
    """The metadata of the cron."""


# Select field aliases for client-side typing of `select` parameters.
# These mirror the server's allowed field sets.

AssistantSelectField = Literal[
    "assistant_id",
    "graph_id",
    "name",
    "description",
    "config",
    "context",
    "created_at",
    "updated_at",
    "metadata",
    "version",
]

ThreadSelectField = Literal[
    "thread_id",
    "created_at",
    "updated_at",
    "metadata",
    "config",
    "context",
    "status",
    "values",
    "interrupts",
]

RunSelectField = Literal[
    "run_id",
    "thread_id",
    "assistant_id",
    "created_at",
    "updated_at",
    "status",
    "metadata",
    "kwargs",
    "multitask_strategy",
]

CronSelectField = Literal[
    "cron_id",
    "assistant_id",
    "thread_id",
    "end_time",
    "schedule",
    "created_at",
    "updated_at",
    "user_id",
    "payload",
    "next_run_date",
    "metadata",
    "now",
    "on_run_completed",
]

PrimitiveData = str | int | float | bool | None

QueryParamTypes = (
    Mapping[str, PrimitiveData | Sequence[PrimitiveData]]
    | list[tuple[str, PrimitiveData]]
    | tuple[tuple[str, PrimitiveData], ...]
    | str
    | bytes
)


class RunCreate(TypedDict):
    """Defines the parameters for initiating a background run."""

    thread_id: str | None
    """The identifier of the thread to run. If not provided, the run is stateless."""
    assistant_id: str
    """The identifier of the assistant to use for this run."""
    input: dict | None
    """Initial input data for the run."""
    metadata: dict | None
    """Additional metadata to associate with the run."""
    config: Config | None
    """Configuration options for the run."""
    context: Context | None
    """The static context of the run."""
    checkpoint_id: str | None
    """The identifier of a checkpoint to resume from."""
    interrupt_before: list[str] | None
    """List of node names to interrupt execution before."""
    interrupt_after: list[str] | None
    """List of node names to interrupt execution after."""
    webhook: str | None
    """URL to send webhook notifications about the run's progress."""
    multitask_strategy: MultitaskStrategy | None
    """Strategy for handling concurrent runs on the same thread."""


class Item(TypedDict):
    """Represents a single document or data entry in the graph's Store.

    Items are used to store cross-thread memories.
    """

    namespace: list[str]
    """The namespace of the item. A namespace is analogous to a document's directory."""
    key: str
    """The unique identifier of the item within its namespace.

    In general, keys needn't be globally unique.
    """
    value: dict[str, Any]
    """The value stored in the item. This is the document itself."""
    created_at: datetime
    """The timestamp when the item was created."""
    updated_at: datetime
    """The timestamp when the item was last updated."""


class ListNamespaceResponse(TypedDict):
    """Response structure for listing namespaces."""

    namespaces: list[list[str]]
    """A list of namespace paths, where each path is a list of strings."""


class SearchItem(Item, total=False):
    """Item with an optional relevance score from search operations.

    Attributes:
        score (Optional[float]): Relevance/similarity score. Included when
            searching a compatible store with a natural language query.
    """

    score: float | None


class SearchItemsResponse(TypedDict):
    """Response structure for searching items."""

    items: list[SearchItem]
    """A list of items matching the search criteria."""


class StreamPart(NamedTuple):
    """Represents a part of a stream response."""

    event: str
    """The type of event for this stream part."""
    data: dict
    """The data payload associated with the event."""
    id: str | None = None
    """The ID of the event."""


class Send(TypedDict):
    """Represents a message to be sent to a specific node in the graph.

    This type is used to explicitly send messages to nodes in the graph, typically
    used within Command objects to control graph execution flow.
    """

    node: str
    """The name of the target node to send the message to."""
    input: dict[str, Any] | None
    """Optional dictionary containing the input data to be passed to the node.

    If None, the node will be called with no input."""


class Command(TypedDict, total=False):
    """Represents one or more commands to control graph execution flow and state.

    This type defines the control commands that can be returned by nodes to influence
    graph execution. It lets you navigate to other nodes, update graph state,
    and resume from interruptions.
    """

    goto: Send | str | Sequence[Send | str]
    """Specifies where execution should continue. Can be:

        - A string node name to navigate to
        - A Send object to execute a node with specific input
        - A sequence of node names or Send objects to execute in order
    """
    update: dict[str, Any] | Sequence[tuple[str, Any]]
    """Updates to apply to the graph's state. Can be:

        - A dictionary of state updates to merge
        - A sequence of (key, value) tuples for ordered updates
    """
    resume: Any
    """Value to resume execution with after an interruption.
       Used in conjunction with interrupt() to implement control flow.
    """


class RunCreateMetadata(TypedDict):
    """Metadata for a run creation request."""

    run_id: str
    """The ID of the run."""

    thread_id: str | None
    """The ID of the thread."""


class _TypedDictLikeV1(Protocol):
    """Protocol to represent types that behave like TypedDicts

    Version 1: using `ClassVar` for keys."""

    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]


class _TypedDictLikeV2(Protocol):
    """Protocol to represent types that behave like TypedDicts

    Version 2: not using `ClassVar` for keys."""

    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]


class _DataclassLike(Protocol):
    """Protocol to represent types that behave like dataclasses.

    Inspired by the private _DataclassT from dataclasses that uses a similar protocol as a bound.
    """

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


class _BaseModelLike(Protocol):
    """Protocol to represent types that behave like Pydantic `BaseModel`."""

    model_config: ClassVar[dict[str, Any]]
    __pydantic_core_schema__: ClassVar[Any]

    def model_dump(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


_JSONLike: TypeAlias = None | str | int | float | bool
_JSONMap: TypeAlias = Mapping[
    str, Union[_JSONLike, list[_JSONLike], "_JSONMap", list["_JSONMap"]]
]

Input: TypeAlias = (
    _TypedDictLikeV1 | _TypedDictLikeV2 | _DataclassLike | _BaseModelLike | _JSONMap
)

Context: TypeAlias = Input
