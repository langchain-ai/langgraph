"""Data models for interacting with the LangGraph API."""

from datetime import datetime
from typing import Any, Dict, Literal, NamedTuple, Optional, Sequence, TypedDict, Union

Json = Optional[dict[str, Any]]
"""Represents a JSON-like structure, which can be None or a dictionary with string keys and any values."""

RunStatus = Literal["pending", "error", "success", "timeout", "interrupted"]
"""
Represents the status of a run:
- "pending": The run is waiting to start.
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

StreamMode = Literal[
    "values", "messages", "updates", "events", "debug", "custom", "messages-tuple"
]
"""
Defines the mode of streaming:
- "values": Stream only the values.
- "messages": Stream complete messages.
- "updates": Stream updates to the state.
- "events": Stream events occurring during execution.
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
    """Namespace for the checkpoint, used for organization and retrieval."""
    checkpoint_id: Optional[str]
    """Optional unique identifier for the checkpoint itself."""
    checkpoint_map: Optional[dict[str, Any]]
    """Optional dictionary containing checkpoint-specific data."""


class GraphSchema(TypedDict):
    """Defines the structure and properties of a graph."""

    graph_id: str
    """The ID of the graph."""
    input_schema: Optional[dict]
    """The schema for the graph input.
    Missing if unable to generate JSON schema from graph."""
    output_schema: Optional[dict]
    """The schema for the graph output.
    Missing if unable to generate JSON schema from graph."""
    state_schema: Optional[dict]
    """The schema for the graph state.
    Missing if unable to generate JSON schema from graph."""
    config_schema: Optional[dict]
    """The schema for the graph config.
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
    created_at: datetime
    """The time the assistant was created."""
    metadata: Json
    """The assistant metadata."""
    version: int
    """The version of the assistant"""


class AssistantVersion(AssistantBase):
    """Represents a specific version of an assistant."""

    pass


class Assistant(AssistantBase):
    """Represents an assistant with additional properties."""

    updated_at: datetime
    """The last time the assistant was updated."""
    name: str
    """The name of the assistant"""


class Interrupt(TypedDict, total=False):
    """Represents an interruption in the execution flow."""

    value: Any
    """The value associated with the interrupt."""
    when: Literal["during"]
    """When the interrupt occurred."""
    resumable: bool
    """Whether the interrupt can be resumed."""
    ns: Optional[list[str]]
    """Optional namespace for the interrupt."""


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
    interrupts: Dict[str, list[Interrupt]]
    """Interrupts which were thrown in this thread"""


class ThreadTask(TypedDict):
    """Represents a task within a thread."""

    id: str
    name: str
    error: Optional[str]
    interrupts: list[Interrupt]
    checkpoint: Optional[Checkpoint]
    state: Optional["ThreadState"]
    result: Optional[dict[str, Any]]


class ThreadState(TypedDict):
    """Represents the state of a thread."""

    values: Union[list[dict], dict[str, Any]]
    """The state values."""
    next: Sequence[str]
    """The next nodes to execute. If empty, the thread is done until new input is 
    received."""
    checkpoint: Checkpoint
    """The ID of the checkpoint."""
    metadata: Json
    """Metadata for this state"""
    created_at: Optional[str]
    """Timestamp of state creation"""
    parent_checkpoint: Optional[Checkpoint]
    """The ID of the parent checkpoint. If missing, this is the root checkpoint."""
    tasks: Sequence[ThreadTask]
    """Tasks to execute in this step. If already attempted, may contain an error."""


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
    thread_id: Optional[str]
    """The ID of the thread."""
    end_time: Optional[datetime]
    """The end date to stop running the cron."""
    schedule: str
    """The schedule to run, cron format."""
    created_at: datetime
    """The time the cron was created."""
    updated_at: datetime
    """The last time the cron was updated."""
    payload: dict
    """The run payload to use for creating new run."""


class RunCreate(TypedDict):
    """Defines the parameters for initiating a background run."""

    thread_id: Optional[str]
    """The identifier of the thread to run. If not provided, the run is stateless."""
    assistant_id: str
    """The identifier of the assistant to use for this run."""
    input: Optional[dict]
    """Initial input data for the run."""
    metadata: Optional[dict]
    """Additional metadata to associate with the run."""
    config: Optional[Config]
    """Configuration options for the run."""
    checkpoint_id: Optional[str]
    """The identifier of a checkpoint to resume from."""
    interrupt_before: Optional[list[str]]
    """List of node names to interrupt execution before."""
    interrupt_after: Optional[list[str]]
    """List of node names to interrupt execution after."""
    webhook: Optional[str]
    """URL to send webhook notifications about the run's progress."""
    multitask_strategy: Optional[MultitaskStrategy]
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

    score: Optional[float]


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


class Send(TypedDict):
    node: str
    input: Optional[dict[str, Any]]


class Command(TypedDict, total=False):
    goto: Union[Send, str, Sequence[Union[Send, str]]]
    update: dict[str, Any]
    resume: Any
