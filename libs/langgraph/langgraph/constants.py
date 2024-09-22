from types import MappingProxyType
from typing import Any, Mapping


# Interrupt, Send re-exported for backwards compatibility
def __getattr__(name: str) -> Any:
    if name in globals():
        return globals()[name]
    elif name == "Interrupt":
        from langgraph.pregel.types import Interrupt

        return Interrupt
    elif name == "Send":
        from langgraph.pregel.types import Send

        return Send
    raise AttributeError(f"module {__name__} has no attribute {name}")


# --- Empty read-only containers ---
EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})
EMPTY_SEQ: tuple[str, ...] = tuple()

# --- Reserved write keys ---
INPUT = "__input__"
# for values passed as input to the graph
INTERRUPT = "__interrupt__"
# for dynamic interrupts raised by nodes
ERROR = "__error__"
# for errors raised by nodes
NO_WRITES = "__no_writes__"
# marker to signal node didn't write anything
SCHEDULED = "__scheduled__"
# marker to signal node was scheduled (in distributed mode)
TASKS = "__pregel_tasks"
# for Send objects returned by nodes/edges, corresponds to PUSH below
START = "__start__"
# marker for the first (maybe virtual) node in graph-style Pregel
END = "__end__"
# marker for the last (maybe virtual) node in graph-style Pregel

# --- Reserved config.configurable keys ---
CONFIG_KEY_SEND = "__pregel_send"
# holds the `write` function that accepts writes to state/edges/reserved keys
CONFIG_KEY_READ = "__pregel_read"
# holds the `read` function that returns a copy of the current state
CONFIG_KEY_CHECKPOINTER = "__pregel_checkpointer"
# holds a `BaseCheckpointSaver` passed from parent graph to child graphs
CONFIG_KEY_STREAM = "__pregel_stream"
# holds a `StreamProtocol` passed from parent graph to child graphs
CONFIG_KEY_STREAM_WRITER = "__pregel_stream_writer"
# holds a `StreamWriter` for stream_mode=custom
CONFIG_KEY_STORE = "__pregel_store"
# holds a `BaseStore` made available to managed values
CONFIG_KEY_RESUMING = "__pregel_resuming"
# holds a boolean indicating if subgraphs should resume from a previous checkpoint
CONFIG_KEY_TASK_ID = "__pregel_task_id"
# holds the task ID for the current task
CONFIG_KEY_DEDUPE_TASKS = "__pregel_dedupe_tasks"
# holds a boolean indicating if tasks should be deduplicated (for distributed mode)
CONFIG_KEY_ENSURE_LATEST = "__pregel_ensure_latest"
# holds a boolean indicating whether to assert the requested checkpoint is the latest
# (for distributed mode)
CONFIG_KEY_DELEGATE = "__pregel_delegate"
# holds a boolean indicating whether to delegate subgraphs (for distributed mode)
CONFIG_KEY_CHECKPOINT_MAP = "checkpoint_map"
# holds a mapping of checkpoint_ns -> checkpoint_id for parent graphs
CONFIG_KEY_CHECKPOINT_ID = "checkpoint_id"
# holds the current checkpoint_id, if any
CONFIG_KEY_CHECKPOINT_NS = "checkpoint_ns"
# holds the current checkpoint_ns, "" for root graph

# --- Other constants ---
PUSH = "__pregel_push"
# denotes push-style tasks, ie. those created by Send objects
PULL = "__pregel_pull"
# denotes pull-style tasks, ie. those triggered by edges
RUNTIME_PLACEHOLDER = "__pregel_runtime_placeholder__"
# placeholder for managed values replaced at runtime
TAG_HIDDEN = "langsmith:hidden"
# tag to hide a node/edge from certain tracing/streaming environments
NS_SEP = "|"
# for checkpoint_ns, separates each level (ie. graph|subgraph|subsubgraph)
NS_END = ":"
# for checkpoint_ns, for each level, separates the namespace from the task_id

RESERVED = {
    # reserved write keys
    INPUT,
    INTERRUPT,
    ERROR,
    NO_WRITES,
    SCHEDULED,
    TASKS,
    # reserved config.configurable keys
    CONFIG_KEY_SEND,
    CONFIG_KEY_READ,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_STREAM_WRITER,
    CONFIG_KEY_STORE,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_DEDUPE_TASKS,
    CONFIG_KEY_ENSURE_LATEST,
    CONFIG_KEY_DELEGATE,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    # other constants
    PUSH,
    PULL,
    RUNTIME_PLACEHOLDER,
    TAG_HIDDEN,
    NS_SEP,
    NS_END,
}
