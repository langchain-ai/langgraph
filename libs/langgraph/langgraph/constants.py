import sys
from types import MappingProxyType
from typing import Any, Mapping

from langgraph.types import Interrupt, Send  # noqa: F401

# Interrupt, Send re-exported for backwards compatibility


# --- Empty read-only containers ---
EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})
EMPTY_SEQ: tuple[str, ...] = tuple()

# --- Public constants ---
TAG_HIDDEN = sys.intern("langsmith:hidden")
# tag to hide a node/edge from certain tracing/streaming environments
START = sys.intern("__start__")
# the first (maybe virtual) node in graph-style Pregel
END = sys.intern("__end__")
# the last (maybe virtual) node in graph-style Pregel

# --- Reserved write keys ---
INPUT = sys.intern("__input__")
# for values passed as input to the graph
INTERRUPT = sys.intern("__interrupt__")
# for dynamic interrupts raised by nodes
ERROR = sys.intern("__error__")
# for errors raised by nodes
NO_WRITES = sys.intern("__no_writes__")
# marker to signal node didn't write anything
SCHEDULED = sys.intern("__scheduled__")
# marker to signal node was scheduled (in distributed mode)
TASKS = sys.intern("__pregel_tasks")
# for Send objects returned by nodes/edges, corresponds to PUSH below

# --- Reserved config.configurable keys ---
CONFIG_KEY_SEND = sys.intern("__pregel_send")
# holds the `write` function that accepts writes to state/edges/reserved keys
CONFIG_KEY_READ = sys.intern("__pregel_read")
# holds the `read` function that returns a copy of the current state
CONFIG_KEY_CHECKPOINTER = sys.intern("__pregel_checkpointer")
# holds a `BaseCheckpointSaver` passed from parent graph to child graphs
CONFIG_KEY_STREAM = sys.intern("__pregel_stream")
# holds a `StreamProtocol` passed from parent graph to child graphs
CONFIG_KEY_STREAM_WRITER = sys.intern("__pregel_stream_writer")
# holds a `StreamWriter` for stream_mode=custom
CONFIG_KEY_STORE = sys.intern("__pregel_store")
# holds a `BaseStore` made available to managed values
CONFIG_KEY_RESUMING = sys.intern("__pregel_resuming")
# holds a boolean indicating if subgraphs should resume from a previous checkpoint
CONFIG_KEY_TASK_ID = sys.intern("__pregel_task_id")
# holds the task ID for the current task
CONFIG_KEY_DEDUPE_TASKS = sys.intern("__pregel_dedupe_tasks")
# holds a boolean indicating if tasks should be deduplicated (for distributed mode)
CONFIG_KEY_ENSURE_LATEST = sys.intern("__pregel_ensure_latest")
# holds a boolean indicating whether to assert the requested checkpoint is the latest
# (for distributed mode)
CONFIG_KEY_DELEGATE = sys.intern("__pregel_delegate")
# holds a boolean indicating whether to delegate subgraphs (for distributed mode)
CONFIG_KEY_CHECKPOINT_MAP = sys.intern("checkpoint_map")
# holds a mapping of checkpoint_ns -> checkpoint_id for parent graphs
CONFIG_KEY_CHECKPOINT_ID = sys.intern("checkpoint_id")
# holds the current checkpoint_id, if any
CONFIG_KEY_CHECKPOINT_NS = sys.intern("checkpoint_ns")
# holds the current checkpoint_ns, "" for root graph

# --- Other constants ---
PUSH = sys.intern("__pregel_push")
# denotes push-style tasks, ie. those created by Send objects
PULL = sys.intern("__pregel_pull")
# denotes pull-style tasks, ie. those triggered by edges
NS_SEP = sys.intern("|")
# for checkpoint_ns, separates each level (ie. graph|subgraph|subsubgraph)
NS_END = sys.intern(":")
# for checkpoint_ns, for each level, separates the namespace from the task_id
CONF = sys.intern("configurable")
# key for the configurable dict in RunnableConfig

RESERVED = {
    TAG_HIDDEN,
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
    NS_SEP,
    NS_END,
    CONF,
}
