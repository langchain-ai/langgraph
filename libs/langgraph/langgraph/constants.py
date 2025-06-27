import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, cast

from langgraph.types import Interrupt, Send  # noqa: F401

# Interrupt, Send re-exported for backwards compatibility


# --- Empty read-only containers ---
EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})
EMPTY_SEQ: tuple[str, ...] = tuple()
MISSING = object()

# --- Public constants ---
TAG_NOSTREAM = sys.intern("nostream")
"""Tag to disable streaming for a chat model."""
TAG_NOSTREAM_ALT = sys.intern("langsmith:nostream")
"""Tag to disable streaming for a chat model. (Deprecated in favour of "nostream")"""
TAG_HIDDEN = sys.intern("langsmith:hidden")
"""Tag to hide a node/edge from certain tracing/streaming environments."""
START = sys.intern("__start__")
"""The first (maybe virtual) node in graph-style Pregel."""
END = sys.intern("__end__")
"""The last (maybe virtual) node in graph-style Pregel."""
SELF = sys.intern("__self__")
"""The implicit branch that handles each node's Control values."""
PREVIOUS = sys.intern("__previous__")

# --- Reserved write keys ---
INPUT = sys.intern("__input__")
# for values passed as input to the graph
INTERRUPT = sys.intern("__interrupt__")
# for dynamic interrupts raised by nodes
RESUME = sys.intern("__resume__")
# for values passed to resume a node after an interrupt
ERROR = sys.intern("__error__")
# for errors raised by nodes
NO_WRITES = sys.intern("__no_writes__")
# marker to signal node didn't write anything
TASKS = sys.intern("__pregel_tasks")
# for Send objects returned by nodes/edges, corresponds to PUSH below
RETURN = sys.intern("__return__")
# for writes of a task where we simply record the return value

# --- Reserved cache namespaces ---
CACHE_NS_WRITES = sys.intern("__pregel_ns_writes")
# cache namespace for node writes

# --- Reserved config.configurable keys ---
CONFIG_KEY_SEND = sys.intern("__pregel_send")
# holds the `write` function that accepts writes to state/edges/reserved keys
CONFIG_KEY_READ = sys.intern("__pregel_read")
# holds the `read` function that returns a copy of the current state
CONFIG_KEY_CALL = sys.intern("__pregel_call")
# holds the `call` function that accepts a node/func, args and returns a future
CONFIG_KEY_CHECKPOINTER = sys.intern("__pregel_checkpointer")
# holds a `BaseCheckpointSaver` passed from parent graph to child graphs
CONFIG_KEY_STREAM = sys.intern("__pregel_stream")
# holds a `StreamProtocol` passed from parent graph to child graphs
CONFIG_KEY_STREAM_WRITER = sys.intern("__pregel_stream_writer")
# holds a `StreamWriter` for stream_mode=custom
CONFIG_KEY_STORE = sys.intern("__pregel_store")
# holds a `BaseStore` made available to managed values
CONFIG_KEY_CACHE = sys.intern("__pregel_cache")
# holds a `BaseCache` made available to subgraphs
CONFIG_KEY_RESUMING = sys.intern("__pregel_resuming")
# holds a boolean indicating if subgraphs should resume from a previous checkpoint
CONFIG_KEY_TASK_ID = sys.intern("__pregel_task_id")
# holds the task ID for the current task
CONFIG_KEY_THREAD_ID = sys.intern("thread_id")
# holds the thread ID for the current invocation
CONFIG_KEY_CHECKPOINT_MAP = sys.intern("checkpoint_map")
# holds a mapping of checkpoint_ns -> checkpoint_id for parent graphs
CONFIG_KEY_CHECKPOINT_ID = sys.intern("checkpoint_id")
# holds the current checkpoint_id, if any
CONFIG_KEY_CHECKPOINT_NS = sys.intern("checkpoint_ns")
# holds the current checkpoint_ns, "" for root graph
CONFIG_KEY_NODE_FINISHED = sys.intern("__pregel_node_finished")
# holds a callback to be called when a node is finished
CONFIG_KEY_SCRATCHPAD = sys.intern("__pregel_scratchpad")
# holds a mutable dict for temporary storage scoped to the current task
CONFIG_KEY_PREVIOUS = sys.intern("__pregel_previous")
# holds the previous return value from a stateful Pregel graph.
CONFIG_KEY_RUNNER_SUBMIT = sys.intern("__pregel_runner_submit")
# holds a function that receives tasks from runner, executes them and returns results
CONFIG_KEY_CHECKPOINT_DURING = sys.intern("__pregel_checkpoint_during")
# holds a boolean indicating whether to checkpoint during the run (or only at the end)

# --- Other constants ---
PUSH = sys.intern("__pregel_push")
# denotes push-style tasks, ie. those created by Send objects
PULL = sys.intern("__pregel_pull")
# denotes pull-style tasks, ie. those triggered by edges
NS_SEP = sys.intern("|")
# for checkpoint_ns, separates each level (ie. graph|subgraph|subsubgraph)
NS_END = sys.intern(":")
# for checkpoint_ns, for each level, separates the namespace from the task_id
CONF = cast(Literal["configurable"], sys.intern("configurable"))
# key for the configurable dict in RunnableConfig
NULL_TASK_ID = sys.intern("00000000-0000-0000-0000-000000000000")
# the task_id to use for writes that are not associated with a task
CONFIG_KEY_RESUME_MAP = sys.intern("__pregel_resume_map")
# holds a mapping of task ns -> resume value for resuming tasks

RESERVED = {
    TAG_HIDDEN,
    # reserved write keys
    INPUT,
    INTERRUPT,
    RESUME,
    ERROR,
    NO_WRITES,
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
