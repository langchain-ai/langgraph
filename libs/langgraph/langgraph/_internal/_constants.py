"""Constants used for Pregel operations."""

import sys
from typing import Literal, cast

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
PREVIOUS = sys.intern("__previous__")
# the implicit branch that handles each node's Control values


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
CONFIG_KEY_RUNNER_SUBMIT = sys.intern("__pregel_runner_submit")
# holds a function that receives tasks from runner, executes them and returns results
CONFIG_KEY_DURABILITY = sys.intern("__pregel_durability")
# holds the durability mode, one of "sync", "async", or "exit"
CONFIG_KEY_RUNTIME = sys.intern("__pregel_runtime")
# holds a `Runtime` instance with context, store, stream writer, etc.
CONFIG_KEY_RESUME_MAP = sys.intern("__pregel_resume_map")
# holds a mapping of task ns -> resume value for resuming tasks

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
OVERWRITE = sys.intern("__overwrite__")
# dict key for the overwrite value, used as `{'__overwrite__': value}`

# redefined to avoid circular import with langgraph.constants
_TAG_HIDDEN = sys.intern("langsmith:hidden")

RESERVED = {
    _TAG_HIDDEN,
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
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_RESUME_MAP,
    # other constants
    PUSH,
    PULL,
    NS_SEP,
    NS_END,
    CONF,
}
