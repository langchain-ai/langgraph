package pregel

import (
	"context"
	"sync"
)

// Constants for task types and reserved keys
const (
	// Task types
	PUSH = "__pregel_push" // Denotes push-style tasks, ie. those created by Send objects
	PULL = "__pregel_pull" // Denotes pull-style tasks, ie. those triggered by edges

	// Reserved write keys
	INPUT     = "__input__"      // For values passed as input to the graph
	INTERRUPT = "__interrupt__"  // For dynamic interrupts raised by nodes
	RESUME    = "__resume__"     // For values passed to resume a node after an interrupt
	ERROR     = "__error__"      // For errors raised by nodes
	NO_WRITES = "__no_writes__"  // Marker to signal node didn't write anything
	SCHEDULED = "__scheduled__"  // Marker to signal node was scheduled (in distributed mode)
	TASKS     = "__pregel_tasks" // For Send objects returned by nodes/edges
	RETURN    = "__return__"     // For writes of a task where we simply record the return value

	// Public constants
	START    = "__start__"    // The first (maybe virtual) node in graph-style Pregel
	END      = "__end__"      // The last (maybe virtual) node in graph-style Pregel
	SELF     = "__self__"     // The implicit branch that handles each node's Control values
	PREVIOUS = "__previous__" // Previous value

	// Other constants
	NS_SEP       = "|"                                    // For checkpoint_ns, separates each level (ie. graph|subgraph|subsubgraph)
	NS_END       = ":"                                    // For checkpoint_ns, for each level, separates the namespace from the task_id
	NULL_TASK_ID = "00000000-0000-0000-0000-000000000000" // The task_id to use for writes that are not associated with a task
	CONF         = "configurable"                         // Key for the configurable dict in RunnableConfig

	// Reserved config.configurable keys
	CONFIG_KEY_SEND              = "__pregel_send"              // Holds the `write` function that accepts writes to state/edges/reserved keys
	CONFIG_KEY_READ              = "__pregel_read"              // Holds the `read` function that returns a copy of the current state
	CONFIG_KEY_CALL              = "__pregel_call"              // Holds the `call` function that accepts a node/func, args and returns a future
	CONFIG_KEY_CHECKPOINTER      = "__pregel_checkpointer"      // Holds a `BaseCheckpointSaver` passed from parent graph to child graphs
	CONFIG_KEY_STREAM            = "__pregel_stream"            // Holds a `StreamProtocol` passed from parent graph to child graphs
	CONFIG_KEY_STREAM_WRITER     = "__pregel_stream_writer"     // Holds a `StreamWriter` for stream_mode=custom
	CONFIG_KEY_STORE             = "__pregel_store"             // Holds a `BaseStore` made available to managed values
	CONFIG_KEY_CACHE             = "__pregel_cache"             // Holds a `BaseCache` made available to subgraphs
	CONFIG_KEY_RESUMING          = "__pregel_resuming"          // Holds a boolean indicating if subgraphs should resume from a previous checkpoint
	CONFIG_KEY_TASK_ID           = "__pregel_task_id"           // Holds the task ID for the current task
	CONFIG_KEY_DEDUPE_TASKS      = "__pregel_dedupe_tasks"      // Holds a boolean indicating if tasks should be deduplicated (for distributed mode)
	CONFIG_KEY_ENSURE_LATEST     = "__pregel_ensure_latest"     // Holds a boolean indicating whether to assert the requested checkpoint is the latest
	CONFIG_KEY_DELEGATE          = "__pregel_delegate"          // Holds a boolean indicating whether to delegate subgraphs (for distributed mode)
	CONFIG_KEY_THREAD_ID         = "thread_id"                  // Holds the thread ID for the current invocation
	CONFIG_KEY_CHECKPOINT_MAP    = "checkpoint_map"             // Holds a mapping of checkpoint_ns -> checkpoint_id for parent graphs
	CONFIG_KEY_CHECKPOINT_ID     = "checkpoint_id"              // Holds the current checkpoint_id, if any
	CONFIG_KEY_CHECKPOINT_NS     = "checkpoint_ns"              // Holds the current checkpoint_ns, "" for root graph
	CONFIG_KEY_NODE_FINISHED     = "__pregel_node_finished"     // Holds a callback to be called when a node is finished
	CONFIG_KEY_SCRATCHPAD        = "__pregel_scratchpad"        // Holds a mutable dict for temporary storage scoped to the current task
	CONFIG_KEY_PREVIOUS          = "__pregel_previous"          // Holds the previous return value from a stateful Pregel graph
	CONFIG_KEY_RUNNER_SUBMIT     = "__pregel_runner_submit"     // Holds a function that receives tasks from runner, executes them and returns results
	CONFIG_KEY_CHECKPOINT_DURING = "__pregel_checkpoint_during" // Holds a boolean indicating whether to checkpoint during the run (or only at the end)
	CONFIG_KEY_RESUME_MAP        = "__pregel_resume_map"        // Holds a mapping of task ns -> resume value for resuming tasks
	TAG_HIDDEN                   = "langsmith:hidden"           // Holds a boolean indicating whether to hide a node/edge from certain tracing/streaming environments.
)

// StreamMode defines how the graph streams its output
type StreamMode string

// WRITES_IDX_MAP maps special channel names to negative indices
// to avoid conflicts with regular writes.
var WRITES_IDX_MAP = map[string]int{
	ERROR:     -1,
	SCHEDULED: -2,
	INTERRUPT: -3,
	RESUME:    -4,
}

// TS
// export type PendingWriteValue = unknown;

// export type PendingWrite<Channel = string> = [Channel, PendingWriteValue];

// export type CheckpointPendingWrite<TaskId = string> = [
//   TaskId,
//   ...PendingWrite<string>
// ];
// Py
// PendingWrite = Tuple[str, str, Any]

type PendingWrite struct {
	TaskID  string
	Channel string
	Value   interface{}
}

const (
	// StreamValues emits all values in the state after each step
	StreamValues StreamMode = "values"
	// StreamUpdates emits only the node or task names and updates
	StreamUpdates StreamMode = "updates"
	// StreamCustom emits custom data from inside nodes or tasks
	StreamCustom StreamMode = "custom"
	// StreamMessages emits LLM messages token-by-token
	StreamMessages StreamMode = "messages"
	// StreamDebug emits debug events with as much information as possible
	StreamDebug StreamMode = "debug"
)

// PregelTask represents a task in the Pregel system

type PregelTask struct {
	ID         string
	Name       string
	Path       []interface{}
	Error      error
	Interrupts []interface{}
	Result     interface{}
}

// PregelExecutableTask represents a task that can be executed
type PregelExecutableTask struct {
	PregelTask
	Input       interface{}
	Node        NodeRunnable
	Writes      []Write
	Config      RunnableConfig
	Triggers    []string
	RetryPolicy interface{}
	CacheKey    *CacheKey
	Writers     map[string]interface{} // Flat writers
	Subgraphs   map[string]interface{} // Subgraphs
}

// StreamChunk is what the consumer receives.
type StreamChunk struct {
	Namespace []string // sub-graph path (reserved for future use)
	Mode      StreamMode
	Payload   any
}

type StreamOptions struct {
	Mode             StreamMode
	OutputChannels   []string        // defaults to all non-context channels
	InterruptBefore  []string        // interrupt gate (before)
	InterruptAfter   []string        // interrupt gate (after)
	MaxConcurrency   int             // overrides config[ "max_concurrency" ]
	CheckpointDuring *bool           // nil → inherit config
	Debug            *bool           // nil → inherit graph.debug
	Context          context.Context // optional, default = context.Background()
}

// CacheKey represents a key for caching
type CacheKey struct {
	Namespace []string
	Key       string
	TTL       int64
}

type PregelNode struct {
	Node        NodeRunnable
	Triggers    []string
	Metadata    map[string]interface{}
	Tags        []string
	CachePolicy interface{} // CachePolicy equivalent
	RetryPolicy interface{} // RetryPolicy equivalent
	FlatWriters map[string]interface{}
	Subgraphs   map[string]interface{}
	Retry       RetryPolicy
}

type NodeRunnable interface {
	Invoke(ctx context.Context, input any, cfg RunnableConfig, loop LoopCallback) ([]Write, error)
}

type Write struct {
	Channel string
	Value   any
}

// Checkpoint represents a checkpoint in the Pregel system
type Checkpoint struct {
	ID              string
	ChannelValues   map[string]interface{} `json:"channel_values,omitempty"`
	ChannelVersions map[string]int64       `json:"channel_versions,omitempty"`
	VersionsSeen    map[string]interface{} `json:"versions_seen,omitempty"`
	PendingSends    []Send                 `json:"pending_sends,omitempty"`
	Version         int                    `json:"version,omitempty"`
	Timestamp       string                 `json:"timestamp,omitempty"`
}

// NewCheckpoint creates a new Checkpoint with all fields properly initialized
func NewCheckpoint() Checkpoint {
	return Checkpoint{
		ChannelValues:   make(map[string]interface{}),
		ChannelVersions: make(map[string]int64),
		VersionsSeen:    make(map[string]interface{}),
		PendingSends:    make([]Send, 0),
	}
}

// Send represents a message to be sent to a node
type Send struct {
	Node string
	Arg  interface{}
}

// Call represents a function call
type Call struct {
	Func        interface{}   // Function to call
	Input       []interface{} // Arguments
	Callbacks   interface{}   // Callbacks
	CachePolicy interface{}   // CachePolicy
	Retry       interface{}   // RetryPolicy
}

// PregelTaskWrites represents writes from a task
type PregelTaskWrites struct {
	Path     []interface{}
	Name     string
	Writes   []interface{} // Deque in Python
	Triggers []string
}

// ---------------------------------------------------------------------------
// Interfaces from previous snippets (slim versions here)
// ---------------------------------------------------------------------------

type RetryPolicy struct {
	MaxAttempts int
	BackoffMs   int
}

type BaseChannel interface {
	Set(v any)
	Get() any
}

type simpleChan struct{ val atomicValue }

type atomicValue struct {
	mu sync.RWMutex
	v  any
}

func (a *atomicValue) Store(v any) {
	a.mu.Lock()
	a.v = v
	a.mu.Unlock()
}
func (a *atomicValue) Load() (v any) { a.mu.RLock(); v = a.v; a.mu.RUnlock(); return }

func (c *simpleChan) Set(v any) { c.val.Store(v) }
func (c *simpleChan) Get() any  { return c.val.Load() }

// Managed values -------------------------------------------------------------

type WritableManagedValue interface {
	Update([]any) error
}

type ManagedValueMapping map[string]WritableManagedValue

type SendPacket struct {
	Node string
	Arg  any
}

type BaseCheckpointSaver interface {
	Put(cfg RunnableConfig, cp Checkpoint, md map[string]any, newVers map[string]int) error
	GetTuple(cfg RunnableConfig) (*Checkpoint, error)
}

// Stores ---------------------------------------------------------------------

type BaseStore interface{}

// Loop callback interface passed to Nodes for localWrite / localRead
type LoopCallback interface {
	Send(taskID string, writes []Write)
	Read(selectKeys []string) map[string]any
	AcceptPush(originTask PregelExecutableTask, writeIdx int, call *Call) (*PregelExecutableTask, error)
}

// RunnableConfig represents configuration for a Runnable.
// Fields are optional
type RunnableConfig struct {
	Tags           []string               `json:"tags,omitempty"`            // Tags for this call and sub-calls.
	Metadata       map[string]interface{} `json:"metadata,omitempty"`        // Metadata for this call and sub-calls.
	Callbacks      interface{}            `json:"callbacks,omitempty"`       // Callbacks for this call and sub-calls.
	RunName        *string                `json:"run_name,omitempty"`        // Name for the tracer run for this call.
	MaxConcurrency int                    `json:"max_concurrency,omitempty"` // Max number of parallel calls.
	RecursionLimit int                    `json:"recursion_limit,omitempty"` // Max recursion depth.
	Configurable   map[string]interface{} `json:"configurable,omitempty"`    // Runtime values for configurable attributes.
	RunID          *string                `json:"run_id,omitempty"`          // Unique identifier for the tracer run (UUID as string).
}
