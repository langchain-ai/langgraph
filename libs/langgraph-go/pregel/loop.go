package pregel

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"
)

type GraphInterrupt struct {
	Interrupts any
}

func (e GraphInterrupt) Error() string { return "graph interrupted" }

type GraphDelegate struct {
	Payload map[string]any
}

func (e GraphDelegate) Error() string { return "graph delegation requested" }

func hashID(checkpointID string, parts ...string) string {
	b, _ := hex.DecodeString(checkpointID)
	h := sha256.New()
	h.Write(b)
	for _, p := range parts {
		h.Write([]byte(p))
	}
	return hex.EncodeToString(h.Sum(nil))
}

type PregelLoop struct {
	ctx         context.Context
	cancel      context.CancelFunc
	cfg         RunnableConfig
	store       BaseStore
	checkpoint  Checkpoint
	checkporter BaseCheckpointSaver

	processes map[string]PregelNode
	channels  map[string]BaseChannel
	managed   ManagedValueMapping

	step            int
	stop            int
	interruptBefore []string
	interruptAfter  []string

	pendingWrites []WriteRecord
	tasks         map[string]*PregelExecutableTask
	toInterrupt   []*PregelExecutableTask

	triggerToNodes map[string][]string
	updatedChans   map[string]struct{}

	// synchronisation / workers
	workers int
	wg      sync.WaitGroup
	errMu   sync.Mutex
	runErr  error
	// Streaming
	streamCh   chan<- StreamChunk
	streamMode StreamMode

	pendingMu               sync.Mutex
	checkpointPendingWrites []PendingWrite

	checkpointer     Checkpointer // interface with PutWrites()
	checkpointConfig RunnableConfig

	emit func(task *PregelExecutableTask, writes []Write, cached bool)
}

type WriteRecord struct {
	Task  string
	Chan  string
	Value any
}

// NewLoop initialises a fully-featured loop.
func NewLoop(
	ctx context.Context,
	checkpoint Checkpoint,
	processes map[string]PregelNode,
	channels map[string]BaseChannel,
	managed ManagedValueMapping,
	cfg RunnableConfig,
	checkporter BaseCheckpointSaver,
	store BaseStore,
) *PregelLoop {
	c, cancel := context.WithCancel(ctx)
	// Ensure checkpoint is properly initialized
	if checkpoint.ChannelVersions == nil {
		checkpoint = NewCheckpoint()
	}
	loop := &PregelLoop{
		ctx:           c,
		cancel:        cancel,
		checkpoint:    checkpoint,
		processes:     processes,
		channels:      channels,
		managed:       managed,
		cfg:           cfg,
		checkporter:   checkporter,
		store:         store,
		step:          0,
		stop:          cfg.RecursionLimit,
		workers:       cfg.MaxConcurrency,
		pendingWrites: make([]WriteRecord, 0, 16),
		tasks:         map[string]*PregelExecutableTask{},
	}
	if loop.workers <= 0 {
		loop.workers = 1
	}
	return loop
}

// Run blocks until completion (or first error)
func (l *PregelLoop) Run() error {
	defer l.cancel()

	for {
		more, err := l.tick(nil)
		if err != nil {
			if errors.As(err, &GraphInterrupt{}) {
				return nil
			}
			return err
		}
		if !more {
			break
		}
	}
	return nil
}

// tick executes a single iteration of the Pregel loop.
// Returns true if more iterations are needed, false if done.
func (l *PregelLoop) tick(inputKeys []string) (bool, error) {
	// TODO: Use inputKeys to get the first values.
	// Check if we need to evaluate interrupts before execution
	if err := l.evaluateInterrupt("before"); err != nil {
		return false, err
	}

	// Build tasks
	tasks, err := PrepareNextTasks(
		l.ctx,
		l.checkpoint,
		convertPending(l.pendingWrites),
		l.processes,
		l.channels,
		l.managed,
		l.cfg,
		l.step,
		true,
		l.store,
		l.checkporter,
		l.triggerToNodes,
		l.updatedChans,
	)
	if err != nil {
		return false, err
	}
	if len(tasks) == 0 {
		return false, nil // done, no more tasks
	}
	l.tasks = make(map[string]*PregelExecutableTask)
	for k, v := range tasks {
		te := v.(PregelExecutableTask)
		l.tasks[k] = &te
	}

	// parallel execute
	workCh := make(chan *PregelExecutableTask)
	errCh := make(chan error, l.workers)

	for i := 0; i < l.workers; i++ {
		go l.worker(workCh, errCh)
	}

	for _, t := range l.tasks {
		if len(t.Writes) > 0 {
			continue // already satisfied
		}
		workCh <- t
	}
	close(workCh)

	for i := 0; i < l.workers; i++ {
		if err := <-errCh; err != nil {
			return false, err
		}
	}

	// All tasks finished; apply writes
	if err := l.applyWrites(); err != nil {
		return false, err
	}
	// checkpoint
	if err := l.saveCheckpoint(); err != nil {
		return false, err
	}

	// Check if we need to evaluate interrupts after execution
	if err := l.evaluateInterrupt("after"); err != nil {
		return false, err
	}

	// Check if we've exceeded the recursion limit
	l.step++
	if l.step > l.stop {
		return false, fmt.Errorf("exceeded recursion limit (%d)", l.stop)
	}

	return true, nil
}

// prepareAndExecuteStep is kept for backward compatibility
func (l *PregelLoop) prepareAndExecuteStep() error {
	more, err := l.tick(nil)
	if err != nil {
		return err
	}
	if !more {
		return nil
	}
	return nil
}

func (l *PregelLoop) worker(in <-chan *PregelExecutableTask, out chan<- error) {
	for task := range in {
		err := l.runTask(task)
		out <- err
	}
}

func (l *PregelLoop) runTask(t *PregelExecutableTask) error {
	// retry loop
	attempts := 0
	max := 1
	if p, ok := l.processes[t.Name]; ok {
		max = maxAttempts(p.Retry)
	}
	for {
		attempts++
		select {
		case <-l.ctx.Done():
			return l.ctx.Err()
		default:
		}

		writes, err := t.Node.Invoke(l.ctx, t.Input, t.Config, l)
		if err == nil {
			for _, w := range writes {
				l.recordWrite(t.ID, w.Channel, w.Value)
			}
			t.Writes = writes
			return nil
		}

		if attempts >= max {
			return err
		}
		time.Sleep(backoffDelay(attempts))
	}
}

// putWrites is called by PregelRunner (or nested tasks via the SEND helper)
// to persist writes produced by a task *during the current super-step*.
// It is safe for concurrent use.
func (l *PregelLoop) putWrites(taskID string, writes []Write) {
	if len(writes) == 0 {
		return
	}

	// ---------------------------------------------------------------------
	// 1.  Deduplicate if every write is for a “special” indexed channel.
	//     (“last one wins”, exactly like in TS / Python)
	// ---------------------------------------------------------------------
	allIndexed := true
	for _, w := range writes {
		if _, ok := WRITES_IDX_MAP[w.Channel]; !ok {
			allIndexed = false
			break
		}
	}
	if allIndexed {
		dedup := make(map[string]Write, len(writes))
		for _, w := range writes {
			dedup[w.Channel] = w
		}
		writes = make([]Write, 0, len(dedup))
		for _, w := range dedup {
			writes = append(writes, w)
		}
	}

	// ---------------------------------------------------------------------
	// 2.  Merge into l.checkpointPendingWrites.
	//     We need a mutex because PregelRunner goroutines call us in parallel.
	// ---------------------------------------------------------------------
	l.pendingMu.Lock()
	for _, w := range writes {
		replaced := false

		// If it is an indexed channel and an entry already exists for (task,channel),
		// overwrite it (=> keep only the newest write).
		if _, special := WRITES_IDX_MAP[w.Channel]; special {
			for i := range l.checkpointPendingWrites {
				pw := &l.checkpointPendingWrites[i]
				if pw.TaskID == taskID && pw.Channel == w.Channel {
					pw.Value = w.Value
					replaced = true
					break
				}
			}
		}

		// Otherwise (or if not found) just append.
		if !replaced {
			l.checkpointPendingWrites = append(
				l.checkpointPendingWrites,
				PendingWrite{TaskID: taskID, Channel: w.Channel, Value: w.Value},
			)
		}
	}
	l.pendingMu.Unlock()

	// ---------------------------------------------------------------------
	// 3.  Forward the writes to the configured checkpointer (if any).
	//     We don’t block the caller – a quick “fire-and-forget” goroutine
	//     is fine because checkpointer.PutWrites() is thread-safe by design.
	// ---------------------------------------------------------------------
	// if l.checkpointer != nil {
	// 	cfg := l.checkpointConfig // shallow copy is enough – we never mutate it
	// 	go l.checkpointer.PutWrites(cfg, writes, taskID)
	// }

	// ---------------------------------------------------------------------
	// 4.  Emit stream/debug output if the loop is already running.
	// ---------------------------------------------------------------------
	if len(l.tasks) > 0 {
		l.outputWrites(taskID, writes, false)
	}
}

// outputWrites mirrors TS _outputWrites (omits hidden tasks & handles modes).
// This is a *minimal* version; extend if you need streaming/debug UI parity.
func (l *PregelLoop) outputWrites(taskID string, writes []Write, cached bool) {
	task, ok := l.tasks[taskID]
	if !ok {
		return
	}
	for _, tag := range task.Config.Tags {
		if tag == TAG_HIDDEN {
			return
		}
	}
	// TODO: implement streaming
	// delegate to whatever streaming mechanism you implemented…
	// if l.emit != nil {
	// 	l.emit(task, writes, cached)
	// }
}

func maxAttempts(r RetryPolicy) int {
	if r.MaxAttempts <= 0 {
		return 1
	}
	return r.MaxAttempts
}

func backoffDelay(at int) time.Duration { return time.Duration(at) * 50 * time.Millisecond }

func (l *PregelLoop) Send(taskID string, writes []Write) {
	for _, w := range writes {
		l.recordWrite(taskID, w.Channel, w.Value)
	}
}

// Read returns a copy of current channel values
func (l *PregelLoop) Read(selectKeys []string) map[string]any {
	out := map[string]any{}
	for _, k := range selectKeys {
		if ch, ok := l.channels[k]; ok {
			out[k] = ch.Get()
		}
	}
	return out
}

func (l *PregelLoop) AcceptPush(origin PregelExecutableTask, writeIdx int, call *Call) (*PregelExecutableTask, error) {
	ppath := origin.Path
	newPath := []interface{}{PUSH, ppath, writeIdx, origin.ID, call}
	cpid, _ := hex.DecodeString(l.checkpoint.ID)
	nullVer := -1
	task, err := PrepareSingleTask(
		l.ctx,
		newPath,
		"",
		l.checkpoint,
		cpid,
		nullVer,
		convertPending(l.pendingWrites),
		l.processes,
		l.channels,
		l.managed,
		l.cfg,
		l.step,
		true,
		l.store,
		l.checkporter,
	)
	if err != nil {
		return nil, err
	}
	if task == nil {
		return nil, nil
	}
	te := task.(PregelExecutableTask)
	l.tasks[te.ID] = &te
	return &te, nil
}

func (l *PregelLoop) recordWrite(taskID, ch string, val any) {
	l.pendingWrites = append(l.pendingWrites, WriteRecord{taskID, ch, val})
}

func convertPending(ws []WriteRecord) []interface{} {
	out := make([]interface{}, 0, len(ws))
	for _, w := range ws {
		out = append(out, []interface{}{w.Task, w.Chan, w.Value})
	}
	return out
}

func (l *PregelLoop) applyWrites() error {
	if len(l.pendingWrites) == 0 {
		return nil
	}
	for _, wr := range l.pendingWrites {
		ch, ok := l.channels[wr.Chan]
		if !ok {
			ch = &simpleChan{}
			l.channels[wr.Chan] = ch
		}
		ch.Set(wr.Value)
		// TODO: Handle other version types.
		if _, exists := l.checkpoint.ChannelVersions[wr.Chan]; !exists {
			l.checkpoint.ChannelVersions[wr.Chan] = 0
		}
		l.checkpoint.ChannelVersions[wr.Chan]++
	}
	l.pendingWrites = l.pendingWrites[:0]
	return nil
}

func (l *PregelLoop) saveCheckpoint() error {
	if l.checkporter == nil {
		return nil
	}
	md := map[string]any{
		"step":   l.step,
		"source": "loop",
		"time":   time.Now().UTC().Format(time.RFC3339Nano),
	}
	return l.checkporter.Put(l.cfg, l.checkpoint, md, nil)
}

func (l *PregelLoop) evaluateInterrupt(stage string) error {
	var conditions []string
	if stage == "before" {
		conditions = l.interruptBefore
	} else {
		conditions = l.interruptAfter
	}
	if len(conditions) == 0 {
		return nil
	}
	seen := map[string]struct{}{}
	for _, t := range l.tasks {
		for _, trg := range t.Triggers {
			seen[trg] = struct{}{}
		}
	}
	for _, cond := range conditions {
		if _, ok := seen[cond]; ok || cond == "*" {
			return GraphInterrupt{}
		}
	}
	return nil
}

type Result struct {
	Err error
}
