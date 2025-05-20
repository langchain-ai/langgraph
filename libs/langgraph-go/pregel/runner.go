// runner.go
package pregel

import (
	"context"
	"errors"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
)

// PregelRunner is responsible for executing the set of tasks that a
// PregelLoop prepared for the *current super-step*.  It runs them with
// respect to retry-policy, max-concurrency, timeouts, cancellation and
// Pregel-specific error semantics (GraphInterrupt / GraphBubbleUp).
type PregelRunner struct {
	loop         *PregelLoop
	nodeFinished func(string) // Optional user-callback
}

// NewPregelRunner links the runner to its parent loop.
func NewPregelRunner(loop *PregelLoop, nodeFinished func(string)) *PregelRunner {
	return &PregelRunner{loop: loop, nodeFinished: nodeFinished}
}

// TickOptions mirrors the semantics in the TS/Python implementations.
type TickOptions struct {
	Timeout        time.Duration      // Deadline for the whole super-step
	RetryPolicy    RetryPolicy        // Per-task retry policy
	OnStepWrite    func(int, []Write) // Hook after *all* writes are committed
	MaxConcurrency int                // ≤0  ⇒ unlimited
	Ctx            context.Context    // Root ctx (optional)
}

// Tick executes every task whose Writes slice is still empty.
// It returns when *all* tasks have completed (successfully or not) **or**
// when the first non-interrupt error bubbles up.
func (r *PregelRunner) tick(opt TickOptions) error {
	// Choose base context
	ctx := opt.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	// We cancel siblings on first fatal error
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Optional global timeout
	if opt.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, opt.Timeout)
		defer cancel()
	}

	// Gather tasks that still need to run in this super-step
	var pending []*PregelExecutableTask
	for _, t := range r.loop.tasks {
		if len(t.Writes) == 0 {
			pending = append(pending, t)
		}
	}
	if len(pending) == 0 {
		return nil // nothing to do
	}

	// errgroup manages goroutines and collects the first returned error
	g, gctx := errgroup.WithContext(ctx)

	maxConc := opt.MaxConcurrency
	if maxConc <= 0 {
		maxConc = len(pending)
	}
	sem := make(chan struct{}, maxConc)

	var mu sync.Mutex

	for _, task := range pending {
		task := task // capture
		sem <- struct{}{}
		g.Go(func() error {
			defer func() { <-sem }()

			err := runWithRetry(gctx, opt.RetryPolicy, func(c context.Context) error {
				// NOTE: Node.Run must honour ctx for cancellation / deadlines.
				writes, runErr := task.Node.Invoke(c, task.Input, task.Config, r.loop)
				if runErr == nil {
					task.Writes = writes
				}
				return runErr
			})

			r.commit(task, err)

			switch {
			case err == nil:
				return nil
			case errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded):
				return err // propagate
			}

			var gi GraphInterrupt
			if errors.As(err, &gi) {
				mu.Lock()
				defer mu.Unlock()
				// kep track so that loop can raise combined interrupt later
				return gi
			}

			cancel()
			return err
		})
	}

	// Wait for all goroutines (or first fatal error)
	if err := g.Wait(); err != nil {
		return err
	}

	// Step-level callback after *all* commits
	if opt.OnStepWrite != nil {
		var all []Write
		for _, t := range r.loop.tasks {
			all = append(all, t.Writes...)
		}
		opt.OnStepWrite(r.loop.step, all)
	}

	return nil
}

// commit replicates the Python/TS commit semantics.
func (r *PregelRunner) commit(task *PregelExecutableTask, execErr error) {
	// On success ensure at least one NO_WRITES marker so loop knows it's done.
	if execErr == nil && len(task.Writes) == 0 {
		task.Writes = append(task.Writes, Write{Channel: NO_WRITES})
	}

	// Persist writes (or error) through the loop’s thread-safe adaptor.
	switch {
	case execErr == nil:
		r.loop.putWrites(task.ID, task.Writes)
	case errors.As(execErr, new(GraphInterrupt)):
		// Interrupt carries its own writes payload
		r.loop.putWrites(task.ID, task.Writes)
	default:
		// Record generic error
		r.loop.putWrites(task.ID, []Write{{Channel: ERROR, Value: execErr}})
	}

	// optional callback
	if execErr == nil && r.nodeFinished != nil {
		r.nodeFinished(task.Name)
	}
}

// runWithRetry is a minimal exponential-back-off retry helper.
func runWithRetry(ctx context.Context, pol RetryPolicy, fn func(context.Context) error) error {
	if pol.MaxAttempts <= 0 {
		pol.MaxAttempts = 1
	}
	// if pol.Backoff == nil {
	// 	// default: exponential capped at 2 s
	// 	pol.Backoff = func(attempt int) time.Duration {
	// 		d := time.Duration(math.Pow(2, float64(attempt))) * 50 * time.Millisecond
	// 		if d > 2*time.Second {
	// 			d = 2 * time.Second
	// 		}
	// 		return d
	// 	}
	// }
	// if pol.Retryable == nil {
	// 	pol.Retryable = func(error) bool { return true }
	// }

	var err error
	for attempt := 0; attempt < pol.MaxAttempts; attempt++ {
		if err = fn(ctx); err == nil { // || !pol.Retryable(err) {
			return err
		}
		// // wait before next try
		// wait := pol.Backoff(attempt)
		// select {
		// case <-time.After(wait):
		// case <-ctx.Done():
		// 	return ctx.Err()
		// }
	}
	return err
}

/* --------------------------------------------------------------------------
   Missing symbols?  If your project does not yet declare the following items
   just add minimal stubs like the ones below (remove before wiring in
   real implementations to avoid duplicates).

// Constants that mark write types
const (
	ERROR     = "error"
	NO_WRITES = "no_writes"
)

// GraphInterrupt / BubbleUp marker errors
type GraphInterrupt struct{ Msg string }
func (g GraphInterrupt) Error() string { return g.Msg }
type GraphBubbleUp struct{ error }

// Minimal Write + RetryPolicy
type Write struct{ Channel string; Value any }

type RetryPolicy struct {
	MaxAttempts int
	Backoff     func(attempt int) time.Duration
	Retryable   func(error) bool
}

// PregelExecutableTask, PregelLoop, etc. should exist elsewhere.
// -------------------------------------------------------------------------- */
