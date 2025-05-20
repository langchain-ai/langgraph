package pregel

import (
	"context"
	"errors"
)

// Pregel is the top-level graph object.
type Pregel struct {
	Name     string
	Nodes    map[string]PregelNode
	Channels map[string]BaseChannel
	LoopCfg  RunnableConfig
	Checkptr BaseCheckpointSaver
	Store    BaseStore
	Debug    bool
}

func (g *Pregel) Stream(
	input any,
	cfg RunnableConfig,
	opts *StreamOptions,
) (<-chan StreamChunk, <-chan error) {
	eventCh := make(chan StreamChunk, 16)
	errCh := make(chan error, 1)
	if opts == nil {
		opts = &StreamOptions{}
	}
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	mode := opts.Mode
	if mode == "" {
		mode = StreamValues
	}
	// TODO: opts.Debug

	// ensure output channels are set / valid
	outChans := opts.OutputChannels
	if len(outChans) == 0 {
		for k := range g.Channels {
			if _, ok := g.Channels[k]; ok {
				outChans = append(outChans, k)
			}
		}
	}

	if opts.MaxConcurrency > 0 {
		cfg.MaxConcurrency = opts.MaxConcurrency
	}
	if cfg.MaxConcurrency == 0 {
		cfg.MaxConcurrency = 4
	}
	if cfg.RecursionLimit == 0 {
		cfg.RecursionLimit = 25
	}
	if opts.CheckpointDuring != nil {
		cfg.Configurable[CONFIG_KEY_CHECKPOINT_DURING] = *opts.CheckpointDuring
	}
	checkpoint, err := EmptyCheckpoint()
	if err != nil {
		errCh <- err
		return nil, errCh
	}
	loop := NewLoop(
		ctx,
		*checkpoint,
		g.Nodes,
		g.channelsAsConcrete(),
		nil, // managed values
		cfg,
		nil, // g.checkpointer, // may be nil
		nil, // g.store,
	)
	loop.interruptBefore = opts.InterruptBefore
	loop.interruptAfter = opts.InterruptAfter
	loop.streamCh = eventCh
	loop.streamMode = mode
	// loop.debug = debug

	go func() {
		defer close(eventCh)
		defer close(errCh)

		// Create a runner to execute tasks
		runner := NewPregelRunner(loop, nil)

		// Use the tick method in a loop instead of Run()
		for {
			more, err := loop.tick(outChans)
			if err != nil {
				// Check if this is a GraphInterrupt error
				var interrupt GraphInterrupt
				if errors.As(err, &interrupt) {
					// Handle interrupt gracefully
					break
				}
				// Otherwise, it's a real error
				errCh <- err
				return
			}

			runnerOpts := TickOptions{
				MaxConcurrency: cfg.MaxConcurrency,
			}

			if opts.Debug != nil && *opts.Debug {
				runnerOpts.OnStepWrite = func(step int, writes []Write) {
					// TODO: Handle debugging info
				}
			}

			if err := runner.tick(runnerOpts); err != nil {
				errCh <- err
				return
			}

			// No more iterations needed, we're done
			if !more {
				break
			}
		}

	}()

	return eventCh, errCh
}

func (g *Pregel) channelsAsConcrete() map[string]BaseChannel {
	out := make(map[string]BaseChannel, len(g.Channels))
	for k, v := range g.Channels {
		if ch, ok := v.(BaseChannel); ok {
			out[k] = ch
		}
	}
	return out
}
