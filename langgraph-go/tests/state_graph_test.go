package tests

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

type stateGraphState struct {
	Noop bool `json:"noop"`
}

func TestBasicStateGraphWithoutInterrupt(t *testing.T) {
	var (
		orderMu sync.Mutex
		orders  = make(map[string]int32)
		seq     int32
	)
	record := func(name string) {
		idx := atomic.AddInt32(&seq, 1)
		orderMu.Lock()
		orders[name] = idx
		orderMu.Unlock()
	}
	graph := ag.NewBasicStateGraph[stateGraphState]()
	graph.AddNode("A", func(_ *ag.Context, state stateGraphState) (stateGraphState, error) {
		record("A")
		return state, nil
	})
	graph.AddNode("B1", func(_ *ag.Context, state stateGraphState) (stateGraphState, error) {
		record("B1")
		return state, nil
	})
	graph.AddNode("B2", func(_ *ag.Context, state stateGraphState) (stateGraphState, error) {
		record("B2")
		return state, nil
	})
	graph.AddNode("C1", func(_ *ag.Context, state stateGraphState) (stateGraphState, error) {
		record("C1")
		return state, nil
	})
	graph.AddNode("C2", func(_ *ag.Context, state stateGraphState) (stateGraphState, error) {
		record("C2")
		return state, nil
	})
	graph.AddNode("C3", func(_ *ag.Context, state stateGraphState) (stateGraphState, error) {
		record("C3")
		return state, nil
	})
	graph.AddNode("D", func(_ *ag.Context, state stateGraphState) (stateGraphState, error) {
		record("D")
		return state, nil
	})

	graph.AddEdge("A", "B1")
	graph.AddEdge("A", "B2")
	graph.AddEdge("B1", "C1")
	graph.AddEdge("B1", "C2")
	graph.AddEdge("B2", "C3")
	graph.AddEdge("C1", "D")
	graph.AddEdge("C2", "D")
	graph.AddEdge("C3", "D")

	_, err := graph.Compile().Invoke(stateGraphState{})
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}
	for _, name := range []string{"A", "B1", "B2", "C1", "C2", "C3", "D"} {
		if _, ok := orders[name]; !ok {
			t.Fatalf("node %s did not execute; orders=%v", name, orders)
		}
	}
	maxB := maxInt32(orders["B1"], orders["B2"])
	minC := minInt32(orders["C1"], minInt32(orders["C2"], orders["C3"]))
	maxC := maxInt32(orders["C1"], maxInt32(orders["C2"], orders["C3"]))
	if !(orders["A"] < orders["B1"] && orders["A"] < orders["B2"]) {
		t.Fatalf("A should run before B-step, orders=%v", orders)
	}
	if !(maxB < minC) {
		t.Fatalf("B-step should finish before C-step, orders=%v", orders)
	}
	if !(maxC < orders["D"]) {
		t.Fatalf("C-step should finish before D, orders=%v", orders)
	}
}

func TestBasicStateGraphWithInterrupt(t *testing.T) {
	type interruptState struct {
		A bool `json:"a"`
		B bool `json:"b"`
	}
	graph := ag.NewBasicStateGraph[interruptState]()
	graph.AddNode("A", func(_ *ag.Context, state interruptState) (interruptState, error) {
		state.A = true
		return state, nil
	})
	graph.AddNode("B", func(_ *ag.Context, state interruptState) (interruptState, error) {
		if !state.A {
			return state, fmt.Errorf("B should observe A=true")
		}
		state.B = true
		return state, nil
	})
	graph.AddEdge("A", "B")
	graph.EnableInterruptOnSuperstep(1, "resume_channel")

	handler, err := graph.Compile().Start(interruptState{})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}

	doneCh := make(chan interruptState, 1)
	errCh := make(chan error, 1)
	go func() {
		result, runErr := handler.WaitForResult()
		if runErr != nil {
			errCh <- runErr
			return
		}
		doneCh <- result
	}()

	select {
	case <-doneCh:
		t.Fatalf("run should pause for interrupt, but completed early")
	case err := <-errCh:
		t.Fatalf("run should pause for interrupt, but failed early: %v", err)
	case <-time.After(120 * time.Millisecond):
		// expected: paused
	}

	if err := handler.PublishToChannel("resume_channel", "go"); err != nil {
		t.Fatalf("publish interrupt failed: %v", err)
	}

	select {
	case err := <-errCh:
		t.Fatalf("run failed after interrupt: %v", err)
	case result := <-doneCh:
		if !(result.A && result.B) {
			t.Fatalf("unexpected final state: %#v", result)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timeout waiting for resumed run completion")
	}
}

func minInt32(a int32, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

func maxInt32(a int32, b int32) int32 {
	if a > b {
		return a
	}
	return b
}
