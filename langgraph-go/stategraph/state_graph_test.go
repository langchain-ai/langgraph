package stategraph_test

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sg "github.com/langchain-ai/langgraph/langgraph-go/stategraph"
)

type stateGraphState struct {
	Noop bool `json:"noop"`
}

type orderRecorder struct {
	mu     sync.Mutex
	orders map[string]int32
	seq    int32
}

func newOrderRecorder() *orderRecorder {
	return &orderRecorder{orders: make(map[string]int32)}
}

func (f *orderRecorder) record(name string) {
	idx := atomic.AddInt32(&f.seq, 1)
	f.mu.Lock()
	f.orders[name] = idx
	f.mu.Unlock()
}

type orderGraph struct {
	recorder *orderRecorder
}

func newOrderGraph() *orderGraph {
	return &orderGraph{
		recorder: newOrderRecorder(),
	}
}

func (g *orderGraph) A(_ *sg.Context, state stateGraphState) (stateGraphState, error) {
	g.recorder.record("A")
	return state, nil
}

func (g *orderGraph) B1(_ *sg.Context, state stateGraphState) (stateGraphState, error) {
	g.recorder.record("B1")
	return state, nil
}

func (g *orderGraph) B2(_ *sg.Context, state stateGraphState) (stateGraphState, error) {
	g.recorder.record("B2")
	return state, nil
}

func (g *orderGraph) C1(_ *sg.Context, state stateGraphState) (stateGraphState, error) {
	g.recorder.record("C1")
	return state, nil
}

func (g *orderGraph) C2(_ *sg.Context, state stateGraphState) (stateGraphState, error) {
	g.recorder.record("C2")
	return state, nil
}

func (g *orderGraph) C3(_ *sg.Context, state stateGraphState) (stateGraphState, error) {
	g.recorder.record("C3")
	return state, nil
}

func (g *orderGraph) D(_ *sg.Context, state stateGraphState) (stateGraphState, error) {
	g.recorder.record("D")
	return state, nil
}

func TestBasicStateGraphWithoutInterrupt(t *testing.T) {
	fixture := newOrderGraph()
	graph := sg.NewBasicStateGraph[stateGraphState]()
	graph.AddNode(fixture.A)
	graph.AddNode(fixture.B1)
	graph.AddNode(fixture.B2)
	graph.AddNode(fixture.C1)
	graph.AddNode(fixture.C2)
	graph.AddNode(fixture.C3)
	graph.AddNode(fixture.D)

	graph.AddEdge(fixture.A, fixture.B1)
	graph.AddEdge(fixture.A, fixture.B2)
	graph.AddEdge(fixture.B1, fixture.C1)
	graph.AddEdge(fixture.B1, fixture.C2)
	graph.AddEdge(fixture.B2, fixture.C3)
	graph.AddEdge(fixture.C1, fixture.D)
	graph.AddEdge(fixture.C2, fixture.D)
	graph.AddEdge(fixture.C3, fixture.D)

	_, err := graph.Compile().Invoke(stateGraphState{})
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}

	fixture.recorder.mu.Lock()
	orders := make(map[string]int32, len(fixture.recorder.orders))
	for k, v := range fixture.recorder.orders {
		orders[k] = v
	}
	fixture.recorder.mu.Unlock()

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

type interruptState struct {
	A bool `json:"a"`
	B bool `json:"b"`
}

type interruptFixture struct{}

func (f *interruptFixture) A(_ *sg.Context, state interruptState) (interruptState, error) {
	state.A = true
	return state, nil
}

func (f *interruptFixture) B(ctx *sg.Context, state interruptState) (interruptState, error) {
	if !state.A {
		return state, fmt.Errorf("B should observe A=true")
	}
	value, err := ctx.Interrupt("resume_channel")
	if err != nil {
		return state, err
	}
	s, ok := value.(string)
	if !ok || s != "go" {
		return state, fmt.Errorf("unexpected interrupt payload: %#v", value)
	}
	state.B = true
	return state, nil
}

func TestBasicStateGraphWithInterrupt(t *testing.T) {
	fixture := &interruptFixture{}
	graph := sg.NewBasicStateGraph[interruptState]()
	graph.AddNode(fixture.A)
	graph.AddNode(fixture.B)
	graph.AddEdge(fixture.A, fixture.B)

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

	if err := handler.Interrupt("resume_channel", "go"); err != nil {
		t.Fatalf("resume interrupt failed: %v", err)
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
