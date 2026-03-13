package tests

import (
	"reflect"
	"testing"
	"time"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

type updateElisionState struct {
	X  int             `json:"x"`
	S  updateStruct    `json:"s"`
	M  map[string]int  `json:"m"`
	L  []int           `json:"l"`
	PS *updateStruct   `json:"ps"`
	PM *map[string]int `json:"pm"`
	PL *[]int          `json:"pl"`
}

type updateStruct struct {
	V int `json:"v"`
}

type updateElisionWorkflow struct{}

func makeState(v int) updateElisionState {
	m := map[string]int{"n": v}
	l := []int{v}
	return updateElisionState{
		X:  v,
		S:  updateStruct{V: v},
		M:  map[string]int{"n": v},
		L:  []int{v},
		PS: &updateStruct{V: v},
		PM: &m,
		PL: &l,
	}
}

func (w *updateElisionWorkflow) startNoopNode(ctx *ag.Context, _ any, _ updateElisionState) (ag.Command, error) {
	return ag.Command{
		Goto: []ag.Send{
			{Node: w.fastNode},
			{Node: w.slowNoopNode},
		},
	}, nil
}

func (w *updateElisionWorkflow) startChangedNode(ctx *ag.Context, _ any, _ updateElisionState) (ag.Command, error) {
	return ag.Command{
		Goto: []ag.Send{
			{Node: w.fastNode},
			{Node: w.slowChangedNode},
		},
	}, nil
}

func (w *updateElisionWorkflow) fastNode(ctx *ag.Context, _ any, state updateElisionState) (ag.Command, error) {
	_ = state
	return ag.Command{Update: makeState(1)}, nil
}

func (w *updateElisionWorkflow) slowNoopNode(ctx *ag.Context, _ any, state updateElisionState) (ag.Command, error) {
	time.Sleep(100 * time.Millisecond)
	// Returns same state as initial snapshot; SDK should elide this update.
	return ag.Command{Update: state}, nil
}

func (w *updateElisionWorkflow) slowChangedNode(ctx *ag.Context, _ any, _ updateElisionState) (ag.Command, error) {
	time.Sleep(100 * time.Millisecond)
	// Real change should not be elided.
	return ag.Command{Update: makeState(2)}, nil
}

func assertStateEquals(t *testing.T, got updateElisionState, expected updateElisionState) {
	t.Helper()
	if got.X != expected.X {
		t.Fatalf("unexpected X: got=%d want=%d", got.X, expected.X)
	}
	if got.S != expected.S {
		t.Fatalf("unexpected S: got=%#v want=%#v", got.S, expected.S)
	}
	if !reflect.DeepEqual(got.M, expected.M) {
		t.Fatalf("unexpected M: got=%#v want=%#v", got.M, expected.M)
	}
	if !reflect.DeepEqual(got.L, expected.L) {
		t.Fatalf("unexpected L: got=%#v want=%#v", got.L, expected.L)
	}
	if got.PS == nil || expected.PS == nil || *got.PS != *expected.PS {
		t.Fatalf("unexpected PS: got=%#v want=%#v", got.PS, expected.PS)
	}
	if got.PM == nil || expected.PM == nil || !reflect.DeepEqual(*got.PM, *expected.PM) {
		t.Fatalf("unexpected PM: got=%#v want=%#v", got.PM, expected.PM)
	}
	if got.PL == nil || expected.PL == nil || !reflect.DeepEqual(*got.PL, *expected.PL) {
		t.Fatalf("unexpected PL: got=%#v want=%#v", got.PL, expected.PL)
	}
}

func TestNoopSlowUpdateDoesNotOverrideFastUpdate(t *testing.T) {
	workflow := &updateElisionWorkflow{}
	graph := ag.NewAdvancedStateGraph[updateElisionState]()
	graph.AddEntryNode(workflow.startNoopNode)
	graph.AddNode(workflow.fastNode)
	graph.AddFinishNode(workflow.slowNoopNode)

	handler, err := graph.Compile().Start(nil, makeState(0))
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	result, err := handler.WaitForResult()
	if err != nil {
		t.Fatalf("result failed: %v", err)
	}
	assertStateEquals(t, result, makeState(1))
}

func TestChangedSlowUpdateOverridesFastUpdate(t *testing.T) {
	workflow := &updateElisionWorkflow{}
	graph := ag.NewAdvancedStateGraph[updateElisionState]()
	graph.AddEntryNode(workflow.startChangedNode)
	graph.AddNode(workflow.fastNode)
	graph.AddFinishNode(workflow.slowChangedNode)

	handler, err := graph.Compile().Start(nil, makeState(0))
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	result, err := handler.WaitForResult()
	if err != nil {
		t.Fatalf("result failed: %v", err)
	}
	assertStateEquals(t, result, makeState(2))
}
