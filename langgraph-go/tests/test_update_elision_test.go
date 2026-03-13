package tests

import (
	"testing"
	"time"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

type updateElisionState struct {
	X int `json:"x"`
}

type updateElisionWorkflow struct{}

func (w *updateElisionWorkflow) startNode(ctx *ag.Context, _ any, _ updateElisionState) (ag.Command, error) {
	return ag.Command{
		Goto: []ag.Send{
			{Node: w.fastNode},
			{Node: w.slowNode},
		},
	}, nil
}

func (w *updateElisionWorkflow) fastNode(ctx *ag.Context, _ any, state updateElisionState) (ag.Command, error) {
	state.X = 1
	return ag.Command{Update: state}, nil
}

func (w *updateElisionWorkflow) slowNode(ctx *ag.Context, _ any, state updateElisionState) (ag.Command, error) {
	time.Sleep(100 * time.Millisecond)
	// Returns same X as initial snapshot; SDK should elide this update.
	return ag.Command{Update: state}, nil
}

func TestNoopSlowUpdateDoesNotOverrideFastUpdate(t *testing.T) {
	workflow := &updateElisionWorkflow{}
	graph := ag.NewAdvancedStateGraph[updateElisionState]()
	graph.AddEntryNode(workflow.startNode)
	graph.AddNode(workflow.fastNode)
	graph.AddFinishNode(workflow.slowNode)

	handler, err := graph.Compile().Start(nil, updateElisionState{X: 0})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	result, err := handler.WaitForResult()
	if err != nil {
		t.Fatalf("result failed: %v", err)
	}
	if result.X != 1 {
		t.Fatalf("expected fast update to win, got x=%d", result.X)
	}
}
