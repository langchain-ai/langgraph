package tests

import (
	"strings"
	"testing"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

func TestNewAdvancedStateGraphRejectsNonStructState(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic for non-struct StateT")
		}
	}()
	_ = ag.NewAdvancedStateGraph[map[string]any]()
}

type stateTypeA struct {
	X int `json:"x"`
}

type stateTypeB struct {
	X int `json:"x"`
}

type wrongUpdateWorkflow struct{}

func (w *wrongUpdateWorkflow) startNode(ctx *ag.Context, _ any, _ stateTypeA) (ag.Command, error) {
	return ag.Command{Goto: []ag.Send{{Node: w.badNode}}}, nil
}

func (w *wrongUpdateWorkflow) badNode(ctx *ag.Context, _ any, _ stateTypeA) (ag.Command, error) {
	return ag.Command{Update: stateTypeB{X: 1}}, nil
}

func TestNodeUpdateTypeMustMatchGraphStateType(t *testing.T) {
	workflow := &wrongUpdateWorkflow{}
	graph := ag.NewAdvancedStateGraph[stateTypeA]()
	graph.AddEntryNode(workflow.startNode)
	graph.AddFinishNode(workflow.badNode)

	handler, err := graph.Compile().Start(nil, stateTypeA{X: 0})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	_, err = handler.WaitForResult()
	if err == nil {
		t.Fatalf("expected runtime error for wrong update type")
	}
	if !strings.Contains(err.Error(), "update type mismatch") {
		t.Fatalf("unexpected error: %v", err)
	}
}
