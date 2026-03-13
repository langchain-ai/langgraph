package tests

import (
	"fmt"
	"testing"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

type primitiveWorkflow struct {
}

type primitiveState struct {
	Count int      `json:"count"`
	Logs  []string `json:"logs"`
	Done  string   `json:"done"`
}

func (w *primitiveWorkflow) startNode(ctx *ag.Context, input int, state primitiveState) (ag.Command, error) {
	state.Logs = append(state.Logs, fmt.Sprintf("start:%d", input))
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.middleNode, NodeInput: "from_start"},
		},
	}, nil
}

func (w *primitiveWorkflow) middleNode(ctx *ag.Context, input string, state primitiveState) (ag.Command, error) {
	state.Logs = append(state.Logs, "middle:"+input)
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.finishNode, NodeInput: "from_middle"},
		},
	}, nil
}

func (w *primitiveWorkflow) finishNode(ctx *ag.Context, input string, state primitiveState) (ag.Command, error) {
	state.Logs = append(state.Logs, "finish:"+input)
	state.Done = input
	return ag.Command{Update: state}, nil
}

func TestInputAndStatePrimitivesCompatible(t *testing.T) {
	workflow := &primitiveWorkflow{}
	graph := ag.NewAdvancedStateGraph[primitiveState]()

	graph.AddEntryNode(workflow.startNode)
	graph.AddNode(workflow.middleNode)
	graph.AddFinishNode(workflow.finishNode)

	handler, err := graph.Compile().Start(100, primitiveState{
		Count: 1,
		Logs:  []string{},
		Done:  "",
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}

	result, err := handler.WaitForResult()
	if err != nil {
		t.Fatalf("result failed: %v", err)
	}
	if result.Done != "from_middle" {
		t.Fatalf("unexpected done: %v", result.Done)
	}
	if result.Count != 1 {
		t.Fatalf("unexpected count: %v", result.Count)
	}
	if len(result.Logs) != 3 || result.Logs[0] != "start:100" || result.Logs[1] != "middle:from_start" || result.Logs[2] != "finish:from_middle" {
		t.Fatalf("unexpected logs: %#v", result.Logs)
	}
}
