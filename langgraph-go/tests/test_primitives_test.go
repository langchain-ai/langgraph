package tests

import (
	"fmt"
	"testing"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

type primitiveWorkflow struct {
}

func logsSlice(state map[string]any) []string {
	raw, ok := state["logs"]
	if !ok || raw == nil {
		return []string{}
	}
	switch v := raw.(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return []string{}
	}
}

func (w *primitiveWorkflow) startNode(ctx *ag.Context, input int, state map[string]any) (ag.Command, error) {
	logs := logsSlice(state)
	logs = append(logs, fmt.Sprintf("start:%d", input))
	state["logs"] = logs
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.middleNode, NodeInput: "from_start"},
		},
	}, nil
}

func (w *primitiveWorkflow) middleNode(ctx *ag.Context, input string, state map[string]any) (ag.Command, error) {
	logs := logsSlice(state)
	logs = append(logs, "middle:"+input)
	state["logs"] = logs
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.finishNode, NodeInput: "from_middle"},
		},
	}, nil
}

func (w *primitiveWorkflow) finishNode(ctx *ag.Context, input string, state map[string]any) (ag.Command, error) {
	logs := logsSlice(state)
	logs = append(logs, "finish:"+input)
	state["logs"] = logs
	state["done"] = input
	return ag.Command{Update: state}, nil
}

func TestInputAndStatePrimitivesCompatible(t *testing.T) {
	workflow := &primitiveWorkflow{}
	graph := ag.NewAdvancedStateGraph()

	graph.AddEntryNode(workflow.startNode)
	graph.AddNode(workflow.middleNode)
	graph.AddFinishNode(workflow.finishNode)

	handler, err := graph.Compile().Start(map[string]any{
		"logs": []string{},
		"done": nil,
	}, 100)
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}

	result, err := handler.WaitForResult()
	if err != nil {
		t.Fatalf("result failed: %v", err)
	}
	if result["done"] != "from_middle" {
		t.Fatalf("unexpected done: %v", result["done"])
	}
	logs := logsSlice(result)
	if len(logs) != 3 || logs[0] != "start:100" || logs[1] != "middle:from_start" || logs[2] != "finish:from_middle" {
		t.Fatalf("unexpected logs: %#v", logs)
	}
}
