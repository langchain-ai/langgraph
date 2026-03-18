package tests

import (
	"encoding/json"
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

func (w *primitiveWorkflow) startNoFinishNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	state.Logs = append(state.Logs, "start")
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.middleNoFinishNode, NodeInput: "from_start"},
		},
	}, nil
}

func (w *primitiveWorkflow) middleNoFinishNode(ctx *ag.Context, input string, state primitiveState) (ag.Command, error) {
	state.Logs = append(state.Logs, "middle:"+input)
	state.Count += 1
	state.Done = "stopped"
	// No goto and no finish node configured: run should end automatically.
	return ag.Command{Update: state}, nil
}

func TestRunEndsWithoutFinishNode(t *testing.T) {
	workflow := &primitiveWorkflow{}
	graph := ag.NewAdvancedStateGraph[primitiveState]()

	graph.AddEntryNode(workflow.startNoFinishNode)
	graph.AddNode(workflow.middleNoFinishNode)

	handler, err := graph.Compile().Start(nil, primitiveState{
		Count: 7,
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
	if result.Done != "stopped" {
		t.Fatalf("unexpected done: %v", result.Done)
	}
	if result.Count != 8 {
		t.Fatalf("unexpected count: %v", result.Count)
	}
	if len(result.Logs) != 2 || result.Logs[0] != "start" || result.Logs[1] != "middle:from_start" {
		t.Fatalf("unexpected logs: %#v", result.Logs)
	}
}

func (w *primitiveWorkflow) startWaitBatchNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	if err := ctx.PublishToChannel("events", "a"); err != nil {
		return ag.Command{}, err
	}
	if err := ctx.PublishToChannel("events", "b"); err != nil {
		return ag.Command{}, err
	}
	if err := ctx.PublishToChannel("events", "c"); err != nil {
		return ag.Command{}, err
	}
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.waitBatchNode, NodeInput: nil},
		},
	}, nil
}

func (w *primitiveWorkflow) waitBatchNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	event, err := ctx.WaitFor(ag.AnyOf(ag.ChannelCondition{
		Channel: "events",
		Min:     2,
		Max:     4,
	}))
	if err != nil {
		return ag.Command{}, err
	}
	var values []string
	if err := json.Unmarshal(event.Value, &values); err != nil {
		return ag.Command{}, err
	}
	state.Count = len(values)
	state.Logs = values
	state.Done = "ok"
	return ag.Command{Update: state}, nil
}

func TestChannelWaitRespectsMaxM(t *testing.T) {
	workflow := &primitiveWorkflow{}
	graph := ag.NewAdvancedStateGraph[primitiveState]()
	graph.AddAsyncChannel("events")
	graph.AddEntryNode(workflow.startWaitBatchNode)
	graph.AddFinishNode(workflow.waitBatchNode)

	handler, err := graph.Compile().Start(nil, primitiveState{
		Count: 0,
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
	if result.Count != 3 {
		t.Fatalf("unexpected count: %v", result.Count)
	}
	if len(result.Logs) != 3 || result.Logs[0] != "a" || result.Logs[1] != "b" || result.Logs[2] != "c" {
		t.Fatalf("unexpected logs: %#v", result.Logs)
	}
	if result.Done != "ok" {
		t.Fatalf("unexpected done: %v", result.Done)
	}
}
