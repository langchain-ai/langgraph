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
	result, err := ctx.WaitFor(ag.AnyOf(ag.ChannelCondition{
		Channel: "events",
		Min:     2,
		Max:     4,
	}))
	if err != nil {
		return ag.Command{}, err
	}
	if len(result.Conditions) != 1 || !result.Conditions[0].Met {
		return ag.Command{}, fmt.Errorf("expected one met condition")
	}
	values := make([]string, 0, len(result.Conditions[0].Values))
	for _, v := range result.Conditions[0].Values {
		s, _ := v.(string)
		values = append(values, s)
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

func (w *primitiveWorkflow) startAnyOfTwoChannelsNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	if err := ctx.PublishToChannel("alpha", "a1"); err != nil {
		return ag.Command{}, err
	}
	if err := ctx.PublishToChannel("beta", "b1"); err != nil {
		return ag.Command{}, err
	}
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.waitAnyOfTwoChannelsNode, NodeInput: nil},
		},
	}, nil
}

func (w *primitiveWorkflow) waitAnyOfTwoChannelsNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	first, err := ctx.WaitFor(ag.AnyOf(
		ag.ChannelCondition{Channel: "alpha"},
		ag.ChannelCondition{Channel: "beta"},
	))
	if err != nil {
		return ag.Command{}, err
	}
	if len(first.Conditions) != 2 {
		return ag.Command{}, fmt.Errorf("expected 2 condition results, got %d", len(first.Conditions))
	}
	if !first.Conditions[0].Met || first.Conditions[0].ChannelName != "alpha" || len(first.Conditions[0].Values) != 1 || first.Conditions[0].Values[0] != "a1" {
		return ag.Command{}, fmt.Errorf("unexpected first condition result: %#v", first.Conditions[0])
	}
	if !first.Conditions[1].Met || first.Conditions[1].ChannelName != "beta" || len(first.Conditions[1].Values) != 1 || first.Conditions[1].Values[0] != "b1" {
		return ag.Command{}, fmt.Errorf("unexpected second condition result: %#v", first.Conditions[1])
	}
	if err := ctx.PublishToChannel("beta", "b2"); err != nil {
		return ag.Command{}, err
	}
	state.Count = 1
	state.Logs = []string{
		"matched=2",
	}
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.verifyBetaAfterAnyOfNode, NodeInput: nil},
		},
	}, nil
}

func (w *primitiveWorkflow) verifyBetaAfterAnyOfNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	second, err := ctx.WaitFor(ag.AnyOf(
		ag.ChannelCondition{Channel: "beta"},
	))
	if err != nil {
		return ag.Command{}, err
	}
	if len(second.Conditions) != 1 || !second.Conditions[0].Met || second.Conditions[0].ChannelName != "beta" || len(second.Conditions[0].Values) != 1 {
		return ag.Command{}, fmt.Errorf("unexpected beta condition result: %#v", second.Conditions)
	}
	payload, _ := second.Conditions[0].Values[0].(string)
	state.Count = 2
	state.Logs = append(state.Logs, fmt.Sprintf("beta=%s", payload))
	state.Done = "ok"
	return ag.Command{Update: state}, nil
}

func TestAnyOfConsumesAllReadyChannels(t *testing.T) {
	workflow := &primitiveWorkflow{}
	graph := ag.NewAdvancedStateGraph[primitiveState]()
	graph.AddAsyncChannel("alpha")
	graph.AddAsyncChannel("beta")
	graph.AddEntryNode(workflow.startAnyOfTwoChannelsNode)
	graph.AddNode(workflow.waitAnyOfTwoChannelsNode)
	graph.AddFinishNode(workflow.verifyBetaAfterAnyOfNode)

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
	if result.Count != 2 {
		t.Fatalf("unexpected count: %v", result.Count)
	}
	if len(result.Logs) != 2 || result.Logs[0] != "matched=2" || result.Logs[1] != "beta=b2" {
		t.Fatalf("unexpected logs: %#v", result.Logs)
	}
	if result.Done != "ok" {
		t.Fatalf("unexpected done: %v", result.Done)
	}
}
