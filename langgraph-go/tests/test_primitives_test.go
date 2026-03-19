package tests

import (
	"fmt"
	"sync"
	"testing"
	"time"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

type primitiveWorkflow struct {
	dbWriteCount int
	intervalMu   sync.Mutex
	intervals    map[string][2]time.Time
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

func (w *primitiveWorkflow) startAllOfTwoChannelsNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	if err := ctx.PublishToChannel("alpha", "a1"); err != nil {
		return ag.Command{}, err
	}
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.waitAllOfTwoChannelsNode, NodeInput: nil},
		},
	}, nil
}

func (w *primitiveWorkflow) waitAllOfTwoChannelsNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	result, err := ctx.WaitFor(ag.AllOf(
		ag.ChannelCondition{Channel: "alpha"},
		ag.ChannelCondition{Channel: "beta"},
	))
	if err != nil {
		return ag.Command{}, err
	}
	if len(result.Conditions) != 2 {
		return ag.Command{}, fmt.Errorf("expected 2 condition results, got %d", len(result.Conditions))
	}
	if !result.Conditions[0].Met || result.Conditions[0].ChannelName != "alpha" || len(result.Conditions[0].Values) != 1 || result.Conditions[0].Values[0] != "a1" {
		return ag.Command{}, fmt.Errorf("unexpected alpha condition result: %#v", result.Conditions[0])
	}
	if !result.Conditions[1].Met || result.Conditions[1].ChannelName != "beta" || len(result.Conditions[1].Values) != 1 || result.Conditions[1].Values[0] != "b1" {
		return ag.Command{}, fmt.Errorf("unexpected beta condition result: %#v", result.Conditions[1])
	}
	state.Count = 2
	state.Logs = []string{"all_of_channels_ok"}
	state.Done = "ok"
	return ag.Command{Update: state}, nil
}

func TestAllOfWaitsUntilAllChannelsAreReady(t *testing.T) {
	workflow := &primitiveWorkflow{}
	graph := ag.NewAdvancedStateGraph[primitiveState]()
	graph.AddAsyncChannel("alpha")
	graph.AddAsyncChannel("beta")
	graph.AddEntryNode(workflow.startAllOfTwoChannelsNode)
	graph.AddFinishNode(workflow.waitAllOfTwoChannelsNode)

	handler, err := graph.Compile().Start(nil, primitiveState{
		Count: 0,
		Logs:  []string{},
		Done:  "",
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	if err := handler.PublishToChannel("beta", "b1"); err != nil {
		t.Fatalf("publish beta failed: %v", err)
	}

	result, err := handler.WaitForResult()
	if err != nil {
		t.Fatalf("result failed: %v", err)
	}
	if result.Count != 2 {
		t.Fatalf("unexpected count: %v", result.Count)
	}
	if len(result.Logs) != 1 || result.Logs[0] != "all_of_channels_ok" {
		t.Fatalf("unexpected logs: %#v", result.Logs)
	}
	if result.Done != "ok" {
		t.Fatalf("unexpected done: %v", result.Done)
	}
}

func (w *primitiveWorkflow) startAllOfChannelTimerNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	if err := ctx.PublishToChannel("alpha", "a1"); err != nil {
		return ag.Command{}, err
	}
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.waitAllOfChannelTimerNode, NodeInput: nil},
		},
	}, nil
}

func (w *primitiveWorkflow) waitAllOfChannelTimerNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	result, err := ctx.WaitFor(ag.AllOf(
		ag.ChannelCondition{Channel: "alpha"},
		ag.TimerCondition{Seconds: 0.05},
	))
	if err != nil {
		return ag.Command{}, err
	}
	if len(result.Conditions) != 2 {
		return ag.Command{}, fmt.Errorf("expected 2 condition results, got %d", len(result.Conditions))
	}
	if !result.Conditions[0].Met || result.Conditions[0].ChannelName != "alpha" || len(result.Conditions[0].Values) != 1 || result.Conditions[0].Values[0] != "a1" {
		return ag.Command{}, fmt.Errorf("unexpected channel condition result: %#v", result.Conditions[0])
	}
	if !result.Conditions[1].Met {
		return ag.Command{}, fmt.Errorf("timer condition should be met: %#v", result.Conditions[1])
	}
	state.Count = 1
	state.Logs = []string{"all_of_channel_timer_ok"}
	state.Done = "ok"
	return ag.Command{Update: state}, nil
}

func TestAllOfChannelAndTimerMarksBothConditions(t *testing.T) {
	workflow := &primitiveWorkflow{}
	graph := ag.NewAdvancedStateGraph[primitiveState]()
	graph.AddAsyncChannel("alpha")
	graph.AddEntryNode(workflow.startAllOfChannelTimerNode)
	graph.AddFinishNode(workflow.waitAllOfChannelTimerNode)

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
	if result.Count != 1 {
		t.Fatalf("unexpected count: %v", result.Count)
	}
	if len(result.Logs) != 1 || result.Logs[0] != "all_of_channel_timer_ok" {
		t.Fatalf("unexpected logs: %#v", result.Logs)
	}
	if result.Done != "ok" {
		t.Fatalf("unexpected done: %v", result.Done)
	}
}

func (w *primitiveWorkflow) startResumeFlagNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	state.Logs = []string{}
	state.Count = 0
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.resumeFlagNode, NodeInput: nil},
		},
	}, nil
}

func (w *primitiveWorkflow) resumeFlagNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	if !ctx.IsResume() {
		// Simulate one-time side effect (e.g. DB write).
		w.dbWriteCount += 1
	}
	if _, err := ctx.WaitFor(ag.AnyOf(ag.TimerCondition{Seconds: 0.02})); err != nil {
		return ag.Command{}, err
	}
	state.Logs = append(state.Logs, fmt.Sprintf("resume=%v", ctx.IsResume()))
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.finishResumeFlagNode, NodeInput: nil},
		},
	}, nil
}

func (w *primitiveWorkflow) finishResumeFlagNode(_ *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	state.Done = "ok"
	return ag.Command{Update: state}, nil
}

func TestIsResumeAvoidsDuplicateSideEffects(t *testing.T) {
	workflow := &primitiveWorkflow{}
	graph := ag.NewAdvancedStateGraph[primitiveState]()
	graph.AddEntryNode(workflow.startResumeFlagNode)
	graph.AddNode(workflow.resumeFlagNode)
	graph.AddFinishNode(workflow.finishResumeFlagNode)

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
	if workflow.dbWriteCount != 1 {
		t.Fatalf("db write should happen once, got=%v", workflow.dbWriteCount)
	}
	if len(result.Logs) != 1 || result.Logs[0] != "resume=true" {
		t.Fatalf("unexpected logs: %#v", result.Logs)
	}
	if result.Done != "ok" {
		t.Fatalf("unexpected done: %v", result.Done)
	}
}

func (w *primitiveWorkflow) startLockedWorkersNode(_ *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	w.intervalMu.Lock()
	w.intervals = map[string][2]time.Time{}
	w.intervalMu.Unlock()
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.lockedWorkerANode, NodeInput: nil},
			{Node: w.lockedWorkerBNode, NodeInput: nil},
			{Node: w.waitLockedWorkersNode, NodeInput: nil},
		},
	}, nil
}

func (w *primitiveWorkflow) lockedWorkerANode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	start := time.Now()
	time.Sleep(40 * time.Millisecond)
	end := time.Now()
	w.intervalMu.Lock()
	w.intervals["a"] = [2]time.Time{start, end}
	w.intervalMu.Unlock()
	if err := ctx.PublishToChannel("done", "a"); err != nil {
		return ag.Command{}, err
	}
	return ag.Command{Update: state}, nil
}

func (w *primitiveWorkflow) lockedWorkerBNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	start := time.Now()
	time.Sleep(40 * time.Millisecond)
	end := time.Now()
	w.intervalMu.Lock()
	w.intervals["b"] = [2]time.Time{start, end}
	w.intervalMu.Unlock()
	if err := ctx.PublishToChannel("done", "b"); err != nil {
		return ag.Command{}, err
	}
	return ag.Command{Update: state}, nil
}

func (w *primitiveWorkflow) waitLockedWorkersNode(ctx *ag.Context, _ any, state primitiveState) (ag.Command, error) {
	if _, err := ctx.WaitFor(ag.AnyOf(ag.ChannelCondition{Channel: "done", Min: 2})); err != nil {
		return ag.Command{}, err
	}
	return ag.Command{
		Update: state,
		Goto:   []ag.Send{{Node: w.finishResumeFlagNode, NodeInput: nil}},
	}, nil
}

func TestStateFieldLockingSerializesConflictingNodes(t *testing.T) {
	workflow := &primitiveWorkflow{}
	graph := ag.NewAdvancedStateGraph[primitiveState]()
	graph.AddAsyncChannel("done")
	graph.AddEntryNode(workflow.startLockedWorkersNode)
	graph.AddNode(
		workflow.lockedWorkerANode,
		ag.NodeStateOption{LockedFields: []string{"counter"}},
	)
	graph.AddNode(
		workflow.lockedWorkerBNode,
		ag.NodeStateOption{LockedFields: []string{"counter"}},
	)
	graph.AddNode(workflow.waitLockedWorkersNode)
	graph.AddFinishNode(workflow.finishResumeFlagNode)

	handler, err := graph.Compile().Start(nil, primitiveState{
		Count: 0,
		Logs:  []string{},
		Done:  "",
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	if _, err := handler.WaitForResult(); err != nil {
		t.Fatalf("result failed: %v", err)
	}

	workflow.intervalMu.Lock()
	ia, okA := workflow.intervals["a"]
	ib, okB := workflow.intervals["b"]
	workflow.intervalMu.Unlock()
	if !okA || !okB {
		t.Fatalf("missing worker intervals: %#v", workflow.intervals)
	}
	serialized := !ia[1].After(ib[0]) || !ib[1].After(ia[0])
	if !serialized {
		t.Fatalf("expected serialized execution, got overlap: a=%v..%v b=%v..%v", ia[0], ia[1], ib[0], ib[1])
	}
}
