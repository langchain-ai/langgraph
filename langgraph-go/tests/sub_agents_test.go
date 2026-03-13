package tests

import (
	"slices"
	"testing"
	"time"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

type decision struct {
	Type     string
	SubAgent string
	Tool     string
	Complete string
}

type mockLLM struct {
	responses [][]decision
	i         int
}

func (m *mockLLM) invoke() []decision {
	if m.i >= len(m.responses) {
		return []decision{}
	}
	resp := m.responses[m.i]
	m.i++
	return resp
}

type lunchWorkflow struct {
	planner *mockLLM
	names   map[string]string
}

func cloneState(state map[string]any) map[string]any {
	out := make(map[string]any, len(state)+2)
	for k, v := range state {
		out[k] = v
	}
	out["output"] = append([]string(nil), outputSlice(state)...)
	return out
}

func outputSlice(state map[string]any) []string {
	raw, ok := state["output"]
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

func (w *lunchWorkflow) llmNode(ctx *ag.Context, state map[string]any) (ag.Command, error) {
	decisions := w.planner.invoke()
	sends := make([]ag.Send, 0, 4)
	for _, d := range decisions {
		if d.Type == "end" {
			next := cloneState(state)
			next["complete"] = d.Complete
			return ag.Command{
				Goto: []ag.Send{
					{Node: w.names["order"], Arg: next},
				},
			}, nil
		}
		if d.Type == "sub_agent" {
			next := cloneState(state)
			next["sub_agent_input"] = d.SubAgent
			sends = append(sends, ag.Send{Node: w.names["sub"], Arg: next})
		}
		if d.Type == "tool" {
			next := cloneState(state)
			next["tool_input"] = d.Tool
			sends = append(sends, ag.Send{Node: w.names["tool"], Arg: next})
		}
	}
	sends = append(sends, ag.Send{Node: w.names["wait"], Arg: cloneState(state)})
	return ag.Command{Goto: sends}, nil
}

func (w *lunchWorkflow) waitNode(ctx *ag.Context, state map[string]any) (ag.Command, error) {
	event, err := ctx.WaitFor(
		ag.AnyOf(
			ag.ChannelCondition{Channel: "tool_completion_channel", N: 1},
			ag.ChannelCondition{Channel: "subagent_completion_channel", N: 1},
			ag.ChannelCondition{Channel: "user_input_channel", N: 1},
			ag.TimerCondition{Seconds: 1},
		),
	)
	if err != nil {
		return ag.Command{}, err
	}

	output := outputSlice(state)
	if event.Condition == "channel" {
		payload := ag.DecodeString(event.Value)
		switch event.Channel {
		case "tool_completion_channel":
			output = append(output, "tool: "+payload)
		case "subagent_completion_channel":
			output = append(output, "sub_agent: "+payload)
		case "user_input_channel":
			output = append(output, "user_input: "+payload)
		}
		state["output"] = output
		return ag.Command{Goto: []ag.Send{{Node: w.names["llm"], Arg: cloneState(state)}}}, nil
	}

	output = append(output, "timer: no updates yet")
	state["output"] = output
	return ag.Command{Goto: []ag.Send{{Node: w.names["wait"], Arg: cloneState(state)}}}, nil
}

func (w *lunchWorkflow) toolNode(ctx *ag.Context, state map[string]any) (ag.Command, error) {
	toolInput, _ := state["tool_input"].(string)
	time.Sleep(100 * time.Millisecond)
	err := ctx.PublishToChannel("tool_completion_channel", "tool completed for: "+toolInput)
	return ag.Command{}, err
}

func (w *lunchWorkflow) subAgentNode(ctx *ag.Context, state map[string]any) (ag.Command, error) {
	subInput, _ := state["sub_agent_input"].(string)
	time.Sleep(5 * time.Second)
	err := ctx.PublishToChannel(
		"subagent_completion_channel",
		"research sub agent completed for: "+subInput,
	)
	return ag.Command{}, err
}

func (w *lunchWorkflow) orderFoodNode(ctx *ag.Context, state map[string]any) (ag.Command, error) {
	complete, _ := state["complete"].(string)
	output := outputSlice(state)
	output = append(output, "order_food: "+complete)
	state["output"] = output
	state["done"] = complete
	return ag.Command{Update: state}, nil
}

func TestSubAgentsEquivalentFlow(t *testing.T) {
	planner := &mockLLM{
		responses: [][]decision{
			{
				{Type: "sub_agent", SubAgent: "research lunch options"},
				{Type: "tool", Tool: "slack_tool"},
			},
			{},
			{},
			{{Type: "sub_agent", SubAgent: "find vegetarian fallback"}},
			{{Type: "end", Complete: "order submitted"}},
		},
	}
	workflow := &lunchWorkflow{
		planner: planner,
		names:   make(map[string]string),
	}

	graph := ag.NewAdvancedStateGraph()
	graph.AddAsyncChannel("tool_completion_channel")
	graph.AddAsyncChannel("subagent_completion_channel")
	graph.AddAsyncChannel("user_input_channel")

	workflow.names["llm"] = graph.AddNode(workflow.llmNode)
	workflow.names["wait"] = graph.AddNode(workflow.waitNode)
	workflow.names["tool"] = graph.AddNode(workflow.toolNode)
	workflow.names["sub"] = graph.AddNode(workflow.subAgentNode)
	workflow.names["order"] = graph.AddNode(workflow.orderFoodNode)

	graph.SetEntryNode(workflow.llmNode)
	graph.SetFinishNode(workflow.orderFoodNode)

	handler, err := graph.Compile().Start(
		map[string]any{
			"input":  "help me get something for lunch",
			"output": []string{},
			"done":   nil,
		},
	)
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}

	time.Sleep(10 * time.Millisecond)
	if err := handler.PublishToChannel("user_input_channel", "No spicy food please"); err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	result, err := handler.WaitForResult()
	if err != nil {
		t.Fatalf("result failed: %v", err)
	}

	output := outputSlice(result)
	if len(output) == 0 {
		t.Fatalf("output is empty, full result=%#v", result)
	}
	if result["done"] != "order submitted" {
		t.Fatalf("unexpected done: %v", result["done"])
	}
	if !slices.Contains(output, "user_input: No spicy food please") {
		t.Fatalf("missing user input output: %#v", output)
	}
	if !slices.Contains(output, "tool: tool completed for: slack_tool") {
		t.Fatalf("missing tool output: %#v", output)
	}
	if !slices.Contains(output, "sub_agent: research sub agent completed for: research lunch options") {
		t.Fatalf("missing first sub-agent output: %#v", output)
	}
	if !slices.Contains(output, "sub_agent: research sub agent completed for: find vegetarian fallback") {
		t.Fatalf("missing second sub-agent output: %#v", output)
	}
	timerCount := 0
	for _, line := range output {
		if line == "timer: no updates yet" {
			timerCount++
		}
	}
	if timerCount < 3 {
		t.Fatalf("expected >=3 timer outputs, got %d, output=%#v", timerCount, output)
	}
	if output[len(output)-1] != "order_food: order submitted" {
		t.Fatalf("unexpected last output: %#v", output[len(output)-1])
	}
}
