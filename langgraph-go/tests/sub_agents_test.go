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
}

type lunchState struct {
	Input  string   `json:"input"`
	Output []string `json:"output"`
	Done   string   `json:"done"`
}

func (w *lunchWorkflow) llmNode(ctx *ag.Context, _ any, _ lunchState) (ag.Command, error) {
	decisions := w.planner.invoke()
	sends := make([]ag.Send, 0, 4)
	for _, d := range decisions {
		if d.Type == "end" {
			return ag.Command{
				Goto: []ag.Send{
					{Node: w.orderFoodNode, NodeInput: d.Complete},
				},
			}, nil
		}
		if d.Type == "sub_agent" {
			sends = append(sends, ag.Send{Node: w.subAgentNode, NodeInput: d.SubAgent})
		}
		if d.Type == "tool" {
			sends = append(sends, ag.Send{Node: w.toolNode, NodeInput: d.Tool})
		}
	}
	sends = append(sends, ag.Send{Node: w.waitNode})
	return ag.Command{Goto: sends}, nil
}

func (w *lunchWorkflow) waitNode(ctx *ag.Context, _ any, state lunchState) (ag.Command, error) {
	event, err := ctx.WaitFor(
		ag.AnyOf(
			ag.ChannelCondition{Channel: "tool_completion_channel"},
			ag.ChannelCondition{Channel: "subagent_completion_channel"},
			ag.ChannelCondition{Channel: "user_input_channel"},
			ag.TimerCondition{Seconds: 1},
		),
	)
	if err != nil {
		return ag.Command{}, err
	}

	output := append([]string(nil), state.Output...)
	hadChannel := false
	for _, cond := range event.Conditions {
		if !cond.Met || cond.ChannelName == "" {
			continue
		}
		for _, raw := range cond.Values {
			payload, _ := raw.(string)
			switch cond.ChannelName {
			case "tool_completion_channel":
				output = append(output, "tool: "+payload)
			case "subagent_completion_channel":
				output = append(output, "sub_agent: "+payload)
			case "user_input_channel":
				output = append(output, "user_input: "+payload)
			}
		}
		hadChannel = true
	}
	if hadChannel {
		state.Output = output
		return ag.Command{Goto: []ag.Send{{Node: w.llmNode}}, Update: state}, nil
	}

	output = append(output, "timer: no updates yet")
	state.Output = output
	return ag.Command{Goto: []ag.Send{{Node: w.waitNode}}, Update: state}, nil
}

func (w *lunchWorkflow) toolNode(ctx *ag.Context, input any, _ lunchState) (ag.Command, error) {
	toolInput, _ := input.(string)
	time.Sleep(100 * time.Millisecond)
	err := ctx.PublishToChannel("tool_completion_channel", "tool completed for: "+toolInput)
	return ag.Command{}, err
}

func (w *lunchWorkflow) subAgentNode(ctx *ag.Context, input any, _ lunchState) (ag.Command, error) {
	subInput, _ := input.(string)
	time.Sleep(5 * time.Second)
	err := ctx.PublishToChannel(
		"subagent_completion_channel",
		"research sub agent completed for: "+subInput,
	)
	return ag.Command{}, err
}

func (w *lunchWorkflow) orderFoodNode(ctx *ag.Context, input any, state lunchState) (ag.Command, error) {
	complete, _ := input.(string)
	output := append([]string(nil), state.Output...)
	output = append(output, "order_food: "+complete)
	state.Output = output
	state.Done = complete
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
	}

	graph := ag.NewAdvancedStateGraph[lunchState]()
	graph.AddAsyncChannel("tool_completion_channel")
	graph.AddAsyncChannel("subagent_completion_channel")
	graph.AddAsyncChannel("user_input_channel")

	graph.AddEntryNode(workflow.llmNode)
	graph.AddNode(workflow.waitNode)
	graph.AddNode(workflow.toolNode)
	graph.AddNode(workflow.subAgentNode)
	graph.AddFinishNode(workflow.orderFoodNode)

	handler, err := graph.Compile().Start(
		nil,
		lunchState{
			Input:  "help me get something for lunch",
			Output: []string{},
			Done:   "",
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

	output := result.Output
	if len(output) == 0 {
		t.Fatalf("output is empty, full result=%#v", result)
	}
	if result.Done != "order submitted" {
		t.Fatalf("unexpected done: %v", result.Done)
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
