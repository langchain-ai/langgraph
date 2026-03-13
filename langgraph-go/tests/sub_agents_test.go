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

type mockPlanner struct {
	responses [][]decision
	i         int
}

func (m *mockPlanner) ainvoke() []decision {
	if m.i >= len(m.responses) {
		return []decision{}
	}
	resp := m.responses[m.i]
	m.i++
	return resp
}

func TestSubAgentsEquivalentFlow(t *testing.T) {
	planner := &mockPlanner{
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

	graph := ag.NewAdvancedStateGraph()
	graph.AddAsyncChannel("tool_completion_channel")
	graph.AddAsyncChannel("subagent_completion_channel")
	graph.AddAsyncChannel("user_input_channel")

	graph.AddNode("llm_node", func(ctx *ag.Context, arg any) (ag.Command, error) {
		state := arg.(map[string]any)
		decisions := planner.ainvoke()
		sends := make([]ag.Send, 0, 4)
		for _, d := range decisions {
			if d.Type == "end" {
				return ag.Command{
					Goto: []ag.Send{
						{
							Node: "order_food_node",
							Arg: map[string]any{
								"state":    state,
								"complete": d.Complete,
							},
						},
					},
				}, nil
			}
			if d.Type == "sub_agent" {
				sends = append(sends, ag.Send{Node: "sub_agent_node", Arg: d.SubAgent})
			}
			if d.Type == "tool" {
				sends = append(sends, ag.Send{Node: "tool_node", Arg: d.Tool})
			}
		}
		sends = append(sends, ag.Send{Node: "wait_node", Arg: state})
		return ag.Command{Goto: sends}, nil
	})

	graph.AddNode("wait_node", func(ctx *ag.Context, arg any) (ag.Command, error) {
		state := arg.(map[string]any)
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

		output := state["output"].([]string)
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
			return ag.Command{Goto: []ag.Send{{Node: "llm_node", Arg: state}}}, nil
		}

		output = append(output, "timer: no updates yet")
		state["output"] = output
		return ag.Command{Goto: []ag.Send{{Node: "wait_node", Arg: state}}}, nil
	})

	graph.AddNode("tool_node", func(ctx *ag.Context, arg any) (ag.Command, error) {
		toolInput := arg.(string)
		time.Sleep(100 * time.Millisecond)
		err := ctx.PublishToChannel("tool_completion_channel", "tool completed for: "+toolInput)
		return ag.Command{}, err
	})

	graph.AddNode("sub_agent_node", func(ctx *ag.Context, arg any) (ag.Command, error) {
		subInput := arg.(string)
		time.Sleep(5 * time.Second)
		err := ctx.PublishToChannel(
			"subagent_completion_channel",
			"research sub agent completed for: "+subInput,
		)
		return ag.Command{}, err
	})

	graph.AddNode("order_food_node", func(ctx *ag.Context, arg any) (ag.Command, error) {
		payload := arg.(map[string]any)
		state := payload["state"].(map[string]any)
		complete := payload["complete"].(string)
		output := state["output"].([]string)
		output = append(output, "order_food: "+complete)
		state["output"] = output
		state["done"] = complete
		return ag.Command{Update: state}, nil
	})

	graph.SetEntryPoint("llm_node")
	graph.SetFinishPoint("order_food_node")

	handler, err := graph.Compile().AStart(
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
	if err := handler.APublishToChannel("user_input_channel", "No spicy food please"); err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	result, err := handler.AResult()
	if err != nil {
		t.Fatalf("result failed: %v", err)
	}

	output := result["output"].([]string)
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

