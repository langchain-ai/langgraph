package tests

import (
	"strings"
	"testing"
	"time"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

type streamState struct {
	Done bool `json:"done"`
}

type streamWorkflow struct{}

func (w *streamWorkflow) startNode(ctx *ag.Context, _ any, state streamState) (ag.Command, error) {
	if err := ctx.SendCustomStreamEvent("high", map[string]any{"step": "start", "value": 1}); err != nil {
		return ag.Command{}, err
	}
	time.Sleep(80 * time.Millisecond)
	if err := ctx.SendCustomStreamEvent("regular", map[string]any{"step": "start", "value": 2}); err != nil {
		return ag.Command{}, err
	}
	return ag.Command{
		Update: state,
		Goto: []ag.Send{
			{Node: w.finishNode},
		},
	}, nil
}

func (w *streamWorkflow) finishNode(ctx *ag.Context, _ any, state streamState) (ag.Command, error) {
	state.Done = true
	return ag.Command{Update: state}, nil
}

func TestCustomStreamReceiveAndClose(t *testing.T) {
	workflow := &streamWorkflow{}
	graph := ag.NewAdvancedStateGraph[streamState]()
	graph.AddCustomOutputStream("high")
	graph.AddCustomOutputStream("regular")
	graph.AddEntryNode(workflow.startNode)
	graph.AddFinishNode(workflow.finishNode)

	handler, err := graph.Compile().Start(nil, streamState{Done: false}, "custom")
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}

	event, err := handler.ReceiveStream("high")
	if err != nil {
		t.Fatalf("receive stream failed: %v", err)
	}
	if event == nil {
		t.Fatalf("expected first stream event, got nil")
	}
	eventMap, ok := event.(map[string]any)
	if !ok {
		t.Fatalf("unexpected event type: %T", event)
	}
	if eventMap["step"] != "start" {
		t.Fatalf("unexpected stream event payload: %#v", eventMap)
	}

	eventRegular, err := handler.ReceiveStream("regular")
	if err != nil {
		t.Fatalf("receive regular stream failed: %v", err)
	}
	eventRegularMap, ok := eventRegular.(map[string]any)
	if !ok {
		t.Fatalf("unexpected regular event type: %T", eventRegular)
	}
	if eventRegularMap["value"] != float64(2) {
		t.Fatalf("unexpected regular stream payload: %#v", eventRegularMap)
	}

	if err := handler.CloseAllStreams(); err != nil {
		t.Fatalf("close all streams failed: %v", err)
	}

	closedEvent, err := handler.ReceiveStream("high")
	if err != nil {
		t.Fatalf("receive stream after close failed: %v", err)
	}
	if closedEvent != nil {
		t.Fatalf("expected nil stream event after close, got %#v", closedEvent)
	}

	result, err := handler.WaitForResult()
	if err != nil {
		t.Fatalf("result failed: %v", err)
	}
	if !result.Done {
		t.Fatalf("expected final state done=true, got %#v", result)
	}
}

func TestOnlyCustomStreamModeSupported(t *testing.T) {
	workflow := &streamWorkflow{}
	graph := ag.NewAdvancedStateGraph[streamState]()
	graph.AddCustomOutputStream("regular")
	graph.AddEntryNode(workflow.startNode)
	graph.AddFinishNode(workflow.finishNode)

	handler, err := graph.Compile().Start(nil, streamState{Done: false}, "values")
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	_, runErr := handler.WaitForResult()
	if runErr == nil {
		t.Fatalf("expected run error for unsupported stream mode")
	}
	if !strings.Contains(runErr.Error(), "only `custom` is supported") {
		t.Fatalf("unexpected error: %v", runErr)
	}
}
