package advancedgraph

import "encoding/json"

type WaitCondition interface {
	toAny() map[string]any
}

type ChannelCondition struct {
	Channel string
	N       int
}

func (c ChannelCondition) toAny() map[string]any {
	n := c.N
	if n <= 0 {
		n = 1
	}
	return map[string]any{
		"kind":    "channel",
		"channel": c.Channel,
		"n":       n,
	}
}

type TimerCondition struct {
	Seconds float64
}

func (t TimerCondition) toAny() map[string]any {
	return map[string]any{
		"kind":    "timer",
		"seconds": t.Seconds,
	}
}

type AnyOfCondition struct {
	Conditions []map[string]any `json:"conditions"`
}

func AnyOf(conditions ...WaitCondition) AnyOfCondition {
	result := AnyOfCondition{Conditions: make([]map[string]any, 0, len(conditions))}
	for _, cond := range conditions {
		result.Conditions = append(result.Conditions, cond.toAny())
	}
	return result
}

type WaitEvent struct {
	Condition string          `json:"condition"`
	Channel   string          `json:"channel,omitempty"`
	Value     json.RawMessage `json:"value,omitempty"`
	Seconds   float64         `json:"seconds,omitempty"`
}

type Send struct {
	Node      any
	NodeInput any
}

type Command struct {
	Update map[string]any
	Goto   []Send
}

func DecodeString(raw json.RawMessage) string {
	var s string
	_ = json.Unmarshal(raw, &s)
	return s
}
