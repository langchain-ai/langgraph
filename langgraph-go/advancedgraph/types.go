package advancedgraph

import (
	"encoding/json"
	"errors"
)

type WaitCondition interface {
	toAny() map[string]any
}

type ChannelCondition struct {
	Channel string
	Min     int
	Max     int
}

func (c ChannelCondition) toAny() map[string]any {
	min := c.Min
	if min <= 0 {
		min = 1
	}
	return map[string]any{
		"kind":    "channel",
		"channel": c.Channel,
		"min":     min,
		"max":     c.Max,
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

type ConditionResult struct {
	Met         bool   `json:"met"`
	ChannelName string `json:"channel_name,omitempty"`
	Values      []any  `json:"values,omitempty"`
}

type WaitForResult struct {
	Conditions []ConditionResult `json:"conditions"`
}

type Send struct {
	Node      any
	NodeInput any
}

type Command struct {
	Update any
	Goto   []Send
}

type ErrWaitRequested struct {
	Condition AnyOfCondition
}

func (e ErrWaitRequested) Error() string {
	return "wait requested"
}

func AsErrWaitRequested(err error) (ErrWaitRequested, bool) {
	var target ErrWaitRequested
	if !errors.As(err, &target) {
		return target, false
	}
	return target, true
}

func DecodeString(raw json.RawMessage) string {
	var s string
	_ = json.Unmarshal(raw, &s)
	return s
}
