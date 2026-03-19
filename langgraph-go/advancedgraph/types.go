package advancedgraph

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
)

type WaitCondition interface {
	json.Marshaler
}

type ChannelCondition struct {
	Channel string
	Min     int
	Max     int
}

type channelConditionJSON struct {
	Kind    string `json:"kind"`
	Channel string `json:"channel"`
	Min     int    `json:"min"`
	Max     int    `json:"max"`
}

func (c ChannelCondition) MarshalJSON() ([]byte, error) {
	min := c.Min
	if min <= 0 {
		min = 1
	}
	return json.Marshal(channelConditionJSON{
		Kind:    "channel",
		Channel: c.Channel,
		Min:     min,
		Max:     c.Max,
	})
}

type TimerCondition struct {
	Seconds float64
}

type timerConditionJSON struct {
	Kind    string  `json:"kind"`
	Seconds float64 `json:"seconds"`
}

func (t TimerCondition) MarshalJSON() ([]byte, error) {
	return json.Marshal(timerConditionJSON{
		Kind:    "timer",
		Seconds: t.Seconds,
	})
}

type AnyOfCondition struct {
	Conditions []WaitCondition
}

type AllOfCondition struct {
	Conditions []WaitCondition
}

type WaitTarget interface {
	waitKind() string
	waitConditions() []WaitCondition
}

func (a AnyOfCondition) waitKind() string {
	return "any_of"
}

func (a AnyOfCondition) waitConditions() []WaitCondition {
	return a.Conditions
}

func (a AllOfCondition) waitKind() string {
	return "all_of"
}

func (a AllOfCondition) waitConditions() []WaitCondition {
	return a.Conditions
}

func AnyOf(conditions ...WaitCondition) AnyOfCondition {
	return AnyOfCondition{Conditions: append([]WaitCondition{}, conditions...)}
}

func AllOf(conditions ...WaitCondition) AllOfCondition {
	return AllOfCondition{Conditions: append([]WaitCondition{}, conditions...)}
}

func (a AnyOfCondition) MarshalJSON() ([]byte, error) {
	result := struct {
		Conditions []json.RawMessage `json:"conditions"`
	}{
		Conditions: make([]json.RawMessage, 0, len(a.Conditions)),
	}
	for i, cond := range a.Conditions {
		if isNilWaitCondition(cond) {
			return nil, fmt.Errorf("any_of condition[%d] is nil", i)
		}
		raw, err := cond.MarshalJSON()
		if err != nil {
			return nil, fmt.Errorf("encode any_of condition[%d]: %w", i, err)
		}
		result.Conditions = append(result.Conditions, json.RawMessage(raw))
	}
	return json.Marshal(result)
}

func (a AllOfCondition) MarshalJSON() ([]byte, error) {
	result := struct {
		Conditions []json.RawMessage `json:"conditions"`
	}{
		Conditions: make([]json.RawMessage, 0, len(a.Conditions)),
	}
	for i, cond := range a.Conditions {
		if isNilWaitCondition(cond) {
			return nil, fmt.Errorf("all_of condition[%d] is nil", i)
		}
		raw, err := cond.MarshalJSON()
		if err != nil {
			return nil, fmt.Errorf("encode all_of condition[%d]: %w", i, err)
		}
		result.Conditions = append(result.Conditions, json.RawMessage(raw))
	}
	return json.Marshal(result)
}

func isNilWaitCondition(cond WaitCondition) bool {
	if cond == nil {
		return true
	}
	v := reflect.ValueOf(cond)
	switch v.Kind() {
	case reflect.Ptr, reflect.Interface, reflect.Slice, reflect.Map, reflect.Func:
		return v.IsNil()
	default:
		return false
	}
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
	Target WaitTarget
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
