package advancedgraph

/*
#cgo CFLAGS: -I${SRCDIR}/../../rust-core/include
#cgo LDFLAGS: -L${SRCDIR}/../../rust-core/target/debug -llanggraph_rust_core
#include "langgraph_rust_core.h"
#include <stdlib.h>
extern char* goNodeCallback(unsigned long user_data, char* node, char* arg_json, char* state_json);
static inline char* rc_run_graph_json_with_go_callback(
    Engine* ptr,
    const char* entry_point,
    const char* finish_point,
    const char* initial_state_json,
    const char* initial_input_json,
    const char* stream_mode,
    const char* node_locked_fields_json,
    unsigned long user_data
) {
    return rc_run_graph_json(
        ptr,
        entry_point,
        finish_point,
        initial_state_json,
        initial_input_json,
        stream_mode,
        node_locked_fields_json,
        user_data,
        goNodeCallback
    );
}
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"unsafe"
)

type RustEngine struct {
	ptr *C.Engine
}

type runGraphCallbackCtx struct {
	exec func(node string, nodeInput any, state map[string]any) (Command, error)
}

var (
	callbackRegistryMu sync.RWMutex
	callbackRegistry   = map[uint64]*runGraphCallbackCtx{}
	callbackNextID     uint64
)

func registerRunGraphCallbackCtx(ctx *runGraphCallbackCtx) uint64 {
	id := atomic.AddUint64(&callbackNextID, 1)
	callbackRegistryMu.Lock()
	callbackRegistry[id] = ctx
	callbackRegistryMu.Unlock()
	return id
}

func unregisterRunGraphCallbackCtx(id uint64) {
	callbackRegistryMu.Lock()
	delete(callbackRegistry, id)
	callbackRegistryMu.Unlock()
}

func getRunGraphCallbackCtx(id uint64) (*runGraphCallbackCtx, bool) {
	callbackRegistryMu.RLock()
	ctx, ok := callbackRegistry[id]
	callbackRegistryMu.RUnlock()
	return ctx, ok
}

//export goNodeCallback
func goNodeCallback(userData C.ulong, node *C.char, argJSON *C.char, stateJSON *C.char) *C.char {
	ctx, ok := getRunGraphCallbackCtx(uint64(userData))
	if !ok {
		return cCallbackEnvelopeError("invalid callback context (possibly stale callback)")
	}

	nodeName := C.GoString(node)

	var nodeInput any
	if err := json.Unmarshal([]byte(C.GoString(argJSON)), &nodeInput); err != nil {
		return cCallbackEnvelopeError(fmt.Sprintf("decode arg failed for `%s`: %v", nodeName, err))
	}
	var state map[string]any
	if err := json.Unmarshal([]byte(C.GoString(stateJSON)), &state); err != nil {
		return cCallbackEnvelopeError(fmt.Sprintf("decode state failed for `%s`: %v", nodeName, err))
	}
	nodeInput = coerceJSONValue(nodeInput)
	stateAny := coerceJSONValue(state)
	state, ok = stateAny.(map[string]any)
	if !ok {
		return cCallbackEnvelopeError(fmt.Sprintf("decoded state has unexpected type for `%s`", nodeName))
	}

	cmd, err := ctx.exec(nodeName, nodeInput, state)
	if err != nil {
		if waitReq, ok := AsErrWaitRequested(err); ok {
			return cCallbackEnvelopeSuspend(waitReq.Target)
		}
		return cCallbackEnvelopeError(err.Error())
	}

	sends := make([]map[string]any, 0, len(cmd.Goto))
	for _, send := range cmd.Goto {
		targetNode, err := resolveSendTarget(send.Node)
		if err != nil {
			return cCallbackEnvelopeError(err.Error())
		}
		sends = append(sends, map[string]any{
			"node": targetNode,
			"arg":  send.NodeInput,
		})
	}
	payload := map[string]any{
		"update": cmd.Update,
		"sends":  sends,
	}
	raw, err := json.Marshal(map[string]any{
		"ok":      true,
		"payload": payload,
	})
	if err != nil {
		return cCallbackEnvelopeError(fmt.Sprintf("encode callback payload failed: %v", err))
	}
	return C.CString(string(raw))
}

func NewRustEngine() *RustEngine {
	return &RustEngine{ptr: C.rc_engine_new()}
}

func (e *RustEngine) Close() {
	if e.ptr != nil {
		C.rc_engine_free(e.ptr)
		e.ptr = nil
	}
}

func (e *RustEngine) AddAsyncChannel(channel string) error {
	cch := C.CString(channel)
	defer C.free(unsafe.Pointer(cch))
	resp := C.rc_add_async_channel(e.ptr, cch)
	return parseRustStatus(resp)
}

func (e *RustEngine) AddCustomOutputStream(streamName string) error {
	cname := C.CString(streamName)
	defer C.free(unsafe.Pointer(cname))
	resp := C.rc_add_custom_output_stream(e.ptr, cname)
	return parseRustStatus(resp)
}

func (e *RustEngine) StartStream(streamMode string) error {
	var cmode *C.char
	if streamMode != "" {
		cmode = C.CString(streamMode)
		defer C.free(unsafe.Pointer(cmode))
	}
	resp := C.rc_start_stream(e.ptr, cmode)
	return parseRustStatus(resp)
}

func (e *RustEngine) ReceiveStream(streamName string) (any, bool, error) {
	cname := C.CString(streamName)
	defer C.free(unsafe.Pointer(cname))
	resp := C.rc_receive_stream_json(e.ptr, cname)
	defer C.rc_string_free(resp)

	raw := C.GoString(resp)
	var status struct {
		OK       bool            `json:"ok"`
		Error    string          `json:"error"`
		HasEvent bool            `json:"has_event"`
		Event    json.RawMessage `json:"event"`
	}
	if err := json.Unmarshal([]byte(raw), &status); err != nil {
		return nil, false, fmt.Errorf("decode rust stream response: %w", err)
	}
	if !status.OK {
		return nil, false, fmt.Errorf("rust stream failed: %s", status.Error)
	}
	if !status.HasEvent {
		return nil, false, nil
	}
	var event any
	if err := json.Unmarshal(status.Event, &event); err != nil {
		return nil, false, fmt.Errorf("decode stream event: %w", err)
	}
	return coerceJSONValue(event), true, nil
}

func (e *RustEngine) SendCustomStreamEvent(streamName string, value any) error {
	payload, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("marshal stream event: %w", err)
	}
	cname := C.CString(streamName)
	cval := C.CString(string(payload))
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cval))
	resp := C.rc_send_custom_stream_event(e.ptr, cname, cval)
	return parseRustStatus(resp)
}

func (e *RustEngine) CloseAllStreams() error {
	resp := C.rc_close_all_streams(e.ptr)
	return parseRustStatus(resp)
}

func (e *RustEngine) Publish(channel string, value any) error {
	payload, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("marshal publish value: %w", err)
	}
	cch := C.CString(channel)
	cval := C.CString(string(payload))
	defer C.free(unsafe.Pointer(cch))
	defer C.free(unsafe.Pointer(cval))
	resp := C.rc_publish_json(e.ptr, cch, cval)
	return parseRustStatus(resp)
}

func (e *RustEngine) WaitAnyOf(cond AnyOfCondition) (WaitEvent, error) {
	payload, err := json.Marshal(cond)
	if err != nil {
		return WaitEvent{}, fmt.Errorf("marshal any_of: %w", err)
	}
	cpayload := C.CString(string(payload))
	defer C.free(unsafe.Pointer(cpayload))
	resp := C.rc_wait_any_of_json(e.ptr, cpayload)
	defer C.rc_string_free(resp)

	raw := C.GoString(resp)
	var status struct {
		OK    bool            `json:"ok"`
		Error string          `json:"error"`
		Event json.RawMessage `json:"event"`
	}
	if err := json.Unmarshal([]byte(raw), &status); err != nil {
		return WaitEvent{}, fmt.Errorf("decode rust wait response: %w", err)
	}
	if !status.OK {
		return WaitEvent{}, fmt.Errorf("rust wait failed: %s", status.Error)
	}
	var event WaitEvent
	if err := json.Unmarshal(status.Event, &event); err != nil {
		return WaitEvent{}, fmt.Errorf("decode wait event: %w", err)
	}
	return event, nil
}

func (e *RustEngine) WaitAllOf(cond AllOfCondition) (WaitEvent, error) {
	payload, err := json.Marshal(cond)
	if err != nil {
		return WaitEvent{}, fmt.Errorf("marshal all_of: %w", err)
	}
	cpayload := C.CString(string(payload))
	defer C.free(unsafe.Pointer(cpayload))
	resp := C.rc_wait_all_of_json(e.ptr, cpayload)
	defer C.rc_string_free(resp)

	raw := C.GoString(resp)
	var status struct {
		OK    bool            `json:"ok"`
		Error string          `json:"error"`
		Event json.RawMessage `json:"event"`
	}
	if err := json.Unmarshal([]byte(raw), &status); err != nil {
		return WaitEvent{}, fmt.Errorf("decode rust wait response: %w", err)
	}
	if !status.OK {
		return WaitEvent{}, fmt.Errorf("rust wait failed: %s", status.Error)
	}
	var event WaitEvent
	if err := json.Unmarshal(status.Event, &event); err != nil {
		return WaitEvent{}, fmt.Errorf("decode wait event: %w", err)
	}
	return event, nil
}

func (e *RustEngine) RunGraph(
	entryPoint string,
	finishPoint string,
	streamMode string,
	initialState any,
	initialInput any,
	nodeLockedFields map[string][]string,
	exec func(node string, nodeInput any, state map[string]any) (Command, error),
) (map[string]any, error) {
	initialJSON, err := json.Marshal(initialState)
	if err != nil {
		return nil, fmt.Errorf("marshal initial state: %w", err)
	}
	initialInputJSON, err := json.Marshal(initialInput)
	if err != nil {
		return nil, fmt.Errorf("marshal initial input: %w", err)
	}
	lockedFieldsJSON, err := json.Marshal(nodeLockedFields)
	if err != nil {
		return nil, fmt.Errorf("marshal node locked fields: %w", err)
	}
	centry := C.CString(entryPoint)
	cfinish := C.CString(finishPoint)
	cinitial := C.CString(string(initialJSON))
	cinitialInput := C.CString(string(initialInputJSON))
	clockedFields := C.CString(string(lockedFieldsJSON))
	var cstreamMode *C.char
	if streamMode != "" {
		cstreamMode = C.CString(streamMode)
	}
	defer C.free(unsafe.Pointer(centry))
	defer C.free(unsafe.Pointer(cfinish))
	defer C.free(unsafe.Pointer(cinitial))
	defer C.free(unsafe.Pointer(cinitialInput))
	defer C.free(unsafe.Pointer(clockedFields))
	if cstreamMode != nil {
		defer C.free(unsafe.Pointer(cstreamMode))
	}

	callbackID := registerRunGraphCallbackCtx(&runGraphCallbackCtx{exec: exec})
	defer unregisterRunGraphCallbackCtx(callbackID)

	resp := C.rc_run_graph_json_with_go_callback(
		e.ptr,
		centry,
		cfinish,
		cinitial,
		cinitialInput,
		cstreamMode,
		clockedFields,
		C.ulong(callbackID),
	)
	defer C.rc_string_free(resp)

	raw := C.GoString(resp)
	var status struct {
		OK    bool           `json:"ok"`
		Error string         `json:"error"`
		State map[string]any `json:"state"`
	}
	if err := json.Unmarshal([]byte(raw), &status); err != nil {
		return nil, fmt.Errorf("decode rust run response: %w", err)
	}
	if !status.OK {
		return nil, fmt.Errorf("rust run failed: %s", status.Error)
	}
	coerced := coerceJSONValue(status.State)
	typed, ok := coerced.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected state type from rust run")
	}
	return typed, nil
}

func cCallbackEnvelopeError(message string) *C.char {
	raw, _ := json.Marshal(map[string]any{
		"ok":    false,
		"error": message,
	})
	return C.CString(string(raw))
}

func cCallbackEnvelopeSuspend(target WaitTarget) *C.char {
	if target == nil {
		return cCallbackEnvelopeError("wait requested with nil target")
	}
	kind := target.waitKind()
	if kind != "any_of" && kind != "all_of" {
		return cCallbackEnvelopeError(fmt.Sprintf("unsupported wait target kind `%s`", kind))
	}
	var payload map[string]any
	if kind == "any_of" {
		payload = map[string]any{
			"kind":   "any_of",
			"any_of": AnyOfCondition{Conditions: target.waitConditions()},
		}
	} else {
		payload = map[string]any{
			"kind":   "all_of",
			"all_of": AllOfCondition{Conditions: target.waitConditions()},
		}
	}
	raw, _ := json.Marshal(map[string]any{
		"ok":      true,
		"suspend": payload,
	})
	return C.CString(string(raw))
}

func resolveSendTarget(target any) (string, error) {
	if name, ok := target.(string); ok {
		if name == "" {
			return "", fmt.Errorf("send target cannot be empty string")
		}
		return name, nil
	}
	rv := reflect.ValueOf(target)
	if rv.IsValid() && rv.Kind() == reflect.Func {
		return NodeName(target), nil
	}
	return "", fmt.Errorf("unsupported send target type %T", target)
}

func coerceJSONValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = coerceJSONValue(val)
		}
		return out
	case []any:
		coerced := make([]any, len(t))
		allStrings := true
		for i, val := range t {
			cv := coerceJSONValue(val)
			coerced[i] = cv
			if _, ok := cv.(string); !ok {
				allStrings = false
			}
		}
		if allStrings {
			out := make([]string, len(coerced))
			for i, item := range coerced {
				out[i] = item.(string)
			}
			return out
		}
		return coerced
	default:
		return v
	}
}

func parseRustStatus(resp *C.char) error {
	defer C.rc_string_free(resp)
	raw := C.GoString(resp)
	var status struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(raw), &status); err != nil {
		return fmt.Errorf("decode rust response: %w", err)
	}
	if !status.OK {
		return fmt.Errorf("rust error: %s", status.Error)
	}
	return nil
}
