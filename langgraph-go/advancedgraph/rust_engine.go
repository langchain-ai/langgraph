package advancedgraph

/*
#cgo CFLAGS: -I${SRCDIR}/../../rust-core/include
#cgo LDFLAGS: -L${SRCDIR}/../../rust-core/target/debug -llanggraph_rust_core
#include "langgraph_rust_core.h"
#include <stdlib.h>
extern char* goNodeCallback(unsigned long user_data, char* node, char* arg_json, char* state_json);
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
			return cCallbackEnvelopeSuspend(waitReq.Condition)
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

func (e *RustEngine) RunGraph(
	entryPoint string,
	finishPoint string,
	initialState any,
	initialInput any,
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
	centry := C.CString(entryPoint)
	cfinish := C.CString(finishPoint)
	cinitial := C.CString(string(initialJSON))
	cinitialInput := C.CString(string(initialInputJSON))
	defer C.free(unsafe.Pointer(centry))
	defer C.free(unsafe.Pointer(cfinish))
	defer C.free(unsafe.Pointer(cinitial))
	defer C.free(unsafe.Pointer(cinitialInput))

	callbackID := registerRunGraphCallbackCtx(&runGraphCallbackCtx{exec: exec})
	defer unregisterRunGraphCallbackCtx(callbackID)

	resp := C.rc_run_graph_json(
		e.ptr,
		centry,
		cfinish,
		cinitial,
		cinitialInput,
		C.ulong(callbackID),
		(C.rc_node_callback_t)(C.goNodeCallback),
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

func cCallbackEnvelopeSuspend(cond AnyOfCondition) *C.char {
	raw, _ := json.Marshal(map[string]any{
		"ok": true,
		"suspend": map[string]any{
			"kind":   "any_of",
			"any_of": cond,
		},
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
