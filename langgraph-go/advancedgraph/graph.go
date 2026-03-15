package advancedgraph

import (
	"encoding/json"
	"fmt"
	"reflect"
	"runtime"
	"strings"
)

type nodeExecutor func(ctx *Context, input any, state map[string]any) (Command, error)

type AdvancedStateGraph[StateT any] struct {
	nodes         map[string]nodeExecutor
	asyncChannels []string
	entryPoint    string
	finishPoint   string
	stateType     reflect.Type
}

func NewAdvancedStateGraph[StateT any]() *AdvancedStateGraph[StateT] {
	stateType := mustTypeOf[StateT]()
	if stateType.Kind() != reflect.Struct {
		panic(fmt.Sprintf("StateT must be a struct, got %s", stateType.String()))
	}
	return &AdvancedStateGraph[StateT]{
		nodes:     make(map[string]nodeExecutor),
		stateType: stateType,
	}
}

// AddNode keeps `fn` as `any` because advanced graph nodes can have different
// input argument types per node, while only `StateT` is globally constrained.
// We validate and adapt node signatures at runtime in compileNodeExecutor.
func (g *AdvancedStateGraph[StateT]) AddNode(fn any) string {
	name := NodeName(fn)
	return g.AddNodeAs(name, fn)
}

func (g *AdvancedStateGraph[StateT]) AddNodeAs(name string, fn any) string {
	if _, exists := g.nodes[name]; exists {
		panic(fmt.Sprintf("node `%s` already exists", name))
	}
	exec, err := compileNodeExecutor(fn, g.stateType)
	if err != nil {
		panic(err)
	}
	g.nodes[name] = exec
	return name
}

func (g *AdvancedStateGraph[StateT]) AddAsyncChannel(name string) {
	g.asyncChannels = append(g.asyncChannels, name)
}

func (g *AdvancedStateGraph[StateT]) AddEntryNode(fn any) string {
	name := NodeName(fn)
	return g.AddEntryNodeAs(name, fn)
}

func (g *AdvancedStateGraph[StateT]) AddEntryNodeAs(name string, fn any) string {
	name = g.AddNodeAs(name, fn)
	g.entryPoint = name
	return name
}

func (g *AdvancedStateGraph[StateT]) AddFinishNode(fn any) string {
	name := NodeName(fn)
	return g.AddFinishNodeAs(name, fn)
}

func (g *AdvancedStateGraph[StateT]) AddFinishNodeAs(name string, fn any) string {
	name = g.AddNodeAs(name, fn)
	g.finishPoint = name
	return name
}

func (g *AdvancedStateGraph[StateT]) Compile() *CompiledGraph[StateT] {
	return &CompiledGraph[StateT]{
		nodes:         g.nodes,
		asyncChannels: g.asyncChannels,
		entryPoint:    g.entryPoint,
		finishPoint:   g.finishPoint,
		stateType:     g.stateType,
	}
}

type CompiledGraph[StateT any] struct {
	nodes         map[string]nodeExecutor
	asyncChannels []string
	entryPoint    string
	finishPoint   string
	stateType     reflect.Type
}

type Context struct {
	engine      *RustEngine
	resumeEvent *WaitEvent
}

func (c *Context) WaitFor(cond AnyOfCondition) (WaitEvent, error) {
	if c.resumeEvent != nil {
		event := *c.resumeEvent
		c.resumeEvent = nil
		return event, nil
	}
	return WaitEvent{}, ErrWaitRequested{Condition: cond}
}

func (c *Context) PublishToChannel(channel string, value any) error {
	return c.engine.Publish(channel, value)
}

type Handler[StateT any] struct {
	engine *RustEngine
	done   chan resultOrErr[StateT]
}

type resultOrErr[StateT any] struct {
	state StateT
	err   error
}

func (h *Handler[StateT]) PublishToChannel(channel string, value any) error {
	return h.engine.Publish(channel, value)
}

func (h *Handler[StateT]) WaitForResult() (StateT, error) {
	res := <-h.done
	return res.state, res.err
}

func (g *CompiledGraph[StateT]) Start(initialInput any, initialState StateT) (*Handler[StateT], error) {
	engine := NewRustEngine()
	for _, ch := range g.asyncChannels {
		if err := engine.AddAsyncChannel(ch); err != nil {
			return nil, err
		}
	}

	handler := &Handler[StateT]{
		engine: engine,
		done:   make(chan resultOrErr[StateT], 1),
	}
	go func() {
		defer engine.Close()
		rawState, err := engine.RunGraph(
			g.entryPoint,
			g.finishPoint,
			initialState,
			initialInput,
			func(node string, nodeInput any, fallbackState map[string]any) (Command, error) {
				fn, ok := g.nodes[node]
				if !ok {
					return Command{}, fmt.Errorf("unknown node `%s`", node)
				}
				if fallbackState == nil {
					return Command{}, fmt.Errorf("node `%s` expected map state argument", node)
				}
				resolvedInput, resumeEvent := unwrapResumeInput(nodeInput)
				return fn(&Context{engine: engine, resumeEvent: resumeEvent}, resolvedInput, fallbackState)
			},
		)
		if err != nil {
			handler.done <- resultOrErr[StateT]{err: err}
			close(handler.done)
			return
		}
		state, err := mapToState[StateT](rawState)
		handler.done <- resultOrErr[StateT]{state: state, err: err}
		close(handler.done)
	}()
	return handler, nil
}

func NodeName(fn any) string {
	rv := reflect.ValueOf(fn)
	if !rv.IsValid() || rv.Kind() != reflect.Func {
		panic("cannot infer node name from non-function value")
	}
	pc := rv.Pointer()
	f := runtime.FuncForPC(pc)
	if f == nil {
		panic("cannot infer node name from nil function")
	}
	full := f.Name()
	if strings.Contains(full, ".func") {
		panic("anonymous functions are not allowed as nodes")
	}
	short := full
	if i := strings.LastIndex(short, "/"); i >= 0 {
		short = short[i+1:]
	}
	if i := strings.LastIndex(short, "."); i >= 0 {
		short = short[i+1:]
	}
	short = strings.TrimSuffix(short, "-fm")
	if short == "" || strings.Contains(short, "func") {
		panic(fmt.Sprintf("cannot infer stable node name from `%s`", full))
	}
	return short
}

func compileNodeExecutor(fn any, expectedStateType reflect.Type) (nodeExecutor, error) {
	rv := reflect.ValueOf(fn)
	if !rv.IsValid() || rv.Kind() != reflect.Func {
		return nil, fmt.Errorf("node must be a function")
	}
	rt := rv.Type()
	if rt.NumIn() != 3 {
		return nil, fmt.Errorf("node `%s` must accept exactly 3 args: (*Context, input, state)", NodeName(fn))
	}
	ctxType := reflect.TypeOf((*Context)(nil))
	if rt.In(0) != ctxType {
		return nil, fmt.Errorf("node `%s` first arg must be *Context", NodeName(fn))
	}
	if rt.NumOut() != 2 {
		return nil, fmt.Errorf("node `%s` must return (Command, error)", NodeName(fn))
	}
	cmdType := reflect.TypeOf(Command{})
	if rt.Out(0) != cmdType {
		return nil, fmt.Errorf("node `%s` first return must be Command", NodeName(fn))
	}
	errType := reflect.TypeOf((*error)(nil)).Elem()
	if !rt.Out(1).Implements(errType) {
		return nil, fmt.Errorf("node `%s` second return must be error", NodeName(fn))
	}

	inputType := rt.In(1)
	stateType := rt.In(2)
	if stateType != expectedStateType {
		return nil, fmt.Errorf(
			"node `%s` state type mismatch: got %s, graph expects %s",
			NodeName(fn),
			stateType.String(),
			expectedStateType.String(),
		)
	}
	return func(ctx *Context, input any, state map[string]any) (Command, error) {
		stateArg, err := convertStateArg(state, stateType)
		if err != nil {
			return Command{}, fmt.Errorf("node `%s` state decode failed: %w", NodeName(fn), err)
		}
		args := []reflect.Value{
			reflect.ValueOf(ctx),
			reflect.Zero(inputType),
			stateArg,
		}
		if input != nil {
			inVal := reflect.ValueOf(input)
			if inVal.Type().AssignableTo(inputType) {
				args[1] = inVal
			} else if inVal.Type().ConvertibleTo(inputType) {
				args[1] = inVal.Convert(inputType)
			} else {
				return Command{}, fmt.Errorf(
					"node `%s` input type mismatch: got %T, want %s",
					NodeName(fn),
					input,
					inputType.String(),
				)
			}
		}
		out := rv.Call(args)
		cmd := out[0].Interface().(Command)
		if cmd.Update != nil {
			updateType := reflect.TypeOf(cmd.Update)
			if updateType != stateType {
				return Command{}, fmt.Errorf(
					"node `%s` update type mismatch: got %s, graph expects %s",
					NodeName(fn),
					updateType.String(),
					stateType.String(),
				)
			}
		}
		if out[1].IsNil() {
			return cmd, nil
		}
		return cmd, out[1].Interface().(error)
	}, nil
}

func convertStateArg(state map[string]any, stateType reflect.Type) (reflect.Value, error) {
	if stateType == reflect.TypeOf(map[string]any{}) {
		return reflect.ValueOf(state), nil
	}
	raw, err := json.Marshal(state)
	if err != nil {
		return reflect.Value{}, fmt.Errorf("marshal state: %w", err)
	}
	if stateType.Kind() == reflect.Ptr {
		target := reflect.New(stateType.Elem())
		if err := json.Unmarshal(raw, target.Interface()); err != nil {
			return reflect.Value{}, fmt.Errorf("unmarshal state into %s: %w", stateType.String(), err)
		}
		return target, nil
	}
	target := reflect.New(stateType)
	if err := json.Unmarshal(raw, target.Interface()); err != nil {
		return reflect.Value{}, fmt.Errorf("unmarshal state into %s: %w", stateType.String(), err)
	}
	return target.Elem(), nil
}

func mapToState[StateT any](raw map[string]any) (StateT, error) {
	var out StateT
	if anyVal, ok := any(raw).(StateT); ok {
		return anyVal, nil
	}
	payload, err := json.Marshal(raw)
	if err != nil {
		return out, fmt.Errorf("marshal state: %w", err)
	}
	if err := json.Unmarshal(payload, &out); err != nil {
		return out, fmt.Errorf("unmarshal state: %w", err)
	}
	return out, nil
}

func mustTypeOf[T any]() reflect.Type {
	var zero T
	t := reflect.TypeOf(zero)
	if t != nil {
		return t
	}
	// Handles nil-able types where zero value has no dynamic type.
	return reflect.TypeOf((*T)(nil)).Elem()
}

func unwrapResumeInput(input any) (any, *WaitEvent) {
	wrapper, ok := input.(map[string]any)
	if !ok {
		return input, nil
	}
	rawArg, hasArg := wrapper["__lg_resume_arg__"]
	rawEvent, hasEvent := wrapper["__lg_resume_event__"]
	if !hasArg || !hasEvent {
		return input, nil
	}
	eventPayload, err := json.Marshal(rawEvent)
	if err != nil {
		return rawArg, nil
	}
	var event WaitEvent
	if err := json.Unmarshal(eventPayload, &event); err != nil {
		return rawArg, nil
	}
	return rawArg, &event
}
