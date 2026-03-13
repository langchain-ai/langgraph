package advancedgraph

import (
	"fmt"
	"reflect"
	"runtime"
	"strings"
)

type NodeFunc[IN any] func(ctx *Context, input IN, state map[string]any) (Command, error)

type nodeExecutor func(ctx *Context, input any, state map[string]any) (Command, error)

type AdvancedStateGraph struct {
	nodes         map[string]nodeExecutor
	asyncChannels []string
	entryPoint    string
	finishPoint   string
}

func NewAdvancedStateGraph() *AdvancedStateGraph {
	return &AdvancedStateGraph{
		nodes: make(map[string]nodeExecutor),
	}
}

func (g *AdvancedStateGraph) AddNode(fn any) string {
	name := NodeName(fn)
	if _, exists := g.nodes[name]; exists {
		panic(fmt.Sprintf("node `%s` already exists", name))
	}
	exec, err := compileNodeExecutor(fn)
	if err != nil {
		panic(err)
	}
	g.nodes[name] = exec
	return name
}

func (g *AdvancedStateGraph) AddAsyncChannel(name string) {
	g.asyncChannels = append(g.asyncChannels, name)
}

func (g *AdvancedStateGraph) SetEntryNode(fn any) {
	g.entryPoint = NodeName(fn)
}

func (g *AdvancedStateGraph) SetFinishNode(fn any) {
	g.finishPoint = NodeName(fn)
}

func (g *AdvancedStateGraph) AddEntryNode(fn any) string {
	name := g.AddNode(fn)
	g.entryPoint = name
	return name
}

func (g *AdvancedStateGraph) AddFinishNode(fn any) string {
	name := g.AddNode(fn)
	g.finishPoint = name
	return name
}

func (g *AdvancedStateGraph) Compile() *CompiledGraph {
	return &CompiledGraph{
		nodes:         g.nodes,
		asyncChannels: g.asyncChannels,
		entryPoint:    g.entryPoint,
		finishPoint:   g.finishPoint,
	}
}

type CompiledGraph struct {
	nodes         map[string]nodeExecutor
	asyncChannels []string
	entryPoint    string
	finishPoint   string
}

type Context struct {
	engine *RustEngine
}

func (c *Context) WaitFor(cond AnyOfCondition) (WaitEvent, error) {
	return c.engine.WaitAnyOf(cond)
}

func (c *Context) PublishToChannel(channel string, value any) error {
	return c.engine.Publish(channel, value)
}

type Handler struct {
	engine *RustEngine
	done   chan resultOrErr
}

type resultOrErr struct {
	state map[string]any
	err   error
}

func (h *Handler) PublishToChannel(channel string, value any) error {
	return h.engine.Publish(channel, value)
}

func (h *Handler) WaitForResult() (map[string]any, error) {
	res := <-h.done
	return res.state, res.err
}

func (g *CompiledGraph) Start(initialState map[string]any, initialInput any) (*Handler, error) {
	engine := NewRustEngine()
	for _, ch := range g.asyncChannels {
		if err := engine.AddAsyncChannel(ch); err != nil {
			return nil, err
		}
	}

	handler := &Handler{
		engine: engine,
		done:   make(chan resultOrErr, 1),
	}
	go func() {
		defer engine.Close()
		state, err := engine.RunGraph(
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
				return fn(&Context{engine: engine}, nodeInput, fallbackState)
			},
		)
		handler.done <- resultOrErr{state: state, err: err}
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

func compileNodeExecutor(fn any) (nodeExecutor, error) {
	rv := reflect.ValueOf(fn)
	if !rv.IsValid() || rv.Kind() != reflect.Func {
		return nil, fmt.Errorf("node must be a function")
	}
	rt := rv.Type()
	if rt.NumIn() != 3 {
		return nil, fmt.Errorf("node `%s` must accept exactly 3 args: (*Context, input, map[string]any)", NodeName(fn))
	}
	ctxType := reflect.TypeOf((*Context)(nil))
	if rt.In(0) != ctxType {
		return nil, fmt.Errorf("node `%s` first arg must be *Context", NodeName(fn))
	}
	stateType := reflect.TypeOf(map[string]any{})
	if rt.In(2) != stateType {
		return nil, fmt.Errorf("node `%s` third arg must be map[string]any", NodeName(fn))
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
	return func(ctx *Context, input any, state map[string]any) (Command, error) {
		args := []reflect.Value{
			reflect.ValueOf(ctx),
			reflect.Zero(inputType),
			reflect.ValueOf(state),
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
		if out[1].IsNil() {
			return cmd, nil
		}
		return cmd, out[1].Interface().(error)
	}, nil
}
