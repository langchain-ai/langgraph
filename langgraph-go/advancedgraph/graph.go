package advancedgraph

import (
	"fmt"
	"reflect"
	"runtime"
	"strings"
)

type NodeFunc func(ctx *Context, input any, state map[string]any) (Command, error)

type AdvancedStateGraph struct {
	nodes         map[string]NodeFunc
	asyncChannels []string
	entryPoint    string
	finishPoint   string
}

func NewAdvancedStateGraph() *AdvancedStateGraph {
	return &AdvancedStateGraph{
		nodes: make(map[string]NodeFunc),
	}
}

func (g *AdvancedStateGraph) AddNode(fn NodeFunc) string {
	name := NodeName(fn)
	if _, exists := g.nodes[name]; exists {
		panic(fmt.Sprintf("node `%s` already exists", name))
	}
	g.nodes[name] = fn
	return name
}

func (g *AdvancedStateGraph) AddAsyncChannel(name string) {
	g.asyncChannels = append(g.asyncChannels, name)
}

func (g *AdvancedStateGraph) SetEntryNode(fn NodeFunc) {
	g.entryPoint = NodeName(fn)
}

func (g *AdvancedStateGraph) SetFinishNode(fn NodeFunc) {
	g.finishPoint = NodeName(fn)
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
	nodes         map[string]NodeFunc
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

func (g *CompiledGraph) Start(initialState map[string]any) (*Handler, error) {
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

func NodeName(fn NodeFunc) string {
	pc := reflect.ValueOf(fn).Pointer()
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
