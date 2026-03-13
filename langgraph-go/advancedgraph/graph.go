package advancedgraph

import (
	"fmt"
)

type NodeFunc func(ctx *Context, arg any) (Command, error)

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

func (g *AdvancedStateGraph) AddNode(name string, fn NodeFunc) {
	g.nodes[name] = fn
}

func (g *AdvancedStateGraph) AddAsyncChannel(name string) {
	g.asyncChannels = append(g.asyncChannels, name)
}

func (g *AdvancedStateGraph) SetEntryPoint(name string) {
	g.entryPoint = name
}

func (g *AdvancedStateGraph) SetFinishPoint(name string) {
	g.finishPoint = name
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
			func(node string, arg any, _ map[string]any) (Command, error) {
				fn, ok := g.nodes[node]
				if !ok {
					return Command{}, fmt.Errorf("unknown node `%s`", node)
				}
				return fn(&Context{engine: engine}, arg)
			},
		)
		handler.done <- resultOrErr{state: state, err: err}
		close(handler.done)
	}()
	return handler, nil
}
