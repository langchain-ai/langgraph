package stategraph

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	ag "github.com/langchain-ai/langgraph/langgraph-go/advancedgraph"
)

type StateNodeFunc[StateT any] func(ctx *Context, state StateT) (StateT, error)

const (
	internalBarrierChannel   = "__stategraph_barrier"
	internalInterruptChannel = "__stategraph_interrupt"
)

type Context struct {
	inner *ag.Context
}

func (c *Context) Interrupt(name string) (any, error) {
	if name == "" {
		return nil, fmt.Errorf("interrupt name cannot be empty")
	}
	event, err := c.inner.WaitFor(ag.AnyOf(ag.ChannelCondition{
		Channel: internalInterruptChannel,
		N:       1,
	}))
	if err != nil {
		if waitReq, ok := ag.AsErrWaitRequested(err); ok {
			return nil, errInterruptRequested{
				Name:      name,
				Condition: waitReq.Condition,
			}
		}
		return nil, err
	}
	if len(event.Value) == 0 {
		return nil, nil
	}

	var payload interruptPayload
	if err := json.Unmarshal(event.Value, &payload); err != nil {
		var value any
		if err := json.Unmarshal(event.Value, &value); err != nil {
			return nil, fmt.Errorf("decode interrupt `%s` value: %w", name, err)
		}
		return value, nil
	}
	if payload.Name != "" && payload.Name != name {
		return nil, fmt.Errorf("interrupt name mismatch: expected `%s`, got `%s`", name, payload.Name)
	}
	if len(payload.Value) == 0 {
		return nil, nil
	}
	var value any
	if err := json.Unmarshal(payload.Value, &value); err != nil {
		return nil, fmt.Errorf("decode interrupt `%s` payload: %w", name, err)
	}
	return value, nil
}

type errInterruptRequested struct {
	Name      string
	Condition ag.AnyOfCondition
}

func (e errInterruptRequested) Error() string {
	if e.Name == "" {
		return "interrupt requested"
	}
	return fmt.Sprintf("interrupt requested: %s", e.Name)
}

func asErrInterruptRequested(err error) (errInterruptRequested, bool) {
	var target errInterruptRequested
	if !errors.As(err, &target) {
		return target, false
	}
	return target, true
}

type BasicStateGraph[StateT any] struct {
	nodes map[string]StateNodeFunc[StateT]
	edges map[string][]string
}

type interruptPayload struct {
	Name  string          `json:"name"`
	Value json.RawMessage `json:"value"`
}

func NewBasicStateGraph[StateT any]() *BasicStateGraph[StateT] {
	return &BasicStateGraph[StateT]{
		nodes: make(map[string]StateNodeFunc[StateT]),
		edges: make(map[string][]string),
	}
}

func (g *BasicStateGraph[StateT]) AddNode(fn StateNodeFunc[StateT]) string {
	name := ag.NodeName(fn)
	if _, exists := g.nodes[name]; exists {
		panic(fmt.Sprintf("node `%s` already exists", name))
	}
	g.nodes[name] = fn
	return name
}

func (g *BasicStateGraph[StateT]) AddEdge(from StateNodeFunc[StateT], to StateNodeFunc[StateT]) {
	fromName := ag.NodeName(from)
	toName := ag.NodeName(to)
	if _, ok := g.nodes[fromName]; !ok {
		panic(fmt.Sprintf("source node `%s` does not exist", fromName))
	}
	if _, ok := g.nodes[toName]; !ok {
		panic(fmt.Sprintf("target node `%s` does not exist", toName))
	}
	g.edges[fromName] = append(g.edges[fromName], toName)
}

type CompiledBasicStateGraph[StateT any] struct {
	inner *ag.CompiledGraph[StateT]
}

type Handler[StateT any] struct {
	inner *ag.Handler[StateT]
}

func (h *Handler[StateT]) WaitForResult() (StateT, error) {
	return h.inner.WaitForResult()
}

func (h *Handler[StateT]) Resume(name string, value any) error {
	if name == "" {
		return fmt.Errorf("interrupt name cannot be empty")
	}
	return h.inner.PublishToChannel(internalInterruptChannel, map[string]any{
		"name":  name,
		"value": value,
	})
}

func (g *BasicStateGraph[StateT]) Compile() *CompiledBasicStateGraph[StateT] {
	if len(g.nodes) == 0 {
		panic("graph has no nodes")
	}
	levels, err := g.computeSupersteps()
	if err != nil {
		panic(err)
	}
	adv := ag.NewAdvancedStateGraph[StateT]()
	adv.AddAsyncChannel(internalBarrierChannel)
	adv.AddAsyncChannel(internalInterruptChannel)

	const finalNodeName = "__stategraph_finish"
	finalNode := func(_ *ag.Context, _ any, state StateT) (ag.Command, error) {
		return ag.Command{Update: state}, nil
	}
	adv.AddFinishNodeAs(finalNodeName, finalNode)

	for stepIdx, stepNodes := range levels {
		for _, nodeName := range stepNodes {
			userFn := g.nodes[nodeName]
			nextBarrier := fmt.Sprintf("__stategraph_barrier_%d", stepIdx+1)
			wrapper := func(ctx *ag.Context, _ any, state StateT) (ag.Command, error) {
				updated, err := userFn(&Context{inner: ctx}, state)
				if err != nil {
					if interruptReq, ok := asErrInterruptRequested(err); ok {
						cond := interruptReq.Condition
						if len(cond.Conditions) == 0 {
							cond = ag.AnyOf(ag.ChannelCondition{
								Channel: internalInterruptChannel,
								N:       1,
							})
						}
						return ag.Command{}, ag.ErrWaitRequested{Condition: cond}
					}
					return ag.Command{}, err
				}
				if err := ctx.PublishToChannel(internalBarrierChannel, map[string]any{
					"step": stepIdx,
				}); err != nil {
					return ag.Command{}, err
				}
				return ag.Command{
					Update: updated,
					Goto:   []ag.Send{{Node: nextBarrier}},
				}, nil
			}
			adv.AddNodeAs(fmt.Sprintf("__stategraph_node_%s", nodeName), wrapper)
		}
	}

	lastBarrier := len(levels)
	for barrierStep := 0; barrierStep <= lastBarrier; barrierStep++ {
		barrierName := fmt.Sprintf("__stategraph_barrier_%d", barrierStep)
		nextStep := barrierStep
		barrier := func(ctx *ag.Context, _ any, state StateT) (ag.Command, error) {
			if nextStep > 0 {
				needed := len(levels[nextStep-1])
				_, err := ctx.WaitFor(ag.AnyOf(ag.ChannelCondition{
					Channel: internalBarrierChannel,
					N:       needed,
				}))
				if err != nil {
					return ag.Command{}, err
				}
			}
			if nextStep >= len(levels) {
				return ag.Command{
					Update: state,
					Goto:   []ag.Send{{Node: finalNodeName}},
				}, nil
			}
			sends := make([]ag.Send, 0, len(levels[nextStep]))
			for _, nodeName := range levels[nextStep] {
				sends = append(sends, ag.Send{
					Node: fmt.Sprintf("__stategraph_node_%s", nodeName),
				})
			}
			return ag.Command{
				Update: state,
				Goto:   sends,
			}, nil
		}
		if barrierStep == 0 {
			adv.AddEntryNodeAs(barrierName, barrier)
		} else {
			adv.AddNodeAs(barrierName, barrier)
		}
	}

	return &CompiledBasicStateGraph[StateT]{
		inner: adv.Compile(),
	}
}

func (g *CompiledBasicStateGraph[StateT]) Start(initialState StateT) (*Handler[StateT], error) {
	raw, err := g.inner.Start(nil, initialState)
	if err != nil {
		return nil, err
	}
	return &Handler[StateT]{inner: raw}, nil
}

func (g *CompiledBasicStateGraph[StateT]) Invoke(initialState StateT) (StateT, error) {
	handler, err := g.Start(initialState)
	if err != nil {
		var zero StateT
		return zero, err
	}
	return handler.WaitForResult()
}

func (g *BasicStateGraph[StateT]) computeSupersteps() ([][]string, error) {
	indegree := make(map[string]int, len(g.nodes))
	for name := range g.nodes {
		indegree[name] = 0
	}
	for from, tos := range g.edges {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("edge source `%s` does not exist", from)
		}
		for _, to := range tos {
			if _, ok := g.nodes[to]; !ok {
				return nil, fmt.Errorf("edge target `%s` does not exist", to)
			}
			indegree[to]++
		}
	}

	queue := make([]string, 0, len(g.nodes))
	level := make(map[string]int, len(g.nodes))
	for name, deg := range indegree {
		if deg == 0 {
			queue = append(queue, name)
		}
	}
	if len(queue) == 0 {
		return nil, fmt.Errorf("graph has no entry nodes (cycle suspected)")
	}

	processed := 0
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		processed++
		currLevel := level[curr]
		for _, to := range g.edges[curr] {
			if level[to] < currLevel+1 {
				level[to] = currLevel + 1
			}
			indegree[to]--
			if indegree[to] == 0 {
				queue = append(queue, to)
			}
		}
	}
	if processed != len(g.nodes) {
		return nil, fmt.Errorf("graph contains a cycle")
	}

	maxLevel := 0
	for _, lv := range level {
		if lv > maxLevel {
			maxLevel = lv
		}
	}
	levels := make([][]string, maxLevel+1)
	for nodeName := range g.nodes {
		lv := level[nodeName]
		levels[lv] = append(levels[lv], nodeName)
	}
	for i := range levels {
		slices.Sort(levels[i])
	}
	return levels, nil
}
