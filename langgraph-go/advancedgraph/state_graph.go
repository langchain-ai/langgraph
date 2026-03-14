package advancedgraph

import (
	"fmt"
	"slices"
	"sync"
)

type StateNodeFunc[StateT any] func(ctx *Context, state StateT) (StateT, error)

type BasicStateGraph[StateT any] struct {
	nodes            map[string]StateNodeFunc[StateT]
	edges            map[string][]string
	interruptChannel string
	interruptSteps   map[int]struct{}
}

func NewBasicStateGraph[StateT any]() *BasicStateGraph[StateT] {
	return &BasicStateGraph[StateT]{
		nodes:          make(map[string]StateNodeFunc[StateT]),
		edges:          make(map[string][]string),
		interruptSteps: make(map[int]struct{}),
	}
}

func (g *BasicStateGraph[StateT]) AddNode(name string, fn StateNodeFunc[StateT]) {
	if name == "" {
		panic("node name cannot be empty")
	}
	if _, exists := g.nodes[name]; exists {
		panic(fmt.Sprintf("node `%s` already exists", name))
	}
	g.nodes[name] = fn
}

func (g *BasicStateGraph[StateT]) AddEdge(from string, to string) {
	if _, ok := g.nodes[from]; !ok {
		panic(fmt.Sprintf("source node `%s` does not exist", from))
	}
	if _, ok := g.nodes[to]; !ok {
		panic(fmt.Sprintf("target node `%s` does not exist", to))
	}
	g.edges[from] = append(g.edges[from], to)
}

// EnableInterruptOnSuperstep enables pause/resume before dispatching a superstep.
// superstep=1 means "after first superstep has completed, before second starts".
func (g *BasicStateGraph[StateT]) EnableInterruptOnSuperstep(superstep int, channel string) {
	if superstep <= 0 {
		panic("interrupt superstep must be >= 1")
	}
	if channel == "" {
		panic("interrupt channel cannot be empty")
	}
	if g.interruptChannel != "" && g.interruptChannel != channel {
		panic("all interrupts must use the same channel")
	}
	g.interruptChannel = channel
	g.interruptSteps[superstep] = struct{}{}
}

type CompiledBasicStateGraph[StateT any] struct {
	inner *CompiledGraph[StateT]
}

func (g *BasicStateGraph[StateT]) Compile() *CompiledBasicStateGraph[StateT] {
	if len(g.nodes) == 0 {
		panic("graph has no nodes")
	}
	levels, err := g.computeSupersteps()
	if err != nil {
		panic(err)
	}
	adv := NewAdvancedStateGraph[StateT]()

	const finalNodeName = "__stategraph_finish"
	finalNode := func(_ *Context, _ any, state StateT) (Command, error) {
		return Command{Update: state}, nil
	}
	adv.AddFinishNodeAs(finalNodeName, finalNode)

	if g.interruptChannel != "" {
		adv.AddAsyncChannel(g.interruptChannel)
	}

	for stepIdx, stepNodes := range levels {
		for _, nodeName := range stepNodes {
			userFn := g.nodes[nodeName]
			nextBarrier := fmt.Sprintf("__stategraph_barrier_%d", stepIdx+1)
			wrapper := func(ctx *Context, _ any, state StateT) (Command, error) {
				updated, err := userFn(ctx, state)
				if err != nil {
					return Command{}, err
				}
				return Command{
					Update: updated,
					Goto:   []Send{{Node: nextBarrier}},
				}, nil
			}
			adv.AddNodeAs(fmt.Sprintf("__stategraph_node_%s", nodeName), wrapper)
		}
	}

	type barrierCounter struct {
		mu     sync.Mutex
		counts map[*RustEngine]int
	}
	counters := make(map[int]*barrierCounter)
	for barrierStep := 1; barrierStep <= len(levels); barrierStep++ {
		counters[barrierStep] = &barrierCounter{
			counts: make(map[*RustEngine]int),
		}
	}

	lastBarrier := len(levels)
	for barrierStep := 0; barrierStep <= lastBarrier; barrierStep++ {
		barrierName := fmt.Sprintf("__stategraph_barrier_%d", barrierStep)
		nextStep := barrierStep
		barrier := func(ctx *Context, _ any, state StateT) (Command, error) {
			if nextStep > 0 {
				counter := counters[nextStep]
				counter.mu.Lock()
				counter.counts[ctx.engine]++
				current := counter.counts[ctx.engine]
				needed := len(levels[nextStep-1])
				if current < needed {
					counter.mu.Unlock()
					return Command{Update: state}, nil
				}
				delete(counter.counts, ctx.engine)
				counter.mu.Unlock()
			}
			if _, needsInterrupt := g.interruptSteps[nextStep]; needsInterrupt {
				cond := AnyOf(ChannelCondition{Channel: g.interruptChannel, N: 1})
				if _, err := ctx.WaitFor(cond); err != nil {
					return Command{}, err
				}
			}
			if nextStep >= len(levels) {
				return Command{
					Update: state,
					Goto:   []Send{{Node: finalNodeName}},
				}, nil
			}

			sends := make([]Send, 0, len(levels[nextStep]))
			for _, nodeName := range levels[nextStep] {
				sends = append(sends, Send{
					Node: fmt.Sprintf("__stategraph_node_%s", nodeName),
				})
			}
			return Command{
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
	return g.inner.Start(nil, initialState)
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
