# Evolve/Extend LangGraph with next level of orchestration

## LangGraph Today: A Strong Foundation with Creative Innovation

LangGraph is already an exceptional orchestration framework. It has introduced a number of creative features that no other workflow engine on the market has even attempted.

**First-class streaming.** No workflow engine has ever integrated streaming as seamlessly as LangGraph. Streaming is not an afterthought bolted on top — it is woven into the core execution model, allowing every node, every tool call, and every LLM interaction to emit incremental output naturally.

**Flexible durability modes.** LangGraph defaults to asynchronous execution and supports sync and "exit" modes as well. This is a significant departure from traditional workflow engines, which typically only offer synchronous execution. The ability to choose a durability mode gives developers fine-grained control over the trade-off between persistence guarantees and execution speed.

**Reusable checkpoints.** The checkpoint system allows state to be captured at any point during graph execution and freely replayed, forked, or resumed later. This enables powerful patterns like time-travel debugging, human-in-the-loop approval flows, and long-running conversations that can be picked up exactly where they left off.

**Double texting.** LangGraph natively handles the real-world scenario where a user sends a new message while a previous one is still being processed — a problem most orchestration frameworks simply ignore.

Beyond these innovative features, LangGraph provides solid support for the foundational workflow execution patterns that developers rely on daily. Sequential execution, or loops and conditional branching. Basic parallelism is also well supported: when multiple LLM calls or tool invocations are independent of each other, they can run concurrently to avoid the latency cost of sequential execution, and their results are merged back into the shared state for downstream processing.

LangGraph also offers a simple and intuitive mechanism for human-in-the-loop interactions, allowing a graph to pause execution and wait for user input before continuing.

Combined with the broader LangChain ecosystem, these have made LangGraph a significant success in the market.

## Emerging Gaps: What LangGraph Struggles to Support

As adoption has grown and use cases have become more sophisticated, we have discovered an increasing number of scenarios and design patterns that LangGraph cannot support well today.

**Complex sub-agent coordination.** A main agent often needs to manage multiple sub-agents, but the coordination involved is far more nuanced than simply launching a batch of sub-agents, waiting for all of them to finish, and then moving on. In practice, a main agent may launch a sub-agent, continue doing other work, spawn additional sub-agents later, wait selectively for certain results, retry with a different strategy if one sub-agent fails, or dynamically decide what to do next based on partial results that arrive at unpredictable times.

LangGraph today lacks the coordination primitives to express this. The current parallelism model groups multiple nodes into a single superstep — all of them execute concurrently, but _all_ must complete before the graph can advance to the next step. There is no way for one node to proceed independently while others are still running, and no built-in mechanism for selective waiting, partial result handling, or dynamic task spawning mid-execution.

Sub-agents also cannot simply be modeled as subgraphs, because subgraphs today execute within the same run. They cannot be scaled up independently — if a sub-agent is resource-intensive, there is no straightforward way to run it on a separate machine. Ideally, launching a sub-agent should be(or opt in) as simple as dispatching it for distributed execution across multiple machines.

**Concurrent input and output (e.g. audio agents).** Audio agents also present a particularly clear example of a pattern LangGraph cannot express today. In a voice interaction, speech input and speech output may happen simultaneously — the agent should be able to process a previous utterance, continue receiving new audio input, and produce output all at the same time. These three activities should not be mutually exclusive.

The closest workaround today is double texting, but it has a fundamental flaw: when a new audio input arrives, the previous one is interrupted and canceled rather than being allowed to gracefully complete. The workflow code itself should have the control to decide whether to stop running.

LangGraph is, at its core, a general-purpose workflow engine. Although we focus primarily on agent development, none of the primitives it offers are exclusive to agents or dedicated solely to agentic use cases. Conversely, there is nothing that a general-purpose workflow engine provides that we can safely assume agent development will _never_ need. 

The difference is probably only priority. For example, durable timer where a step can sleep for hours, days or months before resuming. Traditional workflow engines — those built for general microservice orchestration(which doesn't need streaming) -- they may need durable timers. In the agent development world today, most agents are still relatively simple. There are not yet many scenarios that require a step to wait for hours or days before proceeding.


## Deriving What's Needed from First Principles

Before jumping to solutions, it is worth stepping back and asking a fundamental question: what is an orchestration engine, and what do users expect it to provide?

At its most fundamental level, a workflow engine's value proposition is making a long-running process execute reliably. If a machine crashes, execution should smoothly fail over to another machine and resume from the last point where it was interrupted — not start over from the beginning. So we can reason about what is needed by asking: what would a developer do if they had to build a long-running process _without_ a workflow engine?

Starting from the simple. A developer could write a simple `main` function — a single-threaded program, just like everyone writes when they first learn to code. It would have `if/else` branches, `for` loops, and maybe it would wait for command-line input. Many early agent use cases look exactly like this: execute a sequence of steps, make decisions along the way, loop when necessary.

But if that machine crashes, you probably do not want the process to start over from scratch. You want it to resume from the last step that completed successfully. And if a step fails, you might want it to retry automatically before giving up.

LangGraph handles this case very well. 

There is an important constraint worth calling out explicitly: LangGraph requires the developer to organize their code into **nodes**, which serve as the boundaries at which checkpoints can be taken. This is a constraint shared by every workflow engine — it is simply not feasible to persist a checkpoint after every single line of arbitrary code.

### From Single-Threaded to Concurrent: Where the Model Breaks Down

But as product requirements grow more complex, a single-threaded program is no longer sufficient. The process becomes multi-threaded or multi-process. And in a multi-threaded program, each thread executes independently — when one thread finishes a step and moves on to its next step, it does not need to wait for another thread to finish _its_ current step first.

This is precisely why LangGraph's superstep restriction feels awkward in practice. In the superstep model, all concurrently executing nodes must complete before any of them can advance. But that is not how independent threads work. Each thread should be able to progress at its own pace, checkpoint its own state, and move to its next step without being blocked by unrelated work happening in parallel.

Multiple threads and processes do, however, need to coordinate with each other. In concurrent programming, channels are an essential primitive precisely because they provide a safe, structured way for threads to communicate and synchronize without relying on shared mutable memory — avoiding data races and deadlocks. In some cases, threads may use locking for coordination, but the preferred approach is message passing through channels.

NOTE: "channel" is overloaded term here as it's also an internal term within current LangGraph pregel algorithm. 

LangGraph already has a mechanism that is closely related: `interrupt`. A run can be interrupted, and then another run can resume it. If we look at this through the lens of channels, `interrupt` is essentially a **channel with size 0** — a synchronous rendezvous point where one side blocks until the other side is ready. 

The natural extension:

1. **Variable-size channels.** The channel buffer size should be configurable — size 0 for synchronous handoff (like `interrupt` today), size N for buffered communication where the sender can proceed without waiting, and unbounded for fully asynchronous fire-and-forget messaging.
2. **Channels across boundaries.** Channels should not be limited to communication between separate runs. Nodes within the same graph should also be able to send and receive through channels — mirroring the way both multi-process communication (between runs) and multi-thread communication (between nodes within a run) work in ordinary concurrent programs.
3. **Node-level blocking, not run-level pausing.** When a node waits on a channel (i.e. `interrupt`), only that node should block — the rest of the graph should continue executing. Today, `interrupt` pauses the entire run. In a concurrent program, when one thread blocks on a channel read, the other threads keep running. The same should be true: an interrupt should suspend the individual node, not halt the whole run.

## Summmary of all extension opportunity

### P1: urgently needed
#### Remove the Superstep Restriction

Today, when multiple nodes execute in parallel, they are grouped into a superstep. All nodes in a superstep must complete before any downstream node can begin. This means that even if `b1` finishes quickly and its successor `b11` is ready to run, it must wait for `b2` to finish first.

With the superstep restriction removed, each parallel branch progresses independently. As soon as a node completes, its downstream successor can begin immediately — regardless of what is happening in other branches.

**Current behavior (superstep model):**

```
Step 1:  a
Step 2:  b1, b2        ← both must finish before step 3
Step 3:  b11, b22      ← both start together
```

Even if `b1` finishes in 1 second and `b2` takes 30 seconds, `b11` cannot start until `b2` is done.

**Proposed behavior (independent branches):**

```
Branch 1:  a → b1 → b11 → ...
Branch 2:  a → b2 → b22 → ...
```

Each branch advances at its own pace. `b1` finishing triggers `b11` immediately, without waiting for `b2`.


No API change is needed from the user's perspective — the graph definition stays the same. The change is in the execution semantics: the engine no longer forces all parallel nodes to synchronize at each step boundary. Each branch is checkpointed independently, so if `b1 → b11` completes while `b2` is still running, `b11`'s result is already persisted.

This is necessary for the next one -- Light-weight Interrupt: Only Block the Current Node. Because we want to let other nodes continue to run while a node is waiting on something.

#### Light-weight Interrupt -- wait_for API: Only Block the Current Node Until Channel Has Enough Messages

Today, `interrupt` pauses the entire run. Every node stops, and nothing can proceed until the interrupt is resolved externally. This is the right behavior for a simple single-threaded workflow, but it breaks down when multiple branches are executing concurrently — one branch needing input should not freeze all the others.

The proposed change has three parts:

1. **Named channels.** A graph can declare named channels as coordination points. These are distinct from the graph's state — they are message-passing primitives, not shared memory.
2. **`wait_for` blocks only the current node.** When a node calls `wait_for`, it suspends itself and waits for messages on the specified channel. All other nodes in the graph continue executing normally.
3. A channel can be published from both external and internal

The `wait_for` call takes a channel name and optionally a count `N`, meaning "wait until N messages have arrived on this channel before resuming."

**Prototype:**

See [test_sub_agents.py](../libs/langgraph/tests/advanced-graph/test_sub_agents.py)


### P2: likely needed
#### subGraph redesign
#### durable timers
#### more flexiable waiting conditions on interrupts
#### locking on state fields


### P3: future needed or nice to have
#### RPC