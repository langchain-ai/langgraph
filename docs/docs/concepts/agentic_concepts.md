# Common Agentic Patterns

## Structured Output

It's pretty common to want LLMs inside nodes to return structured output when building agents. This is because that structured output can often be used to route to the next step (e.g. choose between two different edges) or update specific keys of the state.

Since LangGraph nodes can be arbitrary Python functions, you can do this however you want. If you want to use LangChain, [this how-to guide](https://python.langchain.com/v0.2/docs/how_to/structured_output/) is a starting point.

## Tool calling

It's extremely common to want agents to do tool calling. Tool calling refers to choosing from several available tools, and specifying which ones to call and what the inputs should be. This is extremely common in agents, as you often want to let the LLM decide which tools to call and then call those tools.

Since LangGraph nodes can be arbitrary Python functions, you can do this however you want. If you want to use LangChain, [this how-to guide](https://python.langchain.com/v0.2/docs/how_to/tool_calling/) is a starting point.

## Memory

Memory is a key concept to agentic applications. Memory is important because end users often expect the application they are interacting with remember previous interactions. The most simple example of this is chatbots - they clearly need to remember previous messages in a conversation.

LangGraph is perfectly suited to give you full control over the memory of your application. With user defined [`State`](#state) you can specify the exact schema of the memory you want to retain. With [checkpointers](#checkpointer) you can store checkpoints of previous interactions and resume from there in follow up interactions.

## Human-in-the-loop

Agentic systems often require some human-in-the-loop (or "on-the-loop") interaction patterns. This is because agentic systems are still not super reliable, so having a human involved is required for any sensitive tasks/actions. These are all easily enabled in LangGraph, largely due to [checkpointers](#checkpointer). The reason a checkpointer is necessary is that a lot of these interaction patterns involve running a graph up until a certain point, waiting for some sort of human feedback, and then continuing. When you want to "continue" you will need to access the state of the graph previous to getting interrupted, and checkpointers are a built in, highly convenient way to do that.

There are a few common human-in-the-loop interaction patterns we see emerging.

### Approval

A basic one is to have the agent wait for approval before executing certain tools. This may be all tools, or just a subset of tools. This is generally recommend for more sensitive actions (like writing to a database). This can easily be done in LangGraph by setting a [breakpoint](#breakpoints) before specific nodes.

### Wait for input

A similar one is to have the agent wait for human input. This can be done by:

1. Create a node specifically for human input
2. Add a breakpoint before the node
3. Get user input
4. Update the state with that user input, acting as that node
5. Resume execution

### Edit agent actions

This is a more advanced interaction pattern. In this interaction pattern the human can actually edit some of the agent's previous decisions. This can be done either during the flow (after a [breakpoint](#breakpoints), part of the [approval](#approval) flow) or after the fact (as part of [time-travel](#time-travel))

### Time travel

This is a pretty advanced interaction pattern. In this interaction pattern, the human can look back at the list of previous checkpoints, find one they like, optionally [edit it](#edit-agent-actions), and then resume execution from there.

## Map-Reduce

A common pattern in agents is to generate a list of objects, do some work on each of those objects, and then combine the results. This is very similar to the common [map-reduce](https://en.wikipedia.org/wiki/MapReduce) operation. This can be tricky for a few reasons. First, it can be tough to define a structured graph ahead of time because the length of the list of objects may be unknown. Second, in order to do this map-reduce you need multiple versions of the state to exist... but the graph shares a common shared state, so how can this be?

LangGraph supports this via the [Send](#send) api. This can be used to allow a conditional edge to Send multiple different states to multiple nodes. The state it sends can be different from the state of the core graph.

See a how-to guide for this [here](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)

## Multi-agent

A term you may have heard is "multi-agent" architectures. What exactly does this mean?

Given that it is hard to even define an "agent", it's almost impossible to exactly define a "multi-agent" architecture. When most people talk about a multi-agent architecture, they typically mean a system where there are multiple different LLM-based systems. These LLM-based systems can be as simple as a prompt and an LLM call, or as complex as a [ReAct agent](#react-agent).

The big question in multi-agent systems is how they communicate. This involves both the schema of how they communicate, as well as the sequence in which they communicate. LangGraph is perfect for orchestrating these types of systems. It allows you to define multiple agents (each one is a node) an arbitrary state (to encapsulate the schema of how they communicate) as well as the edges (to control the sequence in which they communicate).

## Planning

One of the big things that agentic systems struggle with is long term planning. A common technique to overcome this is to have an explicit planning this. This generally involves calling an LLM to come up with a series of steps to execute. From there, the system then tries to execute the series of tasks (this could use a sub-agent to do so). Optionally, you can revisit the plan after each step and update it if needed.

## Reflection

Agents often struggle to produce reliable results. Therefore, it can be helpful to check whether the agent has completed a task correctly or not. If it has - then you can finish. If it hasn't - then you can take the feedback on why it's not correct and pass it back into another iteration of the agent.

This "reflection" step often uses an LLM, but doesn't have to. A good example of where using an LLM may not be necessary is in coding, when you can try to compile the generated code and use any errors as the feedback.

## ReAct Agent

One of the most common agent architectures is what is commonly called the ReAct agent architecture. In this architecture, an LLM is called repeatedly in a while-loop. At each step the agent decides which tools to call, and what the inputs to those tools should be. Those tools are then executed, and the outputs are fed back into the LLM as observations. The while-loop terminates when the agent decides it is not worth calling any more tools.

One of the few high level, pre-built agents we have in LangGraph - you can use it with [`create_react_agent`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent)

This is named after and based on the [ReAct](https://arxiv.org/abs/2210.03629) paper. However, there are several differences between this paper and our implementation:

- First, we use [tool-calling](#tool-calling) to have LLMs call tools, whereas the paper used prompting + parsing of raw output. This is because tool calling did not exist when the paper was written, but is generally better and more reliable.
- Second, we use messages to prompt the LLM, whereas the paper used string formatting. This is because at the time of writing, LLMs didn't even expose a message-based interface, whereas now that's the only interface they expose.
- Third, the paper required all inputs to the tools to be a single string. This was largely due to LLMs not being super capable at the time, and only really being able to generate a single input. Our implementation allows for using tools that require multiple inputs.
- Forth, the paper only looks at calling a single tool at the time, largely due to limitations in LLMs performance at the time. Our implementation allows for calling multiple tools at a time.
- Finally, the paper asked the LLM to explicitly generate a "Thought" step before deciding which tools to call. This is the "Reasoning" part of "ReAct". Our implementation does not do this by default, largely because LLMs have gotten much better and that is not as necessary. Of course, if you wish to prompt it do so, you certainly can.
