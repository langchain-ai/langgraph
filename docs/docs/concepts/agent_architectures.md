# Agent architectures

Many LLM applications implement a particular control flow of steps before and / or after LLM calls. As an example, [RAG](https://github.com/langchain-ai/rag-from-scratch) performs retrieval of relevant documents to a question, and passes those documents to an LLM in order to ground the model's response. 

Instead of hard-coding a fixed control flow, we sometimes want LLM systems that can pick its own control flow to solve more complex problems! This is one definition of an [agent](https://blog.langchain.dev/what-is-an-agent/): *an agent is a system that uses an LLM to decide the control flow of an application.* There are many ways that an LLM can control application:

- An LLM can route between two potential paths
- An LLM can decide which of many tools to call
- An LLM can decide whether the generated answer is sufficient or more work is needed

As a result, there are many different types of [agent architectures](https://blog.langchain.dev/what-is-a-cognitive-architecture/), which given an LLM varying levels of control. 

![Agent Types](img/agent_types.png)

## Router

A router allows an LLM to select a single step from a specified set of options. This is an agent architecture that exhibits a relatively limited level of control because the LLM usually governs a single decision and can return a narrow set of outputs. 

A key part of routing is returning a response that can parsed to one of several paths to take. This is largely equivalent to classification. Oftentimes [structured outputs](https://python.langchain.com/docs/how_to/structured_output/) are used here to structure the LLM's response so that it can be reliably interpreted.

## ReAct

While a router allows an LLM to make a single decision, more complex agent architectures expand the LLM's control in two key ways:

1. Multi-step decision making: The LLM can control a sequence of decisions rather than just one.
2. Tool access: The LLM can choose from and use a variety of tools to accomplish tasks, and decide not only which one to use but also what the inputs to the tool may be.

[ReAct](https://arxiv.org/abs/2210.03629) is a popular general purpose agent architecture that gives the LLM a LOT of control.
The basic `ReAct` architecture is a relatively simple one that in pseudocode looks something like:

```python
finished = False
steps = []
inputs = ...
while not finished:
    prompt = ...  # some combination of `inputs` and `steps`
    model_prediction = llm.invoke(prompt)  # call the LLM with the formatted prompt
    # check if the model prediction has any tool calls
    if len(model_prediction.tool_calls) > 0:
        observation = ... # execute the tools
        steps += [(model_prediction, observation)]  # add to the list of steps taken
    else:
        # if no tool calls, then agent is finished, and return
        return model_prediction
```

This architecture allows for more complex and flexible agent behaviors, going beyond simple routing to enable dynamic problem-solving across multiple steps. LangGraph contains an implementation of this architecture - you can use it with [`create_react_agent`](../reference/prebuilt.md#create_react_agent).

This architecture heavily uses [tool calling](agentic_concepts.md#tool-calling) to determine which actions to take. Having a model that can do tool calling reliably is important for this architecture.


### Parameters

There are several important parameters that you should consider when building a ReAct agent.

**Model**

What model are you using? It needs to be intelligent enough to do tool calling reliably. While simpler agent architectures can maybe get away with a small model if they are just doing routing, a "good" model is needed.

**System Prompt**

This configures the models behavior. You can set instructions for how the model should behave, how it should call tools, how it should respond, etc.

**Tools**

A key part of a ReAct agent is which tools you give it. These define enable the agent to actually accomplish goals beyound what a standalone LLM could do. Key considerations here are:

- How many tools are you giving it? If too many, may get overwhelmed.
- How should the agent use these tools? It needs to know that somehow. Can either be explained in the system prompt or in the description of the tools.
- What does the tool return? If too much, or in an unweildy shape, it may overwhelm the agent. Should generally a more condensed & constrained data object.

It is important to remember here that tool descriptions (as well as tool argument descriptions) should be thought of as part of the prompt - these greatly affect when and how the agent calls a tool!

## Plan-and-execute

One downside of the ReAct agent is that is does planning at each step. That is - at each step it decides what action to take. That is fine for tasks that only require a single or small amount of tool calls. If there is a "multi-hop" task that requires multiple tool calls or actions in sequence, it is a well observed phenomenom that the LLM degrades in performance the further into the sequence it gets.

For an example of a "multi-hop" task, consider something like: "Find the stars in the TV show Shogun, get their current age, add it all up, and then find the king of England in that year". This is a task that will require multiple tool calls:

1. Finding the cast of Shogun (via web search)
2. For each cast member, find their current age (via web search)
3. Add their ages together (using a calculator)
4. Find the king of England in that year (via web search)

As humans, one thing we naturally do when faced with a complicated problem is break it down into a plan - as we did above! We then go about executing on that plan. We separate the act of planning and the act of execution of individual steps. We can do the same with language models.

Pseudocode for these types of architectures generally look something like:

```python
plan = planning_agent.invoke(inputs)
steps = []
for step in plan:
    steps.append(execution_agent.invoke(inputs, step))
return steps[-1]
```

LangGraph does not provide an off-the-shelf implementation of this, because it is a bit tricky to make this properly generic. We do provide [several example implementations](../tutorials.md#planning-agents) that use this concept.

There are several papers that explore this idea of planning:

- An LLM Compiler for Parallel Function Calling
    - [Paper](https://arxiv.org/abs/2312.04511)
    - [LangGraph Implementation](../tutorials/llm-compiler/LLMCompiler/)
- ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models
    - [Paper](https://arxiv.org/abs/2305.18323)
    - [LangGraph Implementation](../tutorials/rewoo/rewoo/)
- Plan-and-Solve Prompting
    - [Paper](https://arxiv.org/abs/2305.04091)
    - [LangGraph Implementation](../tutorials/plan-and-execute/plan-and-execute/)

In practice, we don't see developers using any one of these implementations strictly as is, but rather taking this idea of planning and incorporating it into their custom cognitive architecture.

### Reflection

While planning can help come with a plan at the start, an equally important process that we do as humans is review our work to determine if it is correct. This same concept can be applied to agents, and is generally called "reflection".

The key question here is what process to use to reflect on work done. The answer to this is pretty domain specfic. Some common ones include:

- Using another LLM to judge the work done (useful for most domains, but needs a custom prompt)
- Using a code environment to lint/run/test code (useful for coding agents)
- Using a heurestic-based (non-LLM) process to flag toxic words (useful for guardrails)

Let's expand a bit on using another LLM to judge the output, since this is one of the most common methods. In this process, you generally have three separate "agents" (even if all those agents are is a prompt and an LLM). These are:

- Agent A: this agent generates the initial response. It has instructions for taking an initial user input and generating an answer.
- Agent B: this agent critiques the initial response from Agent A. It has instructions for taking the initial user input, a generated answer, and then deciding whether it is good enough or not. It often uses tool calling to communicate whether it is good enough or not. If it is decided to be good enough, it will then usually return to the user.
- Agent C: this agent refines the response based on the critique from Agent B. It takes in the initial user query, the generated answer, and a critique of that answer. It then returns a new answer.

The overall agent system then goes between Agent B and Agent C until a stopping criteria is met (Agent B approves the response, some maximum number of steps is reached, etc).

Note that sometimes Agent A and Agent C are the same agent.

Pseudocode for this can look like:

```python
response = agent_a.invoke(inputs)
critique = agent_b.invoke(inputs, response)
while not_ok(critique):
    response = agent_c.invoke(inputs, response, critique)
    critique = agent_b.invoke(inputs, response)
return response
```

LangGraph does not provide an off-the-shelf implementation of this, because it is a bit tricky to make this properly generic. We do provide [several example implementations](../tutorials.md#reflection-critique) that use this concept.

There are several papers that explore this idea of reflection:

- Reflexion
    - [Paper](https://arxiv.org/abs/2303.11366)
    - [LangGraph Implementation](../tutorials/reflexion/reflexion/)
- Language Agent Tree Search
    - [Paper](https://arxiv.org/abs/2310.04406)
    - [LangGraph Implementation](../tutorials/lats/lats/)

In practice, we don't see developers using any one of these implementations strictly as is, but rather taking this idea of planning and incorporating it into their custom cognitive architecture.


## Custom agent architectures

While routers and tool-calling agents (like ReAct) are common, [customizing agent architectures](https://blog.langchain.dev/why-you-should-outsource-your-agentic-infrastructure-but-own-your-cognitive-architecture/) often leads to better performance for specific tasks. 

We see users taking the ideas of the architectures above (routing, tool calling, planning, reflection) and creating their own custom "cognitive architecture" that works best for their use case. That is the power of LangGraph!


