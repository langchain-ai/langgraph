# Multi-agent Systems

An [agent](./agentic_concepts.md#agent-architectures) is *a system that uses an LLM to decide the control flow of an application*. As you develop these systems, they might grow more complex over time, making them harder to manage and scale. For example, you might run into the following problems:

* agent uses too many tools and makes poor decisions about which tool to call next
* context grows too complex for a single agent to keep track of
* there is a need for multiple specialization areas in the system (e.g. planner, researcher, math expert, etc.)

To tackle these, you might consider breaking your application into multiple smaller, independent agents and composing them into a **multi-agent system**. These independent agents can be as simple as a prompt and an LLM call, or as complex as a [ReAct](./agentic_concepts.md#react-implementation) agent.

The primary benefits of using multi-agent systems are:

* **Modularity**: Separate agents make it easier to develop, test, and maintain agentic systems.
* **Specialization**: You can create expert agents focused on specific domains, which helps with the overall system performance.
* **Control**: You can explicitly control how agents communicate (as opposed to relying on function calling).

## Multi-agent architectures

![](./img/multi_agent/architectures.png)

There are several ways to connect agents in a multi-agent system:

* **Unconstrained Mult-Agent** connections: each agent can communicate with any other agent. Any agent can decide which other agent to call next. This architecture is very flexible but becomes inefficient and harder to scale as the number of agents grows.
* **Multi-Agent Workflow** connections: each agent communicates with only a subset of agents. Parts of the flow are deterministic, and only some agents can decide which other agents to call next.
* **Supervisor**, or **one-to-many** connections: each agent communicates with a single [supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/) agent. Supervisor agent makes decisions on which agent should be called next.
* **Single tool-calling agent**: this is a special case of supervisor architecture. Individual agents can be represented as tools. In this case, a supervisor agent uses a tool-calling LLM to decide which of the agent tools to call, as well as the arguments to pass to those agents.

### Unconstrained many-to-many

### Multi-Agent Workflow

LangGraph provides multiple methods to control agent communication sequence:

* **Explicit control flow (graph edges)**: LangGraph allows you to explicitly define the control flow of your application (i.e. the sequence of how agents communicate) explicitly, via [graph edges](./low_level.md#edges). This is the most deterministic variant of the **constrained** multi-agent architecture above -- we always know which agent will be called next ahead of time.

* **Dynamic control flow (conditional edges)**: in LangGraph you can allow LLMs to decide parts of your application control flow. This can be achieved by using [conditional edges](./low_level.md#conditional-edges). A special case of this is a single tool-calling agent that calls [agents as tools](#agents-as-tools). In this case, the tool-calling LLM powering the supervisor agent will make decisions about the order in which the tools (agents) are being called.

### Supervisor

### Supervisor (Tool Calling)


## Communication between agents

The most important thing is to figure out how agents communicate.
There are few different considerations for how they communcicate

**Via tool calls or via state**
...

**What if two agents have different States?**
...

**Communicating on a shared message list**


### Via tool calls or via state**
![](./img/multi_agent/request.png)

### What if two agents have different States?

### Communicating on a shared message list

![](./img/multi_agent/response.png)


#### Share full history

Agents can **share the full history** of their thought process (i.e. "scratchpad") with all other agents. This "scratchpad" would typically look like a [list of messages](./low_level.md#why-use-messages). The benefit of sharing full thought process is that it might help other agents make better decisions and improve reasoning ability for the system as a whole. The downside is that as the number of agents and their complexity grows, the "scratchpad" will grow quickly and might require additional strategies for [memory management](./memory.md/#managing-long-conversation-history).

#### Share final result

Agents can have their own private "scratchpad" and only **share the final result** with the rest of the agents. This approach might work better for systems with many agents or agents that are more complex. There are several ways to achieve this in LangGraph:

* An agent node can have a [private input state schema](https://langchain-ai.github.io/langgraph/how-tos/pass_private_state/) that is distinct from the overall graph state schema. This allows passing additional information during the graph execution that is only needed for executing a particular node.
* [Subgraph](./low_level.md#subgraphs) agents can have independent [input / output state schemas](https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/). In this case it’s important to [add input / output transformations](https://langchain-ai.github.io/langgraph/how-tos/subgraph-transform-state/) so that the parent graph knows how to communicate with the subgraphs.

For agents called as tools, the supervisor determines the inputs based on the tool schema. Additionally, LangGraph allows [passing state](https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#pass-graph-state-to-tools) to individual tools at runtime, so subordinate agents can access parent state, if needed.
