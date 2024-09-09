# What is a multi-agent architecture?

A multi-agent architecture is a system with multiple LLM-based components. These components can be as simple as a prompt and an LLM call, or as complex as a ReAct agent.

The primary advantages of this architecture are:

* **Modularity**: Separate agents facilitate easier development, testing, and maintenance of agentic systems.
* **Specialization**: Developers can create expert agents focused on specific domains, and compose them into more complex applications

## Types of multi-agent architectures in LangGraph

### Agents as nodes

Agents can be defined as nodes in LangGraph. As any other node in the LangGraph, these agent nodes receive the graph state as an input and return an update to the state as their output.

* Simple LLM nodes: single LLMs with custom prompts
* Subgraph nodes: complex graphs called inside the orchestrator graph node

### Agents as tools

Agents can also be defined as tools. In this case, the orchestrator agent (e.g. ReAct agent) would use a tool-calling LLM to decide which of the agent tools to call.

You could also take a "mega-graph" approach – incorporating subordinate agents' nodes directly into the parent, orchestrator graph. However, this is not recommended for complex subordinate agents, as it would make the overall system harder to scale, maintain and debug – you should use subgraphs or tools in those cases.

## Communication in multi-agent systems

A big question in multi-agent systems is how the agents communicate amongst themselves and with the orchestrator agent. This involves both the schema of how they communicate, as well as the sequence in which they communicate. LangGraph is perfect for orchestrating these types of systems and allows you to define both.

### Schema

LangGraph provides a lot of flexibility for how to communicate within multi-agent architectures.

* A node in LangGraph can have a private input state schema that is distinct from the graph state schema. This allows passing additional information during the graph execution that is only needed for executing a particular node.
* Subgraph agents can have independent input/output schemas. In this case it’s important to add input / output transformations so that the parent graph knows how to communicate with the subgraphs.
* For tool-based subordinate agents, the orchestrator determines the inputs based on the tool schema. Additionally, LangGraph allows passing state to individual tools at runtime, so subordinate agents can access parent state, if needed.

### Sequence

LangGraph provides multiple methods to control agent communication sequence:

* **Explicit control flow (graph edges)**: LangGraph allows you to define the control flow of your application (i.e. the sequence of how agents communicate) explicitly, via graph edges.
* **Implicity control flow (tool calling)**: if the orchestrator agent treats subordinate agents as tools, the tool-calling LLM powering the orchestrator will make decisions about the order in which the tools (agents) are being called
* **Dynamic control flow (conditional edges)**: LangGraph also allows you to define conditional edges, where the control flow is dependent on satisfying a given condition. In such cases, you can use an LLM to decide which subordinate agent to call next.
