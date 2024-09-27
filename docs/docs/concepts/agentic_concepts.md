# Agentic Concepts

This guide covers higher level agentic concepts, that are not LangGraph specific. This is a great guide to read if you looking to get familiar with agent concepts in general.

## Agents

There is no standard definition of an agent. The one we believe most accurately describes the technical underpinnings of what most people would call an agent: *an agent is a system that uses an LLM to decide the control flow of an application.*

## Cognitive Architecture

We use this to refer to the architecture of the agentic system. Given some input, the agent will run and produce some output. What is the logic that governs how that output is produced? What steps are run, and in what order? What prompts are used? What retrieval strategies are used? All these components make up the "cognitive architecture" of an agent, and are things you will learn to control!


## Structured Outputs

Structured outputs with LLMs work by providing a specific format or schema that the LLM should follow in its response. This is similar to tool calling, but more general. While tool calling typically involves selecting and using predefined functions, structured outputs can be used for any type of formatted response. Common methods to achieve structured outputs include:

1. Prompt engineering: Instructing the LLM to respond in a specific format.
2. Output parsers: Using post-processing to extract structured data from LLM responses.
3. Tool calling: Leveraging built-in tool calling capabilities of some LLMs to generate structured outputs.

Structured outputs are crucial for agents as they ensure the LLM's decision can be reliably interpreted and acted upon by the system. Learn more about [structured outputs in this how-to guide](https://python.langchain.com/docs/how_to/structured_output/).


## Tool calling

Tools are useful whenever you want an agent to interact with external systems. External systems (e.g., APIs) often require a particular input schema or payload, rather than natural language. When we bind an API, for example, as a tool we given the model awareness of the required input schema. The model will choose to call a tool based upon the natural language input from the user and  it will return an output that adheres to the tool's schema. 

[Many LLM providers support tool calling](https://python.langchain.com/v0.1/docs/integrations/chat/) and [tool calling interface](https://blog.langchain.dev/improving-core-tool-interfaces-and-docs-in-langchain/) in LangChain is simple: you can simply pass any Python `function` into `ChatModel.bind_tools(function)`.

![Tools](img/tool_call.png)

## Memory

Memory is crucial for agents, enabling them to retain and utilize information across multiple steps of problem-solving. It operates on different scales:

1. Short-term memory: Allows the agent to access information acquired during earlier steps in a sequence.
2. Long-term memory: Enables the agent to recall information from previous interactions, such as past messages in a conversation.

LangGraph provides full control over memory implementation:

- [`State`](./low_level.md#state): User-defined schema specifying the exact structure of memory to retain.
- [`Checkpointers`](./persistence.md): Mechanism to store state at every step across different interactions.

This flexible approach allows you to tailor the memory system to your specific agent architecture needs. For a practical guide on adding memory to your graph, see [this tutorial](../how-tos/persistence.ipynb).

Effective memory management enhances an agent's ability to maintain context, learn from past experiences, and make more informed decisions over time.

## Planning

This refers to the agents ability to plan out and reason about a sequence of multiple actions to take in row, where any given step may depend on the output of the step before it. LLMs themselves are getting better at planning, but many still use the "cognitive architecture" surrounding the LLM to make the LLM better at planning.

## Human-in-the-loop

Human involvement can significantly enhance agent reliability, especially for sensitive tasks. This can involve:

- Approving specific actions
- Providing feedback to update the agent's state
- Offering guidance in complex decision-making processes

Human-in-the-loop patterns are crucial when full automation isn't feasible or desirable. Learn more in our [human-in-the-loop guide](./human_in_the_loop.md).

## Sub-graphs

Sub-graphs are essential for managing complex agent architectures, particularly in multi-agent systems. They allow:

- Isolated state management for individual agents
- Hierarchical organization of agent teams
- Controlled communication between agents and the main system

Sub-graphs communicate with the parent graph through overlapping keys in the state schema. This enables flexible, modular agent design. For implementation details, refer to our [sub-graph tutorial](../how-tos/subgraph.ipynb).

## Reflection

Reflection mechanisms can significantly improve agent reliability by:

1. Evaluating task completion and correctness
2. Providing feedback for iterative improvement
3. Enabling self-correction and learning

While often LLM-based, reflection can also use deterministic methods. For instance, in coding tasks, compilation errors can serve as feedback. This approach is demonstrated in [this video using LangGraph for self-corrective code generation](https://www.youtube.com/watch?v=MvNdgmM7uyc).

By leveraging these features, LangGraph enables the creation of sophisticated, task-specific agent architectures that can handle complex workflows, collaborate effectively, and continuously improve their performance.
