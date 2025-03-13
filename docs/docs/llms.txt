# LangGraph

## Quickstart

These guides are designed to help you get started with LangGraph.

- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/): Build a chatbot that can use tools and keep track of conversation history. Add human-in-the-loop capabilities and explore how time-travel works.
- [Common Workflows](https://langchain-ai.github.io/langgraph/tutorials/workflows/): Overview of the most common workflows using LLMs implemented with LangGraph.
- [LangGraph Server Quickstart](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/): Launch a LangGraph server locally and interact with it using REST API and LangGraph Studio Web UI.
- [Deploy with LangGraph Cloud Quickstart](https://langchain-ai.github.io/langgraph/cloud/quick_start/): Deploy a LangGraph app using LangGraph Cloud.

## Concepts

These guides provide explanations of the key concepts behind the LangGraph framework.

- [Why LangGraph?](https://langchain-ai.github.io/langgraph/concepts/high_level/): Motivation for LangGraph, a library for building agentic applications with LLMs.
- [LangGraph Glossary](https://langchain-ai.github.io/langgraph/concepts/low_level/): LangGraph workflows are designed as graphs, with nodes representing different components and edges representing the flow of information between them. This guide provides an overview of the key concepts associated with LangGraph graph primitives.
- [Common Agentic Patterns](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/): An agent uses an LLM to pick its own control flow to solve more complex problems! Agents are a key building block in many LLM applications. This guide explains the different types of agent architectures and how they can be used to control the flow of an application.
- [Multi-Agent Systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/): Complex LLM applications can often be broken down into multiple agents, each responsible for a different part of the application. This guide explains common patterns for building multi-agent systems.
- [Breakpoints](https://langchain-ai.github.io/langgraph/concepts/breakpoints/): Breakpoints allow pausing the execution of a graph at specific points. Breakpoints allow stepping through graph execution for debugging purposes.
- [Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/): Explains different ways of integrating human feedback into a LangGraph application.
- [Time Travel](https://langchain-ai.github.io/langgraph/concepts/time-travel/): Time travel allows you to replay past actions in your LangGraph application to explore alternative paths and debug issues.
- [Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/): LangGraph has a built-in persistence layer, implemented through checkpointers. This persistence layer helps to support powerful capabilities like human-in-the-loop, memory, time travel, and fault-tolerance.
- [Memory](https://langchain-ai.github.io/langgraph/concepts/memory/): Memory in AI applications refers to the ability to process, store, and effectively recall information from past interactions. With memory, your agents can learn from feedback and adapt to users' preferences.
- [Streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/): Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.
- [Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/): `@entrypoint` and `@task` decorators that allow you to add LangGraph functionality to an existing codebase.
- [Durable Execution](https://langchain-ai.github.io/langgraph/concepts/durable_execution/): LangGraph's built-in [persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) layer provides durable execution for workflows, ensuring that the state of each execution step is saved to a durable store. 
- [Pregel](https://langchain-ai.github.io/langgraph/concepts/pregel/): Pregel is LangGraph's runtime, which is responsible for managing the execution of LangGraph applications.
- [FAQ](https://langchain-ai.github.io/langgraph/concepts/faq/): Frequently asked questions about LangGraph.

## How-tos

Here you’ll find answers to “How do I...?” types of questions. 

These guides are **goal-oriented** and concrete. 

They're meant to help you complete a specific task.

### Graph API Basics

- [How to update graph state from nodes](https://langchain-ai.github.io/langgraph/how-tos/state-reducers/)
- [How to create a sequence of steps](https://langchain-ai.github.io/langgraph/how-tos/sequence/)
- [How to create branches for parallel execution](https://langchain-ai.github.io/langgraph/how-tos/branching/)
- [How to create and control loops with recursion limits](https://langchain-ai.github.io/langgraph/how-tos/recursion-limit/)
- [How to visualize your graph](https://langchain-ai.github.io/langgraph/how-tos/visualization/)

### Fine-grained Control

These guides demonstrate LangGraph features that grant fine-grained control over the execution of your graph.

- [How to create map-reduce branches for parallel execution](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [How to update state and jump to nodes in graphs and subgraphs](https://langchain-ai.github.io/langgraph/how-tos/command/)
- [How to add runtime configuration to your graph](https://langchain-ai.github.io/langgraph/how-tos/configuration/)
- [How to add node retries](https://langchain-ai.github.io/langgraph/how-tos/node-retries/)
- [How to return state before hitting recursion limit](https://langchain-ai.github.io/langgraph/how-tos/return-when-recursion-limit-hits/)

### Persistence
 
Persistence makes it easy to persist state across graph runs (per-thread persistence) and across threads (cross-thread persistence). 

These how-to guides show how to add persistence to your graph.

- [How to add thread-level persistence to your graph](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [How to add thread-level persistence to a subgraph](https://langchain-ai.github.io/langgraph/how-tos/subgraph-persistence/)
- [How to add cross-thread persistence to your graph](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/)
- [How to use Postgres checkpointer for persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)
- [How to use MongoDB checkpointer for persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/)
- [How to create a custom checkpointer using Redis](https://langchain-ai.github.io/langgraph/how-tos/persistence_redis/)

See the below guides for how-to add persistence to your workflow using the [Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/):

- [How to add thread-level persistence (functional API)](https://langchain-ai.github.io/langgraph/how-tos/persistence-functional/)
- [How to add cross-thread persistence (functional API)](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence-functional/)

### Memory

LangGraph makes it easy to manage conversation memory in your graph. These how-to guides show how to implement different strategies for that.

- [How to manage conversation history](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/)
- [How to delete messages](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/)
- [How to add summary conversation memory](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/)
- [How to add long-term memory (cross-thread)](https://langchain-ai.github.io/langgraph/how-tos/memory/cross-thread-persistence/)
- [How to use semantic search for long-term memory](https://langchain-ai.github.io/langgraph/how-tos/memory/semantic-search/)

### Human-in-the-loop

Human-in-the-loop functionality allows you to involve humans in the decision-making process of your graph.

These how-to guides show how to implement human-in-the-loop workflows in your graph.

- [How to wait for user input](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/): A basic example that shows how to implement a human-in-the-loop workflow in your graph using the `interrupt` function.
- [How to review tool calls](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review-tool-calls/): Incorporate human-in-the-loop for reviewing/editing/accepting tool call requests before they executed using the `interrupt` function.
- [How to add static breakpoints](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/): Use for debugging purposes. For human-in-the-loop workflows, we recommend the [`interrupt` function](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt) instead.
- [How to edit graph state](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/): Edit graph state using `graph.update_state` method. Use this if implementing a **human-in-the-loop** workflow via **static breakpoints**.

See the below guides for how-to implement human-in-the-loop workflows with the Functional API.

- [How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/)
- [How to review tool calls (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/review-tool-calls-functional/)

### Time Travel

[Time travel](https://langchain-ai.github.io/langgraph/concepts/time-travel/) allows you to replay past actions in your LangGraph application to explore alternative paths and debug issues. These how-to guides show how to use time travel in your graph.

- [How to view and update past graph state](https://langchain-ai.github.io/langgraph/how-tos/time-travel/)

### Streaming

[Streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/) is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.

- [How to stream](https://langchain-ai.github.io/langgraph/how-tos/streaming/)
- [How to stream LLM tokens](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/)
- [How to stream LLM tokens from specific nodes](https://langchain-ai.github.io/langgraph/how-tos/streaming-specific-nodes/)
- [How to stream data from within a tool](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/)
- [How to stream from subgraphs](https://langchain-ai.github.io/langgraph/how-tos/streaming-subgraphs/)
- [How to disable streaming for models that don't support it](https://langchain-ai.github.io/langgraph/how-tos/disable-streaming/)

### Tool calling

[Tool calling](https://python.langchain.com/docs/concepts/tool_calling/) is a type of [chat model](https://python.langchain.com/docs/concepts/chat_models/) API.

It accepts tool schemas, along with messages, as input and returns invocations of those tools as part of the output message.

These how-to guides show common patterns for tool calling with LangGraph:

- [How to call tools using ToolNode](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/)
- [How to handle tool calling errors](https://langchain-ai.github.io/langgraph/how-tos/tool-calling-errors/)
- [How to pass runtime values to tools](https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/)
- [How to pass config to tools](https://langchain-ai.github.io/langgraph/how-tos/pass-config-to-tools/)
- [How to update graph state from tools](https://langchain-ai.github.io/langgraph/how-tos/update-state-from-tools/)
- [How to handle large numbers of tools](https://langchain-ai.github.io/langgraph/how-tos/many-tools/)

### Subgraphs

Subgraphs allow you to reuse an existing graph from another graph. 

These how-to guides show how to use subgraphs:

- [How to use subgraphs](https://langchain-ai.github.io/langgraph/how-tos/subgraph/)
- [How to view and update state in subgraphs](https://langchain-ai.github.io/langgraph/how-tos/subgraphs-manage-state/)
- [How to transform inputs and outputs of a subgraph](https://langchain-ai.github.io/langgraph/how-tos/subgraph-transform-state/)

### Multi-agent

Multi-agent systems are useful to break down complex LLM applications into multiple agents, each responsible for a different part of the application. 

These how-to guides show how to implement multi-agent systems in LangGraph:

- [How to implement handoffs between agents](https://langchain-ai.github.io/langgraph/how-tos/agent-handoffs/)
- [How to build a multi-agent network](https://langchain-ai.github.io/langgraph/how-tos/multi-agent-network/)
- [How to add multi-turn conversation in a multi-agent application](https://langchain-ai.github.io/langgraph/how-tos/multi-agent-multi-turn-convo/)

### State Management

- [How to use Pydantic model as graph state](https://langchain-ai.github.io/langgraph/how-tos/state-model/)
- [How to define input/output schema for your graph](https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/)
- [How to pass private state between nodes inside the graph](https://langchain-ai.github.io/langgraph/how-tos/pass_private_state/)

### Other

- [How to run graph asynchronously](https://langchain-ai.github.io/langgraph/how-tos/async/)
- [How to force tool-calling agent to structure output](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/)
- [How to pass custom LangSmith run ID for graph runs](https://langchain-ai.github.io/langgraph/how-tos/run-id-langsmith/)
- [How to integrate LangGraph with AutoGen, CrewAI, and other frameworks](https://langchain-ai.github.io/langgraph/how-tos/autogen-integration/)

## Use cases 

Explore practical implementations tailored for specific scenarios:

### Chatbots

- [Customer Support](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/): Build a multi-functional support bot for flights, hotels, and car rentals.
- [Prompt Generation from User Requirements](https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting/): Build an information gathering chatbot.
- [Code Assistant](https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/): Build a code analysis and generation assistant.

### RAG

- [Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/): Use an agent to figure out how to retrieve the most relevant information before using the retrieved information to answer the user's question.
- [Adaptive RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/): Adaptive RAG is a strategy for RAG that unites (1) query analysis with (2) active / self-corrective RAG. Implementation of: https://arxiv.org/abs/2403.14403
    - For a version that uses a local LLM: [Adaptive RAG using local LLMs](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/)
- [Corrective RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/): Uses an LLM to grade the quality of the retrieved information from the given source, and if the quality is low, it will try to retrieve the information from another source. Implementation of: https://arxiv.org/pdf/2401.15884.pdf 
    - For a version that uses a local LLM: [Corrective RAG using local LLMs](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/)
- [Self-RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/): Self-RAG is a strategy for RAG that incorporates self-reflection / self-grading on retrieved documents and generations. Implementation of https://arxiv.org/abs/2310.11511.
    - For a version that uses a local LLM: [Self-RAG using local LLMs](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag_local/) 
- [SQL Agent](https://langchain-ai.github.io/langgraph/tutorials/sql-agent/): Build a SQL agent that can answer questions about a SQL database.

### Multi-Agent Systems

- [Network](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/): Enable two or more agents to collaborate on a task
- [Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/): Use an LLM to orchestrate and delegate to individual agents
- [Hierarchical Teams](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/): Orchestrate nested teams of agents to solve problems
