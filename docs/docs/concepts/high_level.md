# Why LangGraph?

## LLM applications

LLMs make it possible to embed intelligence into a new class of applications. There are many patterns for building applications that use LLMs. Workflows have scaffolding of predefined code paths around LLM calls. LLMs can direct the control flow through these predefined code paths, which some consider to be an "agentic system". In other cases, it's possible to remove this scaffolding, creating autonomous agents that can [plan](https://huyenchip.com/2025/01/07/agents.html), take actions via [tool calls](https://python.langchain.com/docs/concepts/tool_calling/), and directly respond [to the feedback from their own actions](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/) with further actions.

![Agent Workflow](img/agent_workflow.png)

## What LangGraph provides

LangGraph provides low-level supporting infrastructure that sits underneath *any* workflow or agent. It does not abstract prompts or architecture, and provides three central benefits:

### Persistence

LangGraph has a [persistence layer](https://langchain-ai.github.io/langgraph/concepts/persistence/), which offers a number of benefits:

- [Memory](https://langchain-ai.github.io/langgraph/concepts/memory/): LangGraph persists arbitrary aspects of your application's state, supporting memory of conversations and other updates within and across user interactions;
- [Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/): Because state is checkpointed, execution can be interrupted and resumed, allowing for decisions, validation, and corrections via human input.

### Streaming

LangGraph also provides support for [streaming](../how-tos/index.md#streaming) workflow / agent state to the user (or developer) over the course of execution. LangGraph supports streaming of both events ([such as feedback from a tool call](../how-tos/streaming.ipynb#updates)) and [tokens from LLM calls](../how-tos/streaming-tokens.ipynb) embedded in an application.

### Debugging and Deployment

LangGraph provides an easy onramp for testing, debugging, and deploying applications via [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/). This includes [Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), an IDE that enables visualization, interaction, and debugging of workflows or agents. This also includes numerous [options](https://langchain-ai.github.io/langgraph/tutorials/deployment/) for deployment. 