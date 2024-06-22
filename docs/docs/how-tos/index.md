---
hide:
  - toc
---

# How-to guides

Welcome to the LangGraph how-to guides! These guides provide practical, step-by-step instructions for accomplishing key tasks in LangGraph.

## Basics

These guides show how to address common needs when building out AI workflows, with special focus placed on [ReAct](https://arxiv.org/abs/2210.03629)-style agents with [tool calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/) (agents that <strong>Re</strong>ason and **Act** to accomplish tasks).

- [How to create a ReAct agent](create-react-agent.ipynb)
- [How to add persistence ("memory") to your graph](persistence.ipynb)
- [How to view and update graph state](time-travel.ipynb)
- [How to run graph asynchronously](async.ipynb)
- [How to stream graph responses](streaming-tokens.ipynb)
- [How to visualize your graph](visualization.ipynb)
- [How to add runtime configuration to your graph](configuration.ipynb)

### Design patterns

Recipes showing how to apply common design patterns in your workflows:

- [How to create subgraphs](subgraph.ipynb)
- [How to create branches for parallel execution](branching.ipynb)
- [How to create map-reduce branches for parallel execution](map-reduce.ipynb)
- [How to add human-in-the-loop](human-in-the-loop.ipynb)

The following examples are useful especially if you are used to LangChain's `AgentExecutor` configurations.

- [How to force an agent to call a tool](force-calling-a-tool-first.ipynb)
- [How to pass runtime values to tools](pass-run-time-values-to-tools.ipynb)
- [How to let agent return tool results directly](dynamically-returning-directly.ipynb)
- [How to have agent respond in structured format](respond-in-format.ipynb)
- [How to manage agent steps](managing-agent-steps.ipynb)

### Advanced

- [How to use a Pydantic model as your state](state-model.ipynb)
- [How to extract structured output with re-prompting](./extraction/retries.ipynb)