---
hide:
  - toc
---

# How-to guides

Welcome to the LangGraph how-to guides! These guides provide practical, step-by-step instructions for accomplishing key tasks in LangGraph.

## Controllability

LangGraph is known for being a highly controllable agent framework.
These how-to guides show how to achieve that controllability.

- [How to create subgraphs](subgraph.ipynb)
- [How to create branches for parallel execution](branching.ipynb)
- [How to create map-reduce branches for parallel execution](map-reduce.ipynb)

## Human in the Loop

One of LangGraph's main benefits is that it makes human-in-the-loop workflows easy.
These guides cover common examples of that.

- [How to add persistence ("memory") to your graph](persistence.ipynb)
- [How to add breakpoints](human-in-the-loop/breakpoints.ipynb)
- [How to edit graph state](human-in-the-loop/edit-graph-state.ipynb)
- [How to wait for user input](human-in-the-loop/wait-user-input.ipynb)
- [How to view and update past graph state](human-in-the-loop/time-travel.ipynb)

## Streaming

LangGraph is built to be streaming first.
These guides show how to use different streaming modes.

- [How to stream LLM tokens](streaming-tokens.ipynb)
- [How to stream arbitrarily nested content](streaming-content.ipynb)

## Other
- [How to run graph asynchronously](async.ipynb)
- [How to visualize your graph](visualization.ipynb)
- [How to add runtime configuration to your graph](configuration.ipynb)
- [How to use a Pydantic model as your state](state-model.ipynb)

## Prebuilt ReAct Agent

These guides show how to use the prebuilt ReAct agent.
Please note that here will we use a **prebuilt agent**. One of the big benefits of LangGraph is that you can easily create your own agent architectures. So while it's fine to start here to build an agent quickly, we would strongly recommend learning how to build your own agent so that you can take full advantage of LangGraph.

- [How to create a ReAct agent](create-react-agent.ipynb)
- [How to add memory to a ReAct agent](create-react-agent-memory.ipynb)
- [How to add a custom system prompt to a ReAct agent](create-react-agent-system-prompt.ipynb)
- [How to add human-in-the-loop processes to a ReAct agent](create-react-agent-hitl.ipynb)
