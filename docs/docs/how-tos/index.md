# How-To Guides

Welcome to the LangGraph How-To Guides! These guides provide practical, step-by-step instructions for accomplishing key tasks in LangGraph. 

## Core

The core guides show how to address common needs when building a out AI workflows, with special focus placed on [ReAct](https://arxiv.org/abs/2210.03629)-style agents with [tool calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/).

- [Persistence](persistence.ipynb): How to give your graph "memory" and resiliance by saving and loading state
- [Time Travel](time-travel.ipynb): How to navigate and manipulate graph state history once it's persisted
- [Async Execution](async.ipynb): How to run nodes asynchronously for improved performance
- [Streaming Responses](streaming-tokens.ipynb): How to stream agent responses in real-time
- [Visualization](visualization.ipynb): How to visualize your graphs

### Design Patterns

Recipes showing how to apply common design patterns in your workflows:

- [Subgraphs](subgraph.ipynb): How to compose subgraphs within a larger graph
- [Branching](branching.ipynb): How to create branching logic in your graphs for parallel node execution
- [Human-in-the-Loop](human-in-the-loop.ipynb): How to incorporate human feedback and intervention
- [Force Calling a Tool First](force-calling-a-tool-first.ipynb): Define a fixed workflow before ceding control to the ReAct agent

### Alternative ways to define State

- [Pydantic State](state-model.ipynb): Use a pydantic model as your state

Update this:
- [Dynamic Direct Return](chat_agent_executor_with_function_calling/dynamically-returning-directly.ipynb)
- Respond in format
- Manage Agent Steps

Add these:

-
