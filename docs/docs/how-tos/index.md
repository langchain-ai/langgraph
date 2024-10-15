---
hide:
  - toc
---

# How-to guides

Welcome to the LangGraph how-to guides! These guides provide practical, step-by-step instructions for accomplishing key tasks in LangGraph.

## Controllability

LangGraph is known for being a highly controllable agent framework.
These how-to guides show how to achieve that controllability.

- [How to create branches for parallel execution](branching.ipynb)
- [How to create map-reduce branches for parallel execution](map-reduce.ipynb)
- [How to control graph recursion limit](recursion-limit.ipynb)

## Persistence

LangGraph makes it easy to persist state across graph runs (thread-level persistence) and across threads (cross-thread persistence). These how-to guides show how to add persistence to your graph.

- [How to add thread-level persistence to your graph](persistence.ipynb)
- [How to add thread-level persistence to subgraphs](subgraph-persistence.ipynb)
- [How to add cross-thread persistence to your graph](cross-thread-persistence.ipynb)
- [How to use Postgres checkpointer for persistence](persistence_postgres.ipynb)
- [How to create a custom checkpointer using MongoDB](persistence_mongodb.ipynb)
- [How to create a custom checkpointer using Redis](persistence_redis.ipynb)

## Memory

LangGraph makes it easy to manage conversation [memory](../concepts/memory.md) in your graph. These how-to guides show how to implement different strategies for that.

- [How to manage conversation history](memory/manage-conversation-history.ipynb)
- [How to delete messages](memory/delete-messages.ipynb)
- [How to add summary conversation memory](memory/add-summary-conversation-history.ipynb)

## Human in the Loop

One of LangGraph's main benefits is that it makes human-in-the-loop workflows easy.
These guides cover common examples of that.

- [How to add breakpoints](human_in_the_loop/breakpoints.ipynb)
- [How to add dynamic breakpoints](human_in_the_loop/dynamic_breakpoints.ipynb)
- [How to edit graph state](human_in_the_loop/edit-graph-state.ipynb)
- [How to wait for user input](human_in_the_loop/wait-user-input.ipynb)
- [How to view and update past graph state](human_in_the_loop/time-travel.ipynb)
- [Review tool calls](human_in_the_loop/review-tool-calls.ipynb)

## Streaming

LangGraph is built to be streaming first.
These guides show how to use different streaming modes.

- [How to stream full state of your graph](stream-values.ipynb)
- [How to stream state updates of your graph](stream-updates.ipynb)
- [How to stream LLM tokens](streaming-tokens.ipynb)
- [How to stream LLM tokens without LangChain models](streaming-tokens-without-langchain.ipynb)
- [How to stream custom data](streaming-content.ipynb)
- [How to configure multiple streaming modes at the same time](stream-multiple.ipynb)
- [How to stream events from within a tool](streaming-events-from-within-tools.ipynb)
- [How to stream events from within a tool without LangChain models](streaming-events-from-within-tools-without-langchain.ipynb)
- [How to stream events from the final node](streaming-from-final-node.ipynb)
- [How to stream from subgraphs](streaming-subgraphs.ipynb)
- [How to disable streaming for models that don't support it](disable-streaming.ipynb)

## Tool calling

- [How to call tools using ToolNode](tool-calling.ipynb)
- [How to handle tool calling errors](tool-calling-errors.ipynb)
- [How to pass runtime values to tools](pass-run-time-values-to-tools.ipynb)
- [How to pass config to tools](pass-config-to-tools.ipynb)
- [How to handle large numbers of tools](many-tools.ipynb)

## Subgraphs

- [How to add and use subgraphs](subgraph.ipynb)
- [How to view and update state in subgraphs](subgraphs-manage-state.ipynb)
- [How to transform inputs and outputs of a subgraph](subgraph-transform-state.ipynb)

## State Management

- [Use Pydantic model as state](state-model.ipynb)
- [Have a separate input and output schema](input_output_schema.ipynb)
- [Pass private state between nodes inside the graph](pass_private_state.ipynb)

## Other

- [How to run graph asynchronously](async.ipynb)
- [How to visualize your graph](visualization.ipynb)
- [How to add runtime configuration to your graph](configuration.ipynb)
- [How to use a Pydantic model as your state](state-model.ipynb)
- [How to add node retries](node-retries.ipynb)
- [How to force function calling agent to structure output](react-agent-structured-output.ipynb)
- [How to pass custom LangSmith run ID for graph runs](run-id-langsmith.ipynb)
- [How to return state before hitting recursion limit](return-when-recursion-limit-hits.ipynb)

## Prebuilt ReAct Agent

These guides show how to use the prebuilt ReAct agent.
Please note that here will we use a **prebuilt agent**. One of the big benefits of LangGraph is that you can easily create your own agent architectures. So while it's fine to start here to build an agent quickly, we would strongly recommend learning how to build your own agent so that you can take full advantage of LangGraph.

- [How to create a ReAct agent](create-react-agent.ipynb)
- [How to add memory to a ReAct agent](create-react-agent-memory.ipynb)
- [How to add a custom system prompt to a ReAct agent](create-react-agent-system-prompt.ipynb)
- [How to add human-in-the-loop processes to a ReAct agent](create-react-agent-hitl.ipynb)
- [How to create prebuilt ReAct agent from scratch](react-agent-from-scratch.ipynb)