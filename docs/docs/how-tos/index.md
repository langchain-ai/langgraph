---
hide:
  - navigation
title: How-to Guides
description: How to accomplish common tasks in LangGraph
---

# How-to Guides

Here you’ll find answers to “How do I...?” types of questions. These guides are **goal-oriented** and concrete; they're meant to help you complete a specific task. For conceptual explanations see the [Conceptual guide](../concepts/index.md). For end-to-end walk-throughs see [Tutorials](../tutorials/index.md). For comprehensive descriptions of every class and function see the [API Reference](../reference/index.md).

## LangGraph

### Controllability

LangGraph provides [low level building primitives](../concepts/low_level.md) that give you control over how you build and execute the graph.

- [How to create branches for parallel execution](branching.ipynb)
- [How to create map-reduce branches for parallel execution](map-reduce.ipynb)
- [How to control graph recursion limit](recursion-limit.ipynb): Set a [recursion limit](../concepts/low_level.md#recursion-limit) to limit the number of supersteps that the graph is allowed to execute during a single execution.
- [How to combine control flow and state updates with Command](command.ipynb)

### Persistence

[LangGraph Persistence](../concepts/persistence.md) makes it easy to persist state across graph runs (**thread-level** persistence) and across threads (**cross-thread** persistence). 

These how-to guides show how to enable persistence:

- [How to add thread-level persistence to graphs](persistence.ipynb)
- [How to add thread-level persistence to **subgraphs**](subgraph-persistence.ipynb)
- [How to add **cross-thread** persistence to graphs](cross-thread-persistence.ipynb): Use the [**Store**][langgraph.store.base.BaseStore] API to share state across conversational threads.


During development, you will often be using the [MemorySaver][langgraph.checkpoint.memory.MemorySaver] checkpointer. For production use, you will want to persist the data to a database. These how-to guides show how to use different databases for persistence:

- [How to use Postgres checkpointer for persistence](persistence_postgres.ipynb)
- [How to use MongoDB checkpointer for persistence](persistence_mongodb.ipynb)
- [How to create a custom checkpointer using Redis](persistence_redis.ipynb): A reference implementation of a custom checkpointer using Redis. Adapt this to your own needs.

### Memory

LangGraph makes it easy to manage conversation [memory](../concepts/memory.md) in your graph.

- [How to manage conversation history](memory/manage-conversation-history.ipynb): **trim** or **filter** messages from the conversation history to fit within the chat model's context window size.
- [How to delete messages](memory/delete-messages.ipynb): **delete** messages from the history using the built-in [RemoveMessages](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.modifier.RemoveMessage.html#langchain_core.messages.modifier.RemoveMessage) primitive.
- [How to add summary conversation memory](memory/add-summary-conversation-history.ipynb): Implement a **running summary** of the conversation history to fit within the chat model's context window size.

Cross-thread memory:

- [How to add long-term memory (cross-thread)](cross-thread-persistence.ipynb)
- [How to use semantic search for long-term memory](memory/semantic-search.ipynb)

### Human-in-the-loop

[Human-in-the-loop](../concepts/human_in_the_loop.md) functionality allows
you to involve humans in the decision-making process of your graph. These how-to guides show how to implement human-in-the-loop workflows in your graph.


Key workflows:

- [How to wait for user input](human_in_the_loop/wait-user-input.ipynb): A basic example that shows how to implement a human-in-the-loop workflow in your graph using the `interrupt` function.
- [How to review tool calls](human_in_the_loop/review-tool-calls.ipynb): Incorporate human-in-the-loop for reviewing/editing/accepting tool call requests before they executed using the `interrupt` function.
 

Other methods:

- [How to add static breakpoints](human_in_the_loop/breakpoints.ipynb): Use for debugging purposes. For [**human-in-the-loop**](../concepts/human_in_the_loop.md) workflows, we recommend the [`interrupt` function][langgraph.types.interrupt] instead.
- [How to edit graph state](human_in_the_loop/edit-graph-state.ipynb): Edit graph state using `graph.update_state` method. Use this if implementing a **human-in-the-loop** workflow via **static breakpoints**.
- [How to add dynamic breakpoints with `NodeInterrupt`](human_in_the_loop/dynamic_breakpoints.ipynb): **Not recommended**: Use the [`interrupt` function](../concepts/human_in_the_loop.md) instead.

### Time Travel

[Time travel](../concepts/time-travel.md) allows you to replay past actions in your LangGraph application to explore alternative paths and debug issues. These how-to guides show how to use time travel in your graph.

- [How to view and update past graph state](human_in_the_loop/time-travel.ipynb)

### Streaming

[Streaming](../concepts/streaming.md) is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.

- [How to stream full state of your graph](stream-values.ipynb): Use `graph.stream(stream_mode="values")` to stream the full state of your graph after each node execution.
- [How to stream state updates of your graph](stream-updates.ipynb): Use `graph.stream(stream_mode="updates")` to stream state updates of your graph after each node execution.
- [How to stream LLM tokens](streaming-tokens.ipynb): Use `stream_mode="messages"` to stream tokens from a chat model as they're generated.
- [How to stream custom data](streaming-content.ipynb): Use [`StreamWriter`][langgraph.types.StreamWriter] to write custom data to the `custom` stream.
- [How to multiple streaming modes the same time](stream-multiple.ipynb): Use `graph.stream(stream_mode=["values", "updates", ...])` to stream multiple modes at the same time.
- [How to stream from subgraphs](streaming-subgraphs.ipynb): Use `graph.stream(stream_mode=..., subgraph=True)` to include data from within **subgraphs**.

Streaming from specific parts of the application:

- [How to stream events from within a tool](streaming-events-from-within-tools.ipynb)
- [How to stream events from the final node](streaming-from-final-node.ipynb)
- [How to stream events from within a tool without LangChain models](streaming-events-from-within-tools-without-langchain.ipynb)

Working with chat models:
 
- [How to stream LLM tokens without LangChain models](streaming-tokens-without-langchain.ipynb): Use LangChain's **callback system** to stream tokens from a custom LLM that is not a [LangChain Chat Model]( https://python.langchain.com/docs/concepts/chat_models/).
- [How to disable streaming for models that don't support it](disable-streaming.ipynb): Pass `disable_streaming=True` when initializing th chat model; e.g., `ChatOpenAI(model="o1", disable_streaming=True)`.


### Tool calling

[Tool calling](https://python.langchain.com/docs/concepts/tool_calling/) is a type of chat model API that accepts tool schemas, along with messages, as input and returns invocations of those tools as part of the output message.

These how-to guides show common patterns for tool calling with LangGraph:

- [How to call tools using ToolNode](tool-calling.ipynb): Use the pre-built [`ToolNode`][langgraph.prebuilt.ToolNode] to execute tools.
- [How to handle tool calling errors](tool-calling-errors.ipynb)
- [How to pass runtime values to tools](pass-run-time-values-to-tools.ipynb)
??? "How to pass config to tools"

    **Full example**: [How to pass config to tools](pass-config-to-tools.ipynb)

    ```python
    from typing import Annotated

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import InjectedToolArg
    from langgraph.store.base import BaseStore

    from langgraph.prebuilt import InjectedState, InjectedStore


    # Can be sync or async; @tool decorator not required
    async def my_tool(
        # These arguments are populated by the LLM
        some_arg: str,
        another_arg: float,
        # The config: RunnableConfig is always available in LangChain calls
        # This is not exposed to the LLM
        config: RunnableConfig,
        # The following three are specific to the prebuilt ToolNode
        # (and `create_react_agent` by extension). If you are invoking the
        # tool on its own (in your own node), then you would need to provide these yourself.
        store: Annotated[BaseStore, InjectedStore],
        # This passes in the full state.
        state: Annotated[State, InjectedState],
        # You can also inject single fields from your state if you
        messages: Annotated[list, InjectedState("messages")]
        # The following is not compatible with create_react_agent or ToolNode
        # You can also exclude other arguments from being shown to the model.
        # These must be provided manually and are useful if you call the tools/functions in your own node
        # some_other_arg=Annotated["MyPrivateClass", InjectedToolArg],
    ):
        """Call my_tool to have an impact on the real world.

        Args:
            some_arg: a very important argument
            another_arg: another argument the LLM will provide
        """ # The docstring becomes the description for your tool and is passed to the model
        print(some_arg, another_arg, config, store, state, messages)
        # Config, some_other_rag, store, and state  are all "hidden" from
        # LangChain models when passed to bind_tools or with_structured_output
        return "... some response"
    ```

- [How to update graph state from tools](update-state-from-tools.ipynb)
- [How to handle large numbers of tools](many-tools.ipynb)

### Subgraphs

[Subgraphs](../concepts/low_level.md#subgraphs) allow you to reuse an existing graph from another graph. These how-to guides show how to use subgraphs:

- [How to add and use subgraphs](subgraph.ipynb)
- [How to view and update state in subgraphs](subgraphs-manage-state.ipynb)
- [How to transform inputs and outputs of a subgraph](subgraph-transform-state.ipynb)

### Multi-agent

[Multi-agent systems](../concepts/multi_agent.md) are useful to break down complex LLM applications into multiple agents, each responsible for a different part of the application. These how-to guides show how to implement multi-agent systems in LangGraph:

- [How to implement handoffs between agents](agent-handoffs.ipynb)
- [How to build a multi-agent network](multi-agent-network.ipynb)
- [How to add multi-turn conversation in a multi-agent application](multi-agent-multi-turn-convo.ipynb)

See the [multi-agent tutorials](../tutorials/index.md#multi-agent-systems) for implementations of other multi-agent architectures.

### State Management

- [How to use Pydantic model as state](state-model.ipynb)
- [How to define input/output schema for your graph](input_output_schema.ipynb)
- [How to pass private state between nodes inside the graph](pass_private_state.ipynb)

### Other

- [How to run graph asynchronously](async.ipynb)
- [How to visualize your graph](visualization.ipynb)
- [How to add runtime configuration to your graph](configuration.ipynb)
- [How to add node retries](node-retries.ipynb)
- [How to force function calling agent to structure output](react-agent-structured-output.ipynb)
- [How to pass custom LangSmith run ID for graph runs](run-id-langsmith.ipynb)
- [How to return state before hitting recursion limit](return-when-recursion-limit-hits.ipynb)
- [How to integrate LangGraph with AutoGen, CrewAI, and other frameworks](autogen-integration.ipynb)

### Prebuilt ReAct Agent

The LangGraph [prebuilt ReAct agent](../reference/prebuilt.md#langgraph.prebuilt.chat_agent_executor.create_react_agent) is pre-built implementation of a [tool calling agent](../concepts/agentic_concepts.md#tool-calling-agent).

One of the big benefits of LangGraph is that you can easily create your own agent architectures. So while it's fine to start here to build an agent quickly, we would strongly recommend learning how to build your own agent so that you can take full advantage of LangGraph.

These guides show how to use the prebuilt ReAct agent:

- [How to create a ReAct agent](create-react-agent.ipynb)
- [How to add memory to a ReAct agent](create-react-agent-memory.ipynb)
- [How to add a custom system prompt to a ReAct agent](create-react-agent-system-prompt.ipynb)
- [How to add human-in-the-loop processes to a ReAct agent](create-react-agent-hitl.ipynb)
- [How to create prebuilt ReAct agent from scratch](react-agent-from-scratch.ipynb)
- [How to add semantic search for long-term memory to a ReAct agent](memory/semantic-search.ipynb#using-in-create-react-agent)

## LangGraph Platform

This section includes how-to guides for LangGraph Platform.

LangGraph Platform is a commercial solution for deploying agentic applications in production, built on the open-source LangGraph framework.

The LangGraph Platform offers a few different deployment options described in the [deployment options guide](../concepts/deployment_options.md).

!!! tip

    * LangGraph is an MIT-licensed open-source library, which we are committed to maintaining and growing for the community.
    * You can always deploy LangGraph applications on your own infrastructure using the open-source LangGraph project without using LangGraph Platform.

### Application Structure

Learn how to set up your app for deployment to LangGraph Platform:

- [How to set up app for deployment (requirements.txt)](../cloud/deployment/setup.md)
- [How to set up app for deployment (pyproject.toml)](../cloud/deployment/setup_pyproject.md)
- [How to set up app for deployment (JavaScript)](../cloud/deployment/setup_javascript.md)
- [How to add semantic search](../cloud/deployment/semantic_search.md)
- [How to customize Dockerfile](../cloud/deployment/custom_docker.md)
- [How to test locally](../cloud/deployment/test_locally.md)
- [How to rebuild graph at runtime](../cloud/deployment/graph_rebuild.md)
- [How to use LangGraph Platform to deploy CrewAI, AutoGen, and other frameworks](autogen-langgraph-platform.ipynb)

### Deployment

LangGraph applications can be deployed using LangGraph Cloud, which provides a range of services to help you deploy, manage, and scale your applications.

- [How to deploy to LangGraph cloud](../cloud/deployment/cloud.md)
- [How to deploy to a self-hosted environment](./deploy-self-hosted.md)
- [How to interact with the deployment using RemoteGraph](./use-remote-graph.md)

### Assistants

[Assistants](../concepts/assistants.md) is a configured instance of a template.

- [How to configure agents](../cloud/how-tos/configuration_cloud.md)
- [How to version assistants](../cloud/how-tos/assistant_versioning.md)

### Threads

- [How to copy threads](../cloud/how-tos/copy_threads.md)
- [How to check status of your threads](../cloud/how-tos/check_thread_status.md)

### Runs

LangGraph Platform supports multiple types of runs besides streaming runs.

- [How to run an agent in the background](../cloud/how-tos/background_run.md)
- [How to run multiple agents in the same thread](../cloud/how-tos/same-thread.md)
- [How to create cron jobs](../cloud/how-tos/cron_jobs.md)
- [How to create stateless runs](../cloud/how-tos/stateless_runs.md)

### Streaming

Streaming the results of your LLM application is vital for ensuring a good user experience, especially when your graph may call multiple models and take a long time to fully complete a run. Read about how to stream values from your graph in these how to guides:

- [How to stream values](../cloud/how-tos/stream_values.md)
- [How to stream updates](../cloud/how-tos/stream_updates.md)
- [How to stream messages](../cloud/how-tos/stream_messages.md)
- [How to stream events](../cloud/how-tos/stream_events.md)
- [How to stream in debug mode](../cloud/how-tos/stream_debug.md)
- [How to stream multiple modes](../cloud/how-tos/stream_multiple.md)

### Human-in-the-loop

When designing complex graphs, relying entirely on the LLM for decision-making can be risky, particularly when it involves tools that interact with files, APIs, or databases. These interactions may lead to unintended data access or modifications, depending on the use case. To mitigate these risks, LangGraph allows you to integrate human-in-the-loop behavior, ensuring your LLM applications operate as intended without undesirable outcomes.

- [How to add a breakpoint](../cloud/how-tos/human_in_the_loop_breakpoint.md)
- [How to wait for user input](../cloud/how-tos/human_in_the_loop_user_input.md)
- [How to edit graph state](../cloud/how-tos/human_in_the_loop_edit_state.md)
- [How to replay and branch from prior states](../cloud/how-tos/human_in_the_loop_time_travel.md)
- [How to review tool calls](../cloud/how-tos/human_in_the_loop_review_tool_calls.md)

### Double-texting

Graph execution can take a while, and sometimes users may change their mind about the input they wanted to send before their original input has finished running. For example, a user might notice a typo in their original request and will edit the prompt and resend it. Deciding what to do in these cases is important for ensuring a smooth user experience and preventing your graphs from behaving in unexpected ways.

- [How to use the interrupt option](../cloud/how-tos/interrupt_concurrent.md)
- [How to use the rollback option](../cloud/how-tos/rollback_concurrent.md)
- [How to use the reject option](../cloud/how-tos/reject_concurrent.md)
- [How to use the enqueue option](../cloud/how-tos/enqueue_concurrent.md)

### Webhooks

- [How to integrate webhooks](../cloud/how-tos/webhooks.md)

### Cron Jobs

- [How to create cron jobs](../cloud/how-tos/cron_jobs.md)

### LangGraph Studio

LangGraph Studio is a built-in UI for visualizing, testing, and debugging your agents.

- [How to connect to a LangGraph Cloud deployment](../cloud/how-tos/test_deployment.md)
- [How to connect to a local dev server](../how-tos/local-studio.md)
- [How to connect to a local deployment (Docker)](../cloud/how-tos/test_local_deployment.md)
- [How to test your graph in LangGraph Studio (MacOS only)](../cloud/how-tos/invoke_studio.md)
- [How to interact with threads in LangGraph Studio](../cloud/how-tos/threads_studio.md)
- [How to add nodes as dataset examples in LangGraph Studio](../cloud/how-tos/datasets_studio.md)

## Troubleshooting

These are the guides for resolving common errors you may find while building with LangGraph. Errors referenced below will have an `lc_error_code` property corresponding to one of the below codes when they are thrown in code.

- [GRAPH_RECURSION_LIMIT](../troubleshooting/errors/GRAPH_RECURSION_LIMIT.md)
- [INVALID_CONCURRENT_GRAPH_UPDATE](../troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE.md)
- [INVALID_GRAPH_NODE_RETURN_VALUE](../troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE.md)
- [MULTIPLE_SUBGRAPHS](../troubleshooting/errors/MULTIPLE_SUBGRAPHS.md)
- [INVALID_CHAT_HISTORY](../troubleshooting/errors/INVALID_CHAT_HISTORY.md)
