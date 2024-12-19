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

??? "How to create branches for parallel execution"
    Full Example: [How to create branches for parallel execution](branching.ipynb)

    LangGraph enables parallel execution of nodes through fan-out and fan-in mechanisms, enhancing graph performance. By defining a state with a reducer function, you can manage how data is aggregated across parallel branches. Here's a concise example demonstrating this setup:

    ```python
    import operator
    from typing import Annotated, Any
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        # The operator.add reducer function appends to the list
        aggregate: Annotated[list, operator.add]

    class ReturnNodeValue:
        def __init__(self, node_secret: str):
            self._value = node_secret

        def __call__(self, state: State) -> Any:
            print(f"Adding {self._value} to {state['aggregate']}")
            return {"aggregate": [self._value]}

    builder = StateGraph(State)
    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_edge(START, "a")
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))
    builder.add_edge("a", ["b", "c"])  # Fan-out to B and C
    builder.add_edge(["b", "c"], "d")  # Fan-in to D
    builder.add_edge("d", END)
    graph = builder.compile()

    # Execute the graph
    graph.invoke({"aggregate": []})
    ```

    In this setup, the graph fans out from node "a" to nodes "b" and "c", then fans in to node "d". The `aggregate` list in the state accumulates values from each node, demonstrating parallel execution and data aggregation.  


??? "How to create map-reduce branches for parallel execution"

    Full Example: [How to create map-reduce branches for parallel execution](map-reduce.ipynb)

    Use the Send API to split your data or tasks into separate branches and process each in parallel, then combine the outputs with a “reduce” step. This lets you dynamically scale the number of parallel tasks without manually wiring each node.

    ```python
    from langgraph.types import Send

    def continue_to_jokes(state):
        # Distribute jokes generation for each subject in parallel
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
    ```

??? "How to control graph recursion limit"

    Full Example: [How to control graph recursion limit](recursion-limit.ipynb)

    Use the [recursion_limit](../concepts/low_level.md#recursion-limit) parameter in your graph’s invoke method to control how many supersteps are allowed before raising a GraphRecursionError. This guards against infinite loops and excessive computation time.

    ```python
    from langgraph.errors import GraphRecursionError

    try:
        # The recursion_limit sets the max number of supersteps
        # StateGraph, START, and END are relevant langgraph primitives.
        graph.invoke({"aggregate": []}, {"recursion_limit": 4})
    except GraphRecursionError:
        print("Recursion limit exceeded!")
    ```

??? "How to combine control flow and state updates with Command"

    Full Example: [How to combine control flow and state updates with Command](command.ipynb)

    Use a [Command][langgraph.types.Command] return type in a node function to simultaneously update the graph’s state and conditionally decide the next node in the graph. Combining both operations in one step removes the need for separate conditional edges.

    ```python
    from typing_extensions import Literal
    from langgraph.types import Command

    def my_node(state: dict) -> Command[Literal["other_node"]]:
        return Command(
            update={"foo": "bar"},   # state update
            goto="other_node"       # control flow
        )
    ```


### Persistence

[LangGraph Persistence](../concepts/persistence.md) makes it easy to persist state across graph runs (**thread-level** persistence) and across threads (**cross-thread** persistence).

These how-to guides show how to enable persistence:

??? "How to add thread-level persistence to graphs"
    Full Example: [How to add thread-level persistence to graphs](persistence.ipynb)

    Use the MemorySaver checkpointer when compiling your StateGraph to store conversation data across interactions. By specifying a thread_id, you can maintain or reset memory for each conversation thread as needed. This preserves context between messages while still allowing fresh starts.

    ```python
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    ```

??? "How to add thread-level persistence to **subgraphs**"
    Full Example: [How to add thread-level persistence to **subgraphs**](subgraph-persistence.ipynb)

    Pass a single checkpointer (e.g., MemorySaver) when compiling the parent graph, and LangGraph automatically propagates it to any child subgraphs. This avoids passing a checkpointer during subgraph compilation and ensures each thread’s state is captured at every step. 

??? "How to add **cross-thread** persistence to graphs"
    Full Example: [How to add **cross-thread** persistence to graphs](cross-thread-persistence.ipynb)

    Use the [**Store**][langgraph.store.base.BaseStore] API to share state across conversational threads.
    Use a shared Store (e.g., InMemoryStore) to persist user data across different threads. Namespaces keep data for each user separate, and the graph's nodes can retrieve or store memories by referencing the store and user_id. This example demonstrates how to compile a StateGraph with MemorySaver and cross-thread persistence enabled.

    ```python
    from langgraph.store.memory import InMemoryStore
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.checkpoint.memory import MemorySaver

    # Initialize a store to hold data across threads
    store = InMemoryStore()

    def my_node(state, config, *, store):
        # Use store to retrieve or store data as needed
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        # For example, store.put(namespace, "key", {"data": "Some info"})
        return state

    builder = StateGraph(MessagesState)
    builder.add_node("my_node", my_node)
    builder.add_edge(START, "my_node")

    # Pass the store when compiling, along with a checkpointer
    graph = builder.compile(checkpointer=MemorySaver(), store=store)
    ```

During development, you will often be using the [MemorySaver][langgraph.checkpoint.memory.MemorySaver] checkpointer. For production use, you will want to persist the data to a database. These how-to guides show how to use different databases for persistence:

??? "How to use Postgres checkpointer for persistence"
    Full Example: [How to use Postgres checkpointer for persistence](persistence_postgres.ipynb)

    Use PostgresSaver or its async variant (AsyncPostgresSaver) from langgraph.checkpoint.postgres to persist conversation or graph states in a PostgreSQL database, enabling your agents or graphs to retain context between runs. Just provide a psycopg connection/pool or a conn string, call setup() once, and pass the checkpointer when compiling (or creating) your StateGraph or agent.

    ```python
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.prebuilt import create_react_agent
    from psycopg import Connection

    DB_URI = "postgresql://user:password@host:port/db"

    with Connection.connect(DB_URI, autocommit=True) as conn:
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()  # Creates necessary tables if not already present
        graph = create_react_agent(model=..., tools=..., checkpointer=checkpointer)
        result = graph.invoke({"messages": [("human", "Hello!")]}, config={"configurable": {"thread_id": "example"}})
    ```

??? "How to use MongoDB checkpointer for persistence"
    Full Example: [How to use MongoDB checkpointer for persistence](persistence_mongodb.ipynb)

    Use the MongoDB checkpointer (MongoDBSaver) from langgraph-checkpoint-mongodb to store and retrieve your graph's state so you can persist interactions across multiple runs. Simply pass the checkpointer into the create_react_agent (or any compiled graph) to automatically handle saving and loading state from your MongoDB instance.

    ```python
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI

    MONGODB_URI = "mongodb://localhost:27017"
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    tools = []  # define your tools here

    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
        response = graph.invoke({"messages": [("user", "What's the weather in sf?")]})
        print(response)
    ```

??? "How to create a custom checkpointer using Redis"
    Full Example: [How to create a custom checkpointer using Redis](persistence_redis.ipynb)

    A reference implementation of a custom checkpointer using Redis. Adapt this to your own needs.

### Memory

LangGraph makes it easy to manage conversation [memory](../concepts/memory.md) in your graph.

??? "How to manage conversation history"
    Full Example: [How to manage conversation history](memory/manage-conversation-history.ipynb)

    **Trim** or **filter** messages from the conversation history to fit within the chat model's context window size.


??? "How to delete messages"
    Full Example: [How to delete messages](memory/delete-messages.ipynb)

    You can remove messages from a conversation by passing RemoveMessage objects to the state, provided your MessagesState (or similar) is set up with a reducer that processes them. This helps keep the message list concise and maintain model requirements (e.g. not starting with an AI message). Make sure the remaining conversation flow still follows any format rules your model requires.

    ```python
    # Minimal example illustrating message removal:
    from langchain_core.messages import RemoveMessage

    # Suppose 'app' is a compiled StateGraph using a MessagesState reducer
    # and 'config' is your configuration dictionary with a specific thread_id.
    messages = app.get_state(config).values["messages"]
    message_id_to_remove = messages[0].id

    app.update_state(
        config,
        {"messages": RemoveMessage(id=message_id_to_remove)}
    )

    # 'messages[0]' is now deleted from the conversation state.
    ```

??? "How to add summary conversation memory"
    Full Example: [How to add summary conversation memory](memory/add-summary-conversation-history.ipynb)

    Implement a **running summary** of the conversation history to fit within the chat model's context window size.


Cross-thread memory:

??? "How to add long-term memory (cross-thread)"
    Full Example: [How to add long-term memory (cross-thread)](cross-thread-persistence.ipynb)

    Use the [**Store**][langgraph.store.base.BaseStore] API to share state across conversational threads.
    Use a shared Store (e.g., InMemoryStore) to persist user data across different threads. Namespaces keep data for each user separate, and the graph's nodes can retrieve or store memories by referencing the store and user_id. This example demonstrates how to compile a StateGraph with MemorySaver and cross-thread persistence enabled.

    ```python
    from langgraph.store.memory import InMemoryStore
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.checkpoint.memory import MemorySaver

    # Initialize a store to hold data across threads
    store = InMemoryStore()

    def my_node(state, config, *, store):
        # Use store to retrieve or store data as needed
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        # For example, store.put(namespace, "key", {"data": "Some info"})
        return state

    builder = StateGraph(MessagesState)
    builder.add_node("my_node", my_node)
    builder.add_edge(START, "my_node")

    # Pass the store when compiling, along with a checkpointer
    graph = builder.compile(checkpointer=MemorySaver(), store=store)
    ```

??? "How to use semantic search for long-term memory"
    Full Example: [How to use semantic search for long-term memory](memory/semantic-search.ipynb)

    Enable semantic search in your agent by providing an index configuration (e.g., embeddings, vector dimensions) when creating an InMemoryStore. Then, simply store entries with store.put(...) and retrieve semantically similar items using store.search(...).

    ```python
    from langchain.embeddings import init_embeddings
    from langgraph.store.memory import InMemoryStore

    # Initialize embeddings and store
    embeddings = init_embeddings("openai:text-embedding-3-small")
    store = InMemoryStore(index={"embed": embeddings, "dims": 1536})

    # Store some items
    store.put(("agent_id", "memories"), "1", {"text": "I love pizza"})

    # Search semantically
    results = store.search(("agent_id", "memories"), "food preferences", limit=1)
    print(results)
    ```

### Human-in-the-loop

[Human-in-the-loop](../concepts/human_in_the_loop.md) functionality allows you to involve humans in the decision-making process of your graph. These how-to guides show how to implement human-in-the-loop workflows in your graph.

Key workflows:

??? "How to wait for user input"
    Full Example: [How to wait for user input](human_in_the_loop/wait-user-input.ipynb)

    A basic example that shows how to implement a human-in-the-loop workflow in your graph using the `interrupt` function.

??? "How to review tool calls"
    Full Example: [How to review tool calls](human_in_the_loop/review-tool-calls.ipynb)

    Incorporate human-in-the-loop for reviewing/editing/accepting tool call requests before they are executed using the `interrupt` function.

Other methods:

??? "How to add static breakpoints"
    Full Example: [How to add static breakpoints](human_in_the_loop/breakpoints.ipynb)

    Static breakpoints are used to pause graph execution at predetermined nodes, aiding in debugging and inspection of state at specific stages. You can set these breakpoints at compile time or run time by specifying `interrupt_before` and `interrupt_after` parameters. Here's a quick example:
    
    ```python
    graph = graph_builder.compile(
        interrupt_before=["node_a"], 
        interrupt_after=["node_b", "node_c"]
    )
    ```


??? "How to edit graph state"
    Full Example: [How to edit graph state](human_in_the_loop/edit-graph-state.ipynb)

    Edit graph state using the `graph.update_state` method. Use this if implementing a **human-in-the-loop** workflow via **static breakpoints**.

??? "How to add dynamic breakpoints with `NodeInterrupt` (not recommended)"
    Full Example: [How to add dynamic breakpoints with `NodeInterrupt`](human_in_the_loop/dynamic_breakpoints.ipynb)

    **Not recommended**: Use the [`interrupt` function](../concepts/human_in_the_loop.md) instead.

### Time Travel

[Time travel](../concepts/time-travel.md) allows you to replay past actions in your LangGraph application to explore alternative paths and debug issues. These how-to guides show how to use time travel in your graph.

??? "How to view and update past graph state"
    Full Example: [How to view and update past graph state](human_in_the_loop/time-travel.ipynb)

    Replay and modify past graph states to explore alternative paths and debug issues in your application.

### Streaming

[Streaming](../concepts/streaming.md) is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.

??? "How to stream full state of your graph"
    Full Example: [How to stream full state of your graph](stream-values.ipynb)  

    Use `graph.stream(stream_mode="values")` to stream the full state of your graph after each node execution.

??? "How to stream state updates of your graph"
    Full Example: [How to stream state updates of your graph](stream-updates.ipynb)  

    Use `graph.stream(stream_mode="updates")` to stream state updates of your graph after each node execution.

??? "How to stream custom data"
    Full Example: [How to stream custom data](streaming-content.ipynb)  

    Use [`StreamWriter`][langgraph.types.StreamWriter] to write custom data to the `custom` stream.

??? "How to multiple streaming modes at the same time"
    Full Example: [How to multiple streaming modes the same time](stream-multiple.ipynb)  

    Use `graph.stream(stream_mode=["values", "updates", ...])` to stream multiple modes at the same time.

Streaming from specific parts of the application:

??? "How to stream from subgraphs"
    Full Example: [How to stream from subgraphs](streaming-subgraphs.ipynb)

    Use `graph.stream(stream_mode=..., subgraph=True)` to include data from within **subgraphs**.

??? "How to stream events from within a tool"
    Full Example: [How to stream events from within a tool](streaming-events-from-within-tools.ipynb)

??? "How to stream events from the final node"
    Full Example: [How to stream events from the final node](streaming-from-final-node.ipynb)


Working with chat models:

??? "How to stream LLM tokens"

    Full Example: [How to stream LLM tokens](streaming-tokens.ipynb)

    Use `stream_mode="messages"` to stream tokens from a chat model as they're generated.

??? "How to disable streaming for models that don't support it"
    Full Example: [How to disable streaming for models that don't support it](disable-streaming.ipynb)

    Pass `disable_streaming=True` when initializing the chat model; e.g., `ChatOpenAI(model="o1", disable_streaming=True)`.

??? "How to stream LLM tokens without LangChain models"
    Full Example: [How to stream LLM tokens without LangChain models](streaming-tokens-without-langchain.ipynb)  

    Use LangChain's **callback system** to stream tokens from a custom LLM that is not a [LangChain Chat Model](https://python.langchain.com/docs/concepts/chat_models/).

??? "How to stream events from within a tool without LangChain models"
	Full Example: [How to stream events from within a tool without LangChain models](streaming-events-from-within-tools-without-langchain.ipynb)


### Tool Calling

[Tool calling](https://python.langchain.com/docs/concepts/tool_calling/) is a type of chat model API that accepts tool schemas, along with messages, as input and returns invocations of those tools as part of the output message.

??? "How to call tools using ToolNode"
    Full Example: [How to call tools using ToolNode](tool-calling.ipynb)

    Use the pre-built [`ToolNode`][langgraph.prebuilt.ToolNode] to execute tools.

??? "How to handle tool calling errors"
    Full Example: [How to handle tool calling errors](tool-calling-errors.ipynb)

??? "How to pass runtime values to tools"
    Full Example: [How to pass runtime values to tools](pass-run-time-values-to-tools.ipynb)

    ```python
    from typing import Annotated
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import InjectedToolArg
    from langgraph.store.base import BaseStore
    from langgraph.prebuilt import InjectedState, InjectedStore

    async def my_tool(
        some_arg: str,
        another_arg: float,
        config: RunnableConfig,
        store: Annotated[BaseStore, InjectedStore],
        state: Annotated[State, InjectedState],
        messages: Annotated[list, InjectedState("messages")]
    ):
        """Call my_tool to have an impact on the real world.

        Args:
            some_arg: a very important argument
            another_arg: another argument the LLM will provide
        """
        print(some_arg, another_arg, config, store, state, messages)
        return "... some response"
    ```

??? "How to pass config to tools"
    Full Example: [How to pass config to tools](pass-config-to-tools.ipynb)

??? "How to update graph state from tools"
    Full Example: [How to update graph state from tools](update-state-from-tools.ipynb)

??? "How to handle large numbers of tools"
    Full Example: [How to handle large numbers of tools](many-tools.ipynb)


### Subgraphs

[Subgraphs](../concepts/low_level.md#subgraphs) allow you to reuse an existing graph from another graph.

??? "How to add and use subgraphs"
    Full Example: [How to add and use subgraphs](subgraph.ipynb)

??? "How to view and update state in subgraphs"
    Full Example: [How to view and update state in subgraphs](subgraphs-manage-state.ipynb)

??? "How to transform inputs and outputs of a subgraph"
    Full Example: [How to transform inputs and outputs of a subgraph](subgraph-transform-state.ipynb)

### Multi-Agent

[Multi-agent systems](../concepts/multi_agent.md) are useful to break down complex LLM applications into multiple agents, each responsible for a different part of the application. These how-to guides show how to implement multi-agent systems in LangGraph:

??? "How to implement handoffs between agents"
    Full Example: [How to implement handoffs between agents](agent-handoffs.ipynb)

??? "How to build a multi-agent network"
    Full Example: [How to build a multi-agent network](multi-agent-network.ipynb)

??? "How to add multi-turn conversation in a multi-agent application"
    Full Example: [How to add multi-turn conversation in a multi-agent application](multi-agent-multi-turn-convo.ipynb)

See the [multi-agent tutorials](../tutorials/index.md#multi-agent-systems) for implementations of other multi-agent architectures.

### State Management

??? "How to use Pydantic model as state"
    Full Example: [How to use Pydantic model as state](state-model.ipynb)

??? "How to define input/output schema for your graph"
    Full Example: [How to define input/output schema for your graph](input_output_schema.ipynb)

??? "How to pass private state between nodes inside the graph"
    Full Example: [How to pass private state between nodes inside the graph](pass_private_state.ipynb)

### Other

??? "How to run graph asynchronously"
    Full Example: [How to run graph asynchronously](async.ipynb)

??? "How to visualize your graph"
    Full Example: [How to visualize your graph](visualization.ipynb)

??? "How to add runtime configuration to your graph"
    Full Example: [How to add runtime configuration to your graph](configuration.ipynb)

??? "How to add node retries"
    Full Example: [How to add node retries](node-retries.ipynb)

??? "How to force function calling agent to structure output"
    Full Example: [How to force function calling agent to structure output](react-agent-structured-output.ipynb)

??? "How to pass custom LangSmith run ID for graph runs"
    Full Example: [How to pass custom LangSmith run ID for graph runs](run-id-langsmith.ipynb)

??? "How to return state before hitting recursion limit"
    Full Example: [How to return state before hitting recursion limit](return-when-recursion-limit-hits.ipynb)

??? "How to integrate LangGraph with AutoGen, CrewAI, and other frameworks"
    Full Example: [How to integrate LangGraph with AutoGen, CrewAI, and other frameworks](autogen-integration.ipynb)


### Prebuilt ReAct Agent

The LangGraph [prebuilt ReAct agent](../reference/prebuilt.md#langgraph.prebuilt.chat_agent_executor.create_react_agent) is a pre-built implementation of a [tool calling agent](../concepts/agentic_concepts.md#tool-calling-agent).

One of the big benefits of LangGraph is that you can easily create your own agent architectures. So while it's fine to start here to build an agent quickly, we would strongly recommend learning how to build your own agent so that you can take full advantage of LangGraph.

These guides show how to use the prebuilt ReAct agent:

??? "How to create a ReAct agent"
    Full Example: [How to create a ReAct agent](create-react-agent.ipynb)

??? "How to add memory to a ReAct agent"
    Full Example: [How to add memory to a ReAct agent](create-react-agent-memory.ipynb)

??? "How to add a custom system prompt to a ReAct agent"
    Full Example: [How to add a custom system prompt to a ReAct agent](create-react-agent-system-prompt.ipynb)

??? "How to add human-in-the-loop processes to a ReAct agent"
    Full Example: [How to add human-in-the-loop processes to a ReAct agent](create-react-agent-hitl.ipynb)

??? "How to create prebuilt ReAct agent from scratch"
    Full Example: [How to create prebuilt ReAct agent from scratch](react-agent-from-scratch.ipynb)

??? "How to add semantic search for long-term memory to a ReAct agent"
    Full Example: [How to add semantic search for long-term memory to a ReAct agent](memory/semantic-search.ipynb#using-in-create-react-agent)

## LangGraph Platform

This section includes how-to guides for LangGraph Platform.

LangGraph Platform is a commercial solution for deploying agentic applications in production, built on the open-source LangGraph framework.

The LangGraph Platform offers a few different deployment options described in the [deployment options guide](../concepts/deployment_options.md).

!!! tip

    * LangGraph is an MIT-licensed open-source library, which we are committed to maintaining and growing for the community.
    * You can always deploy LangGraph applications on your own infrastructure using the open-source LangGraph project without using LangGraph Platform.

### Application Structure

Learn how to set up your app for deployment to LangGraph Platform:

??? "How to set up app for deployment"

    *  [How to set up app for deployment (requirements.txt)](../cloud/deployment/setup.md)
    *  [How to set up app for deployment (pyproject.toml)](../cloud/deployment/setup_pyproject.md)
    *  [How to set up app for deployment (JavaScript)](../cloud/deployment/setup_javascript.md)

??? "How to customize Dockerfile"

    Full Example: [How to customize Dockerfile](../cloud/deployment/custom_docker.md)

??? "How to test locally"

    Full Example: [How to test locally](../cloud/deployment/test_locally.md)

??? "How to rebuild graph at runtime"

    Full Example: [How to rebuild graph at runtime](../cloud/deployment/graph_rebuild.md)

??? "How to use LangGraph Platform to deploy CrewAI, AutoGen, and other frameworks"

    Full Example: [How to use LangGraph Platform to deploy CrewAI, AutoGen, and other frameworks](autogen-langgraph-platform.ipynb)


### Deployment

LangGraph applications can be deployed using LangGraph Cloud, which provides a range of services to help you deploy, manage, and scale your applications.

??? "How to deploy to LangGraph cloud"
	Full Example: [How to deploy to LangGraph cloud](../cloud/deployment/cloud.md)

??? "How to deploy to a self-hosted environment"
	Full Example: [How to deploy to a self-hosted environment](./deploy-self-hosted.md)

??? "How to interact with the deployment using RemoteGraph"
	Full Example: [How to interact with the deployment using RemoteGraph](./use-remote-graph.md)

### Assistants

[Assistants](../concepts/assistants.md) is a configured instance of a template.

??? "How to configure agents"
	Full Example: [How to configure agents](../cloud/how-tos/configuration_cloud.md)

??? "How to version assistants"
	Full Example: [How to version assistants](../cloud/how-tos/assistant_versioning.md)

### Threads

??? "How to copy threads"
	Full Example: [How to copy threads](../cloud/how-tos/copy_threads.md)

??? "How to check status of your threads"
	Full Example: [How to check status of your threads](../cloud/how-tos/check_thread_status.md)

### Runs

LangGraph Platform supports multiple types of runs besides streaming runs.

??? "How to run an agent in the background"
	Full Example: [How to run an agent in the background](../cloud/how-tos/background_run.md)

??? "How to run multiple agents in the same thread"
	Full Example: [How to run multiple agents in the same thread](../cloud/how-tos/same-thread.md)

??? "How to create cron jobs"
	Full Example: [How to create cron jobs](../cloud/how-tos/cron_jobs.md)

??? "How to create stateless runs"
	Full Example: [How to create stateless runs](../cloud/how-tos/stateless_runs.md)

### Streaming

Streaming the results of your LLM application is vital for ensuring a good user experience, especially when your graph may call multiple models and take a long time to fully complete a run. Read about how to stream values from your graph in these how-to guides:

??? "How to stream values"
	Full Example: [How to stream values](../cloud/how-tos/stream_values.md)

??? "How to stream updates"
	Full Example: [How to stream updates](../cloud/how-tos/stream_updates.md)

??? "How to stream messages"
	Full Example: [How to stream messages](../cloud/how-tos/stream_messages.md)

??? "How to stream events"
	Full Example: [How to stream events](../cloud/how-tos/stream_events.md)

??? "How to stream in debug mode"
	Full Example: [How to stream in debug mode](../cloud/how-tos/stream_debug.md)

??? "How to stream multiple modes"
	Full Example: [How to stream multiple modes](../cloud/how-tos/stream_multiple.md)

### Human-in-the-loop

When designing complex graphs, relying entirely on the LLM for decision-making can be risky, particularly when it involves tools that interact with files, APIs, or databases. These interactions may lead to unintended data access or modifications, depending on the use case. To mitigate these risks, LangGraph allows you to integrate human-in-the-loop behavior, ensuring your LLM applications operate as intended without undesirable outcomes.

??? "How to add a breakpoint"
	Full Example: [How to add a breakpoint](../cloud/how-tos/human_in_the_loop_breakpoint.md)

??? "How to wait for user input"
	Full Example: [How to wait for user input](../cloud/how-tos/human_in_the_loop_user_input.md)

??? "How to edit graph state"
	Full Example: [How to edit graph state](../cloud/how-tos/human_in_the_loop_edit_state.md)

??? "How to replay and branch from prior states"
	Full Example: [How to replay and branch from prior states](../cloud/how-tos/human_in_the_loop_time_travel.md)

??? "How to review tool calls"
	Full Example: [How to review tool calls](../cloud/how-tos/human_in_the_loop_review_tool_calls.md)

### Double-texting

Graph execution can take a while, and sometimes users may change their mind about the input they wanted to send before their original input has finished running. For example, a user might notice a typo in their original request and will edit the prompt and resend it. Deciding what to do in these cases is important for ensuring a smooth user experience and preventing your graphs from behaving in unexpected ways.

??? "How to use the interrupt option"
	Full Example: [How to use the interrupt option](../cloud/how-tos/interrupt_concurrent.md)

??? "How to use the rollback option"
	Full Example: [How to use the rollback option](../cloud/how-tos/rollback_concurrent.md)

??? "How to use the reject option"
	Full Example: [How to use the reject option](../cloud/how-tos/reject_concurrent.md)

??? "How to use the enqueue option"
	Full Example: [How to use the enqueue option](../cloud/how-tos/enqueue_concurrent.md)

### Webhooks

??? "How to integrate webhooks"
	Full Example: [How to integrate webhooks](../cloud/how-tos/webhooks.md)

### Cron Jobs

??? "How to create cron jobs"
	Full Example: [How to create cron jobs](../cloud/how-tos/cron_jobs.md)

### LangGraph Studio

LangGraph Studio is a built-in UI for visualizing, testing, and debugging your agents.

??? "How to connect to a LangGraph Cloud deployment"
	Full Example: [How to connect to a LangGraph Cloud deployment](../cloud/how-tos/test_deployment.md)

??? "How to connect to a local dev server"
	Full Example: [How to connect to a local dev server](../how-tos/local-studio.md)

??? "How to connect to a local deployment (Docker)"
	Full Example: [How to connect to a local deployment (Docker)](../cloud/how-tos/test_local_deployment.md)

??? "How to test your graph in LangGraph Studio (MacOS only)"
	Full Example: [How to test your graph in LangGraph Studio (MacOS only)](../cloud/how-tos/invoke_studio.md)

??? "How to interact with threads in LangGraph Studio"
	Full Example: [How to interact with threads in LangGraph Studio](../cloud/how-tos/threads_studio.md)

??? "How to add nodes as dataset examples in LangGraph Studio"
	Full Example: [How to add nodes as dataset examples in LangGraph Studio](../cloud/how-tos/datasets_studio.md)

## Troubleshooting

These are the guides for resolving common errors you may find while building with LangGraph. Errors referenced below will have an `lc_error_code` property corresponding to one of the below codes when they are thrown in code.

- [GRAPH_RECURSION_LIMIT](../troubleshooting/errors/GRAPH_RECURSION_LIMIT.md)
- [INVALID_CONCURRENT_GRAPH_UPDATE](../troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE.md)
- [INVALID_GRAPH_NODE_RETURN_VALUE](../troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE.md)
- [MULTIPLE_SUBGRAPHS](../troubleshooting/errors/MULTIPLE_SUBGRAPHS.md)
- [INVALID_CHAT_HISTORY](../troubleshooting/errors/INVALID_CHAT_HISTORY.md)
