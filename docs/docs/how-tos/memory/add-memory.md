# Add and manage memory

AI applications need [memory](../../concepts/memory.md) to share context across multiple interactions. In LangGraph, you can add two types of memory:

- [Add short-term memory](#add-short-term-memory) as a part of your agent's [state](../../concepts/low_level.md#state) to enable multi-turn conversations.
- [Add long-term memory](#add-long-term-memory) to store user-specific or application-level data across sessions.

## Add short-term memory

**Short-term** memory (thread-level [persistence](../../concepts/persistence.md)) enables agents to track multi-turn conversations. To add short-term memory:

:::python

```python
# highlight-next-line
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# highlight-next-line
checkpointer = InMemorySaver()

builder = StateGraph(...)
# highlight-next-line
graph = builder.compile(checkpointer=checkpointer)

graph.invoke(
    {"messages": [{"role": "user", "content": "hi! i am Bob"}]},
    # highlight-next-line
    {"configurable": {"thread_id": "1"}},
)
```

:::

:::js

```typescript
import { MemorySaver, StateGraph } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const builder = new StateGraph(...);
const graph = builder.compile({ checkpointer });

await graph.invoke(
  { messages: [{ role: "user", content: "hi! i am Bob" }] },
  { configurable: { thread_id: "1" } }
);
```

:::

### Use in production

In production, use a checkpointer backed by a database:

:::python

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
# highlight-next-line
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    builder = StateGraph(...)
    # highlight-next-line
    graph = builder.compile(checkpointer=checkpointer)
```

:::

:::js

```typescript
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable";
const checkpointer = PostgresSaver.fromConnString(DB_URI);

const builder = new StateGraph(...);
const graph = builder.compile({ checkpointer });
```

:::

??? example "Example: using Postgres checkpointer"

    :::python
    ```
    pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
    ```

    !!! Setup
        You need to call `checkpointer.setup()` the first time you're using Postgres checkpointer

    === "Sync"

        ```python
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        # highlight-next-line
        from langgraph.checkpoint.postgres import PostgresSaver

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
        # highlight-next-line
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # checkpointer.setup()

            def call_model(state: MessagesState):
                response = model.invoke(state["messages"])
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            # highlight-next-line
            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1"
                }
            }

            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()

            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "what's my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()
        ```

    === "Async"

        ```python
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        # highlight-next-line
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
        # highlight-next-line
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # await checkpointer.setup()

            async def call_model(state: MessagesState):
                response = await model.ainvoke(state["messages"])
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            # highlight-next-line
            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1"
                }
            }

            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()

            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "what's my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()
        ```
    :::

    :::js
    ```
    npm install @langchain/langgraph-checkpoint-postgres
    ```

    !!! Setup
        You need to call `checkpointer.setup()` the first time you're using Postgres checkpointer

    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";
    import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";
    import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

    const model = new ChatAnthropic({ model: "claude-3-5-haiku-20241022" });

    const DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable";
    const checkpointer = PostgresSaver.fromConnString(DB_URI);
    // await checkpointer.setup();

    const builder = new StateGraph(MessagesZodState)
      .addNode("call_model", async (state) => {
        const response = await model.invoke(state.messages);
        return { messages: [response] };
      })
      .addEdge(START, "call_model");

    const graph = builder.compile({ checkpointer });

    const config = {
      configurable: {
        thread_id: "1"
      }
    };

    for await (const chunk of await graph.stream(
      { messages: [{ role: "user", content: "hi! I'm bob" }] },
      { ...config, streamMode: "values" }
    )) {
      console.log(chunk.messages.at(-1)?.content);
    }

    for await (const chunk of await graph.stream(
      { messages: [{ role: "user", content: "what's my name?" }] },
      { ...config, streamMode: "values" }
    )) {
      console.log(chunk.messages.at(-1)?.content);
    }
    ```
    :::

:::python
??? example "Example: using [MongoDB](https://pypi.org/project/langgraph-checkpoint-mongodb/) checkpointer"

    ```
    pip install -U pymongo langgraph langgraph-checkpoint-mongodb
    ```

    !!! note "Setup"

        To use the MongoDB checkpointer, you will need a MongoDB cluster. Follow [this guide](https://www.mongodb.com/docs/guides/atlas/cluster/) to create a cluster if you don't already have one.

    === "Sync"

        ```python
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        # highlight-next-line
        from langgraph.checkpoint.mongodb import MongoDBSaver

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "localhost:27017"
        # highlight-next-line
        with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:

            def call_model(state: MessagesState):
                response = model.invoke(state["messages"])
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            # highlight-next-line
            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1"
                }
            }

            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()

            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "what's my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()
        ```

    === "Async"

        ```python
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        # highlight-next-line
        from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "localhost:27017"
        # highlight-next-line
        async with AsyncMongoDBSaver.from_conn_string(DB_URI) as checkpointer:

            async def call_model(state: MessagesState):
                response = await model.ainvoke(state["messages"])
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            # highlight-next-line
            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1"
                }
            }

            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()

            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "what's my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()
        ```

??? example "Example: using [Redis](https://pypi.org/project/langgraph-checkpoint-redis/) checkpointer"

    ```
    pip install -U langgraph langgraph-checkpoint-redis
    ```

    !!! Setup
        You need to call `checkpointer.setup()` the first time you're using Redis checkpointer


    === "Sync"

        ```python
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        # highlight-next-line
        from langgraph.checkpoint.redis import RedisSaver

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "redis://localhost:6379"
        # highlight-next-line
        with RedisSaver.from_conn_string(DB_URI) as checkpointer:
            # checkpointer.setup()

            def call_model(state: MessagesState):
                response = model.invoke(state["messages"])
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            # highlight-next-line
            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1"
                }
            }

            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()

            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "what's my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()
        ```

    === "Async"

        ```python
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        # highlight-next-line
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "redis://localhost:6379"
        # highlight-next-line
        async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
            # await checkpointer.asetup()

            async def call_model(state: MessagesState):
                response = await model.ainvoke(state["messages"])
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            # highlight-next-line
            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1"
                }
            }

            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()

            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "what's my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()
        ```

:::

### Use in subgraphs

If your graph contains [subgraphs](../../concepts/subgraphs.md), you only need to provide the checkpointer when compiling the parent graph. LangGraph will automatically propagate the checkpointer to the child subgraphs.

:::python

```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

class State(TypedDict):
    foo: str

# Subgraph

def subgraph_node_1(state: State):
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
# highlight-next-line
subgraph = subgraph_builder.compile()

# Parent graph

builder = StateGraph(State)
# highlight-next-line
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = InMemorySaver()
# highlight-next-line
graph = builder.compile(checkpointer=checkpointer)
```

:::

:::js

```typescript
import { StateGraph, START, MemorySaver } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({ foo: z.string() });

const subgraphBuilder = new StateGraph(State)
  .addNode("subgraph_node_1", (state) => {
    return { foo: state.foo + "bar" };
  })
  .addEdge(START, "subgraph_node_1");
const subgraph = subgraphBuilder.compile();

const builder = new StateGraph(State)
  .addNode("node_1", subgraph)
  .addEdge(START, "node_1");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });
```

:::

If you want the subgraph to have its own memory, you can compile it with the appropriate checkpointer option. This is useful in [multi-agent](../../concepts/multi_agent.md) systems, if you want agents to keep track of their internal message histories.

:::python

```python
subgraph_builder = StateGraph(...)
# highlight-next-line
subgraph = subgraph_builder.compile(checkpointer=True)
```

:::

:::js

```typescript
const subgraphBuilder = new StateGraph(...);
// highlight-next-line
const subgraph = subgraphBuilder.compile({ checkpointer: true });
```

:::

### Read short-term memory in tools { #read-short-term }

LangGraph allows agents to access their short-term memory (state) inside the tools.

:::python

```python
from typing import Annotated
from langgraph.prebuilt import InjectedState, create_react_agent

class CustomState(AgentState):
    # highlight-next-line
    user_id: str

def get_user_info(
    # highlight-next-line
    state: Annotated[CustomState, InjectedState]
) -> str:
    """Look up user info."""
    # highlight-next-line
    user_id = state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_user_info],
    # highlight-next-line
    state_schema=CustomState,
)

agent.invoke({
    "messages": "look up user information",
    # highlight-next-line
    "user_id": "user_123"
})
```

:::

:::js

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  MessagesZodState,
  LangGraphRunnableConfig,
} from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const CustomState = z.object({
  messages: MessagesZodState.shape.messages,
  userId: z.string(),
});

const getUserInfo = tool(
  async (_, config: LangGraphRunnableConfig) => {
    const userId = config.configurable?.userId;
    return userId === "user_123" ? "User is John Smith" : "Unknown user";
  },
  {
    name: "get_user_info",
    description: "Look up user info.",
    schema: z.object({}),
  }
);

const agent = createReactAgent({
  llm: model,
  tools: [getUserInfo],
  stateSchema: CustomState,
});

await agent.invoke({
  messages: [{ role: "user", content: "look up user information" }],
  userId: "user_123",
});
```

:::

See the [Context](../../agents/context.md) guide for more information.

### Write short-term memory from tools { #write-short-term }

To modify the agent's short-term memory (state) during execution, you can return state updates directly from the tools. This is useful for persisting intermediate results or making information accessible to subsequent tools or prompts.

:::python

```python
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command

class CustomState(AgentState):
    # highlight-next-line
    user_name: str

def update_user_info(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig
) -> Command:
    """Look up and update user info."""
    user_id = config["configurable"].get("user_id")
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    # highlight-next-line
    return Command(update={
        # highlight-next-line
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=tool_call_id
            )
        ]
    })

def greet(
    # highlight-next-line
    state: Annotated[CustomState, InjectedState]
) -> str:
    """Use this to greet the user once you found their info."""
    user_name = state["user_name"]
    return f"Hello {user_name}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[update_user_info, greet],
    # highlight-next-line
    state_schema=CustomState
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}}
)
```

:::

:::js

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  MessagesZodState,
  LangGraphRunnableConfig,
  Command,
} from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const CustomState = z.object({
  messages: MessagesZodState.shape.messages,
  userName: z.string().optional(),
});

const updateUserInfo = tool(
  async (_, config: LangGraphRunnableConfig) => {
    const userId = config.configurable?.userId;
    const name = userId === "user_123" ? "John Smith" : "Unknown user";
    return new Command({
      update: {
        userName: name,
        // update the message history
        messages: [
          {
            role: "tool",
            content: "Successfully looked up user information",
            tool_call_id: config.toolCall?.id,
          },
        ],
      },
    });
  },
  {
    name: "update_user_info",
    description: "Look up and update user info.",
    schema: z.object({}),
  }
);

const greet = tool(
  async (_, config: LangGraphRunnableConfig) => {
    const userName = config.configurable?.userName;
    return `Hello ${userName}!`;
  },
  {
    name: "greet",
    description: "Use this to greet the user once you found their info.",
    schema: z.object({}),
  }
);

const agent = createReactAgent({
  llm: model,
  tools: [updateUserInfo, greet],
  stateSchema: CustomState,
});

await agent.invoke(
  { messages: [{ role: "user", content: "greet the user" }] },
  { configurable: { userId: "user_123" } }
);
```

:::

## Add long-term memory

Use long-term memory to store user-specific or application-specific data across conversations.

:::python

```python
# highlight-next-line
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph

# highlight-next-line
store = InMemoryStore()

builder = StateGraph(...)
# highlight-next-line
graph = builder.compile(store=store)
```

:::

:::js

```typescript
import { InMemoryStore, StateGraph } from "@langchain/langgraph";

const store = new InMemoryStore();

const builder = new StateGraph(...);
const graph = builder.compile({ store });
```

:::

### Use in production

In production, use a store backed by a database:

:::python

```python
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
# highlight-next-line
with PostgresStore.from_conn_string(DB_URI) as store:
    builder = StateGraph(...)
    # highlight-next-line
    graph = builder.compile(store=store)
```

:::

:::js

```typescript
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres";

const DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable";
const store = PostgresStore.fromConnString(DB_URI);

const builder = new StateGraph(...);
const graph = builder.compile({ store });
```

:::

??? example "Example: using Postgres store"

    :::python
    ```
    pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
    ```

    !!! Setup
        You need to call `store.setup()` the first time you're using Postgres store

    === "Sync"

        ```python
        from langchain_core.runnables import RunnableConfig
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        from langgraph.checkpoint.postgres import PostgresSaver
        # highlight-next-line
        from langgraph.store.postgres import PostgresStore
        from langgraph.store.base import BaseStore

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

        with (
            # highlight-next-line
            PostgresStore.from_conn_string(DB_URI) as store,
            PostgresSaver.from_conn_string(DB_URI) as checkpointer,
        ):
            # store.setup()
            # checkpointer.setup()

            def call_model(
                state: MessagesState,
                config: RunnableConfig,
                *,
                # highlight-next-line
                store: BaseStore,
            ):
                user_id = config["configurable"]["user_id"]
                namespace = ("memories", user_id)
                # highlight-next-line
                memories = store.search(namespace, query=str(state["messages"][-1].content))
                info = "\n".join([d.value["data"] for d in memories])
                system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

                # Store new memories if the user asks the model to remember
                last_message = state["messages"][-1]
                if "remember" in last_message.content.lower():
                    memory = "User name is Bob"
                    # highlight-next-line
                    store.put(namespace, str(uuid.uuid4()), {"data": memory})

                response = model.invoke(
                    [{"role": "system", "content": system_msg}] + state["messages"]
                )
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            graph = builder.compile(
                checkpointer=checkpointer,
                # highlight-next-line
                store=store,
            )

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1",
                    # highlight-next-line
                    "user_id": "1",
                }
            }
            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
                # highlight-next-line
                config,
                stream_mode="values",
            ):
                chunk["messages"][-1].pretty_print()

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "2",
                    "user_id": "1",
                }
            }

            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "what is my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values",
            ):
                chunk["messages"][-1].pretty_print()
        ```

    === "Async"

        ```python
        from langchain_core.runnables import RunnableConfig
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        # highlight-next-line
        from langgraph.store.postgres.aio import AsyncPostgresStore
        from langgraph.store.base import BaseStore

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

        async with (
            # highlight-next-line
            AsyncPostgresStore.from_conn_string(DB_URI) as store,
            AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
        ):
            # await store.setup()
            # await checkpointer.setup()

            async def call_model(
                state: MessagesState,
                config: RunnableConfig,
                *,
                # highlight-next-line
                store: BaseStore,
            ):
                user_id = config["configurable"]["user_id"]
                namespace = ("memories", user_id)
                # highlight-next-line
                memories = await store.asearch(namespace, query=str(state["messages"][-1].content))
                info = "\n".join([d.value["data"] for d in memories])
                system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

                # Store new memories if the user asks the model to remember
                last_message = state["messages"][-1]
                if "remember" in last_message.content.lower():
                    memory = "User name is Bob"
                    # highlight-next-line
                    await store.aput(namespace, str(uuid.uuid4()), {"data": memory})

                response = await model.ainvoke(
                    [{"role": "system", "content": system_msg}] + state["messages"]
                )
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            graph = builder.compile(
                checkpointer=checkpointer,
                # highlight-next-line
                store=store,
            )

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1",
                    # highlight-next-line
                    "user_id": "1",
                }
            }
            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
                # highlight-next-line
                config,
                stream_mode="values",
            ):
                chunk["messages"][-1].pretty_print()

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "2",
                    "user_id": "1",
                }
            }

            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "what is my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values",
            ):
                chunk["messages"][-1].pretty_print()
        ```
    :::

    :::js
    ```
    npm install @langchain/langgraph-checkpoint-postgres
    ```

    !!! Setup
        You need to call `store.setup()` the first time you're using Postgres store

    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";
    import { StateGraph, MessagesZodState, START, LangGraphRunnableConfig } from "@langchain/langgraph";
    import { PostgresSaver, PostgresStore } from "@langchain/langgraph-checkpoint-postgres";
    import { z } from "zod";
    import { v4 as uuidv4 } from "uuid";

    const model = new ChatAnthropic({ model: "claude-3-5-haiku-20241022" });

    const DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable";

    const store = PostgresStore.fromConnString(DB_URI);
    const checkpointer = PostgresSaver.fromConnString(DB_URI);
    // await store.setup();
    // await checkpointer.setup();

    const callModel = async (
      state: z.infer<typeof MessagesZodState>,
      config: LangGraphRunnableConfig,
    ) => {
      const userId = config.configurable?.userId;
      const namespace = ["memories", userId];
      const memories = await config.store?.search(namespace, { query: state.messages.at(-1)?.content });
      const info = memories?.map(d => d.value.data).join("\n") || "";
      const systemMsg = `You are a helpful assistant talking to the user. User info: ${info}`;

      // Store new memories if the user asks the model to remember
      const lastMessage = state.messages.at(-1);
      if (lastMessage?.content?.toLowerCase().includes("remember")) {
        const memory = "User name is Bob";
        await config.store?.put(namespace, uuidv4(), { data: memory });
      }

      const response = await model.invoke([
        { role: "system", content: systemMsg },
        ...state.messages
      ]);
      return { messages: [response] };
    };

    const builder = new StateGraph(MessagesZodState)
      .addNode("call_model", callModel)
      .addEdge(START, "call_model");

    const graph = builder.compile({
      checkpointer,
      store,
    });

    const config = {
      configurable: {
        thread_id: "1",
        userId: "1",
      }
    };

    for await (const chunk of await graph.stream(
      { messages: [{ role: "user", content: "Hi! Remember: my name is Bob" }] },
      { ...config, streamMode: "values" }
    )) {
      console.log(chunk.messages.at(-1)?.content);
    }

    const config2 = {
      configurable: {
        thread_id: "2",
        userId: "1",
      }
    };

    for await (const chunk of await graph.stream(
      { messages: [{ role: "user", content: "what is my name?" }] },
      { ...config2, streamMode: "values" }
    )) {
      console.log(chunk.messages.at(-1)?.content);
    }
    ```
    :::

:::python
??? example "Example: using [Redis](https://pypi.org/project/langgraph-checkpoint-redis/) store"

    ```
    pip install -U langgraph langgraph-checkpoint-redis
    ```

    !!! Setup
        You need to call `store.setup()` the first time you're using Redis store


    === "Sync"

        ```python
        from langchain_core.runnables import RunnableConfig
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        from langgraph.checkpoint.redis import RedisSaver
        # highlight-next-line
        from langgraph.store.redis import RedisStore
        from langgraph.store.base import BaseStore

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "redis://localhost:6379"

        with (
            # highlight-next-line
            RedisStore.from_conn_string(DB_URI) as store,
            RedisSaver.from_conn_string(DB_URI) as checkpointer,
        ):
            store.setup()
            checkpointer.setup()

            def call_model(
                state: MessagesState,
                config: RunnableConfig,
                *,
                # highlight-next-line
                store: BaseStore,
            ):
                user_id = config["configurable"]["user_id"]
                namespace = ("memories", user_id)
                # highlight-next-line
                memories = store.search(namespace, query=str(state["messages"][-1].content))
                info = "\n".join([d.value["data"] for d in memories])
                system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

                # Store new memories if the user asks the model to remember
                last_message = state["messages"][-1]
                if "remember" in last_message.content.lower():
                    memory = "User name is Bob"
                    # highlight-next-line
                    store.put(namespace, str(uuid.uuid4()), {"data": memory})

                response = model.invoke(
                    [{"role": "system", "content": system_msg}] + state["messages"]
                )
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            graph = builder.compile(
                checkpointer=checkpointer,
                # highlight-next-line
                store=store,
            )

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1",
                    # highlight-next-line
                    "user_id": "1",
                }
            }
            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
                # highlight-next-line
                config,
                stream_mode="values",
            ):
                chunk["messages"][-1].pretty_print()

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "2",
                    "user_id": "1",
                }
            }

            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": "what is my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values",
            ):
                chunk["messages"][-1].pretty_print()
        ```

    === "Async"

        ```python
        from langchain_core.runnables import RunnableConfig
        from langchain.chat_models import init_chat_model
        from langgraph.graph import StateGraph, MessagesState, START
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        # highlight-next-line
        from langgraph.store.redis.aio import AsyncRedisStore
        from langgraph.store.base import BaseStore

        model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

        DB_URI = "redis://localhost:6379"

        async with (
            # highlight-next-line
            AsyncRedisStore.from_conn_string(DB_URI) as store,
            AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer,
        ):
            # await store.setup()
            # await checkpointer.asetup()

            async def call_model(
                state: MessagesState,
                config: RunnableConfig,
                *,
                # highlight-next-line
                store: BaseStore,
            ):
                user_id = config["configurable"]["user_id"]
                namespace = ("memories", user_id)
                # highlight-next-line
                memories = await store.asearch(namespace, query=str(state["messages"][-1].content))
                info = "\n".join([d.value["data"] for d in memories])
                system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

                # Store new memories if the user asks the model to remember
                last_message = state["messages"][-1]
                if "remember" in last_message.content.lower():
                    memory = "User name is Bob"
                    # highlight-next-line
                    await store.aput(namespace, str(uuid.uuid4()), {"data": memory})

                response = await model.ainvoke(
                    [{"role": "system", "content": system_msg}] + state["messages"]
                )
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            graph = builder.compile(
                checkpointer=checkpointer,
                # highlight-next-line
                store=store,
            )

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "1",
                    # highlight-next-line
                    "user_id": "1",
                }
            }
            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
                # highlight-next-line
                config,
                stream_mode="values",
            ):
                chunk["messages"][-1].pretty_print()

            config = {
                "configurable": {
                    # highlight-next-line
                    "thread_id": "2",
                    "user_id": "1",
                }
            }

            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "what is my name?"}]},
                # highlight-next-line
                config,
                stream_mode="values",
            ):
                chunk["messages"][-1].pretty_print()
        ```

:::

### Read long-term memory in tools { #read-long-term }

:::python

```python title="A tool the agent can use to look up user information"
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

# highlight-next-line
store = InMemoryStore() # (1)!

# highlight-next-line
store.put(  # (2)!
    ("users",),  # (3)!
    "user_123",  # (4)!
    {
        "name": "John Smith",
        "language": "English",
    } # (5)!
)

def get_user_info(config: RunnableConfig) -> str:
    """Look up user info."""
    # Same as that provided to `create_react_agent`
    # highlight-next-line
    store = get_store() # (6)!
    user_id = config["configurable"].get("user_id")
    # highlight-next-line
    user_info = store.get(("users",), user_id) # (7)!
    return str(user_info.value) if user_info else "Unknown user"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_user_info],
    # highlight-next-line
    store=store # (8)!
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}}
)
```

1. The `InMemoryStore` is a store that stores data in memory. In a production setting, you would typically use a database or other persistent storage. Please review the [store documentation](../../reference/store.md) for more options. If you're deploying with **LangGraph Platform**, the platform will provide a production-ready store for you.
2. For this example, we write some sample data to the store using the `put` method. Please see the @[BaseStore.put] API reference for more details.
3. The first argument is the namespace. This is used to group related data together. In this case, we are using the `users` namespace to group user data.
4. A key within the namespace. This example uses a user ID for the key.
5. The data that we want to store for the given user.
6. The `get_store` function is used to access the store. You can call it from anywhere in your code, including tools and prompts. This function returns the store that was passed to the agent when it was created.
7. The `get` method is used to retrieve data from the store. The first argument is the namespace, and the second argument is the key. This will return a `StoreValue` object, which contains the value and metadata about the value.
8. The `store` is passed to the agent. This enables the agent to access the store when running tools. You can also use the `get_store` function to access the store from anywhere in your code.
   :::

:::js

```typescript title="A tool the agent can use to look up user information"
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { LangGraphRunnableConfig, InMemoryStore } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const store = new InMemoryStore(); // (1)!

await store.put(
  // (2)!
  ["users"], // (3)!
  "user_123", // (4)!
  {
    name: "John Smith",
    language: "English",
  } // (5)!
);

const getUserInfo = tool(
  async (_, config: LangGraphRunnableConfig) => {
    /**Look up user info.*/
    // Same as that provided to `createReactAgent`
    const store = config.store; // (6)!
    const userId = config.configurable?.userId;
    const userInfo = await store?.get(["users"], userId); // (7)!
    return userInfo?.value ? JSON.stringify(userInfo.value) : "Unknown user";
  },
  {
    name: "get_user_info",
    description: "Look up user info.",
    schema: z.object({}),
  }
);

const agent = createReactAgent({
  llm: model,
  tools: [getUserInfo],
  store, // (8)!
});

// Run the agent
await agent.invoke(
  { messages: [{ role: "user", content: "look up user information" }] },
  { configurable: { userId: "user_123" } }
);
```

1. The `InMemoryStore` is a store that stores data in memory. In a production setting, you would typically use a database or other persistent storage. Please review the [store documentation](../../reference/store.md) for more options. If you're deploying with **LangGraph Platform**, the platform will provide a production-ready store for you.
2. For this example, we write some sample data to the store using the `put` method. Please see the @[BaseStore.put] API reference for more details.
3. The first argument is the namespace. This is used to group related data together. In this case, we are using the `users` namespace to group user data.
4. A key within the namespace. This example uses a user ID for the key.
5. The data that we want to store for the given user.
6. The store is accessible through the config. You can call it from anywhere in your code, including tools and prompts. This function returns the store that was passed to the agent when it was created.
7. The `get` method is used to retrieve data from the store. The first argument is the namespace, and the second argument is the key. This will return a `StoreValue` object, which contains the value and metadata about the value.
8. The `store` is passed to the agent. This enables the agent to access the store when running tools. You can also use the store from the config to access it from anywhere in your code.
   :::

### Write long-term memory from tools { #write-long-term }

:::python

```python title="Example of a tool that updates user information"
from typing_extensions import TypedDict

from langgraph.config import get_store
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore() # (1)!

class UserInfo(TypedDict): # (2)!
    name: str

def save_user_info(user_info: UserInfo, config: RunnableConfig) -> str: # (3)!
    """Save user info."""
    # Same as that provided to `create_react_agent`
    # highlight-next-line
    store = get_store() # (4)!
    user_id = config["configurable"].get("user_id")
    # highlight-next-line
    store.put(("users",), user_id, user_info) # (5)!
    return "Successfully saved user info."

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[save_user_info],
    # highlight-next-line
    store=store
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}} # (6)!
)

# You can access the store directly to get the value
store.get(("users",), "user_123").value
```

1. The `InMemoryStore` is a store that stores data in memory. In a production setting, you would typically use a database or other persistent storage. Please review the [store documentation](../../reference/store.md) for more options. If you're deploying with **LangGraph Platform**, the platform will provide a production-ready store for you.
2. The `UserInfo` class is a `TypedDict` that defines the structure of the user information. The LLM will use this to format the response according to the schema.
3. The `save_user_info` function is a tool that allows an agent to update user information. This could be useful for a chat application where the user wants to update their profile information.
4. The `get_store` function is used to access the store. You can call it from anywhere in your code, including tools and prompts. This function returns the store that was passed to the agent when it was created.
5. The `put` method is used to store data in the store. The first argument is the namespace, and the second argument is the key. This will store the user information in the store.
6. The `user_id` is passed in the config. This is used to identify the user whose information is being updated.
   :::

:::js

```typescript title="Example of a tool that updates user information"
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { LangGraphRunnableConfig, InMemoryStore } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const store = new InMemoryStore(); // (1)!

const UserInfo = z.object({
  // (2)!
  name: z.string(),
});

const saveUserInfo = tool(
  async (
    userInfo: z.infer<typeof UserInfo>,
    config: LangGraphRunnableConfig
  ) => {
    // (3)!
    /**Save user info.*/
    // Same as that provided to `createReactAgent`
    const store = config.store; // (4)!
    const userId = config.configurable?.userId;
    await store?.put(["users"], userId, userInfo); // (5)!
    return "Successfully saved user info.";
  },
  {
    name: "save_user_info",
    description: "Save user info.",
    schema: UserInfo,
  }
);

const agent = createReactAgent({
  llm: model,
  tools: [saveUserInfo],
  store,
});

// Run the agent
await agent.invoke(
  { messages: [{ role: "user", content: "My name is John Smith" }] },
  { configurable: { userId: "user_123" } } // (6)!
);

// You can access the store directly to get the value
const result = await store.get(["users"], "user_123");
console.log(result?.value);
```

1. The `InMemoryStore` is a store that stores data in memory. In a production setting, you would typically use a database or other persistent storage. Please review the [store documentation](../../reference/store.md) for more options. If you're deploying with **LangGraph Platform**, the platform will provide a production-ready store for you.
2. The `UserInfo` schema defines the structure of the user information. The LLM will use this to format the response according to the schema.
3. The `saveUserInfo` function is a tool that allows an agent to update user information. This could be useful for a chat application where the user wants to update their profile information.
4. The store is accessible through the config. You can call it from anywhere in your code, including tools and prompts. This function returns the store that was passed to the agent when it was created.
5. The `put` method is used to store data in the store. The first argument is the namespace, and the second argument is the key. This will store the user information in the store.
6. The `userId` is passed in the config. This is used to identify the user whose information is being updated.
   :::

### Use semantic search

Enable semantic search in your graph's memory store to let graph agents search for items in the store by semantic similarity.

:::python

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

# Create store with semantic search enabled
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})

items = store.search(
    ("user_123", "memories"), query="I'm hungry", limit=1
)
```

:::

:::js

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { InMemoryStore } from "@langchain/langgraph";

// Create store with semantic search enabled
const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
const store = new InMemoryStore({
  index: {
    embeddings,
    dims: 1536,
  },
});

await store.put(["user_123", "memories"], "1", { text: "I love pizza" });
await store.put(["user_123", "memories"], "2", { text: "I am a plumber" });

const items = await store.search(["user_123", "memories"], {
  query: "I'm hungry",
  limit: 1,
});
```

:::

??? example "Long-term memory with semantic search"

    :::python
    ```python
    from typing import Optional

    from langchain.embeddings import init_embeddings
    from langchain.chat_models import init_chat_model
    from langgraph.store.base import BaseStore
    from langgraph.store.memory import InMemoryStore
    from langgraph.graph import START, MessagesState, StateGraph

    llm = init_chat_model("openai:gpt-4o-mini")

    # Create store with semantic search enabled
    embeddings = init_embeddings("openai:text-embedding-3-small")
    store = InMemoryStore(
        index={
            "embed": embeddings,
            "dims": 1536,
        }
    )

    store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
    store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})

    def chat(state, *, store: BaseStore):
        # Search based on user's last message
        items = store.search(
            ("user_123", "memories"), query=state["messages"][-1].content, limit=2
        )
        memories = "\n".join(item.value["text"] for item in items)
        memories = f"## Memories of user\n{memories}" if memories else ""
        response = llm.invoke(
            [
                {"role": "system", "content": f"You are a helpful assistant.\n{memories}"},
                *state["messages"],
            ]
        )
        return {"messages": [response]}


    builder = StateGraph(MessagesState)
    builder.add_node(chat)
    builder.add_edge(START, "chat")
    graph = builder.compile(store=store)

    for message, metadata in graph.stream(
        input={"messages": [{"role": "user", "content": "I'm hungry"}]},
        stream_mode="messages",
    ):
        print(message.content, end="")
    ```
    :::

    :::js
    ```typescript
    import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
    import { StateGraph, START, MessagesZodState, InMemoryStore } from "@langchain/langgraph";
    import { z } from "zod";

    const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

    // Create store with semantic search enabled
    const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
    const store = new InMemoryStore({
      index: {
        embeddings,
        dims: 1536,
      }
    });

    await store.put(["user_123", "memories"], "1", { text: "I love pizza" });
    await store.put(["user_123", "memories"], "2", { text: "I am a plumber" });

    const chat = async (state: z.infer<typeof MessagesZodState>, config) => {
      // Search based on user's last message
      const items = await config.store.search(
        ["user_123", "memories"],
        { query: state.messages.at(-1)?.content, limit: 2 }
      );
      const memories = items.map(item => item.value.text).join("\n");
      const memoriesText = memories ? `## Memories of user\n${memories}` : "";

      const response = await llm.invoke([
        { role: "system", content: `You are a helpful assistant.\n${memoriesText}` },
        ...state.messages,
      ]);

      return { messages: [response] };
    };

    const builder = new StateGraph(MessagesZodState)
      .addNode("chat", chat)
      .addEdge(START, "chat");
    const graph = builder.compile({ store });

    for await (const [message, metadata] of await graph.stream(
      { messages: [{ role: "user", content: "I'm hungry" }] },
      { streamMode: "messages" }
    )) {
      if (message.content) {
        console.log(message.content);
      }
    }
    ```
    :::

See [this guide](../../cloud/deployment/semantic_search.md) for more information on how to use semantic search with LangGraph memory store.

## Manage short-term memory

With [short-term memory](#add-short-term-memory) enabled, long conversations can exceed the LLM's context window. Common solutions are:

- [Trim messages](#trim-messages): Remove first or last N messages (before calling LLM)
- [Delete messages](#delete-messages) from LangGraph state permanently
- [Summarize messages](#summarize-messages): Summarize earlier messages in the history and replace them with a summary
- [Manage checkpoints](#manage-checkpoints) to store and retrieve message history
- Custom strategies (e.g., message filtering, etc.)

This allows the agent to keep track of the conversation without exceeding the LLM's context window.

### Trim messages

Most LLMs have a maximum supported context window (denominated in tokens). One way to decide when to truncate messages is to count the tokens in the message history and truncate whenever it approaches that limit. If you're using LangChain, you can use the trim messages utility and specify the number of tokens to keep from the list, as well as the `strategy` (e.g., keep the last `maxTokens`) to use for handling the boundary.

=== "In an agent"

    :::python
    To trim message history in an agent, use @[`pre_model_hook`][create_react_agent] with the [`trim_messages`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html) function:

    ```python
    # highlight-next-line
    from langchain_core.messages.utils import (
        # highlight-next-line
        trim_messages,
        # highlight-next-line
        count_tokens_approximately
    # highlight-next-line
    )
    from langgraph.prebuilt import create_react_agent

    # This function will be called every time before the node that calls LLM
    def pre_model_hook(state):
        trimmed_messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=384,
            start_on="human",
            end_on=("human", "tool"),
        )
        # highlight-next-line
        return {"llm_input_messages": trimmed_messages}

    checkpointer = InMemorySaver()
    agent = create_react_agent(
        model,
        tools,
        # highlight-next-line
        pre_model_hook=pre_model_hook,
        checkpointer=checkpointer,
    )
    ```
    :::

    :::js
    To trim message history in an agent, use `stateModifier` with the [`trimMessages`](https://js.langchain.com/docs/how_to/trim_messages/) function:

    ```typescript
    import { trimMessages } from "@langchain/core/messages";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";

    // This function will be called every time before the node that calls LLM
    const stateModifier = async (state) => {
      return trimMessages(state.messages, {
        strategy: "last",
        maxTokens: 384,
        startOn: "human",
        endOn: ["human", "tool"],
      });
    };

    const checkpointer = new MemorySaver();
    const agent = createReactAgent({
      llm: model,
      tools,
      stateModifier,
      checkpointer,
    });
    ```
    :::

=== "In a workflow"

    :::python
    To trim message history, use the [`trim_messages`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html) function:

    ```python
    # highlight-next-line
    from langchain_core.messages.utils import (
        # highlight-next-line
        trim_messages,
        # highlight-next-line
        count_tokens_approximately
    # highlight-next-line
    )

    def call_model(state: MessagesState):
        # highlight-next-line
        messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=128,
            start_on="human",
            end_on=("human", "tool"),
        )
        response = model.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    ...
    ```
    :::

    :::js
    To trim message history, use the [`trimMessages`](https://js.langchain.com/docs/how_to/trim_messages/) function:

    ```typescript
    import { trimMessages } from "@langchain/core/messages";

    const callModel = async (state: z.infer<typeof MessagesZodState>) => {
      const messages = trimMessages(state.messages, {
        strategy: "last",
        maxTokens: 128,
        startOn: "human",
        endOn: ["human", "tool"],
      });
      const response = await model.invoke(messages);
      return { messages: [response] };
    };

    const builder = new StateGraph(MessagesZodState)
      .addNode("call_model", callModel);
    // ...
    ```
    :::

??? example "Full example: trim messages"

    :::python
    ```python
    # highlight-next-line
    from langchain_core.messages.utils import (
        # highlight-next-line
        trim_messages,
        # highlight-next-line
        count_tokens_approximately
    # highlight-next-line
    )
    from langchain.chat_models import init_chat_model
    from langgraph.graph import StateGraph, START, MessagesState

    model = init_chat_model("anthropic:claude-3-7-sonnet-latest")
    summarization_model = model.bind(max_tokens=128)

    def call_model(state: MessagesState):
        # highlight-next-line
        messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=128,
            start_on="human",
            end_on=("human", "tool"),
        )
        response = model.invoke(messages)
        return {"messages": [response]}

    checkpointer = InMemorySaver()
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": "hi, my name is bob"}, config)
    graph.invoke({"messages": "write a short poem about cats"}, config)
    graph.invoke({"messages": "now do the same but for dogs"}, config)
    final_response = graph.invoke({"messages": "what's my name?"}, config)

    final_response["messages"][-1].pretty_print()
    ```

    ```
    ================================== Ai Message ==================================

    Your name is Bob, as you mentioned when you first introduced yourself.
    ```
    :::

    :::js
    ```typescript
    import { trimMessages } from "@langchain/core/messages";
    import { ChatAnthropic } from "@langchain/anthropic";
    import { StateGraph, START, MessagesZodState, MemorySaver } from "@langchain/langgraph";
    import { z } from "zod";

    const model = new ChatAnthropic({ model: "claude-3-5-sonnet-20241022" });

    const callModel = async (state: z.infer<typeof MessagesZodState>) => {
      const messages = trimMessages(state.messages, {
        strategy: "last",
        maxTokens: 128,
        startOn: "human",
        endOn: ["human", "tool"],
      });
      const response = await model.invoke(messages);
      return { messages: [response] };
    };

    const checkpointer = new MemorySaver();
    const builder = new StateGraph(MessagesZodState)
      .addNode("call_model", callModel)
      .addEdge(START, "call_model");
    const graph = builder.compile({ checkpointer });

    const config = { configurable: { thread_id: "1" } };
    await graph.invoke({ messages: [{ role: "user", content: "hi, my name is bob" }] }, config);
    await graph.invoke({ messages: [{ role: "user", content: "write a short poem about cats" }] }, config);
    await graph.invoke({ messages: [{ role: "user", content: "now do the same but for dogs" }] }, config);
    const finalResponse = await graph.invoke({ messages: [{ role: "user", content: "what's my name?" }] }, config);

    console.log(finalResponse.messages.at(-1)?.content);
    ```

    ```
    Your name is Bob, as you mentioned when you first introduced yourself.
    ```
    :::

### Delete messages

You can delete messages from the graph state to manage the message history. This is useful when you want to remove specific messages or clear the entire message history.

:::python
To delete messages from the graph state, you can use the `RemoveMessage`. For `RemoveMessage` to work, you need to use a state key with @[`add_messages`][add_messages] [reducer](../../concepts/low_level.md#reducers), like [`MessagesState`](../../concepts/low_level.md#messagesstate).

To remove specific messages:

```python
# highlight-next-line
from langchain_core.messages import RemoveMessage

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        # highlight-next-line
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
```

To remove **all** messages:

```python
# highlight-next-line
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_messages(state):
    # highlight-next-line
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

:::

:::js
To delete messages from the graph state, you can use the `RemoveMessage`. For `RemoveMessage` to work, you need to use a state key with @[`messagesStateReducer`][messagesStateReducer] [reducer](../../concepts/low_level.md#reducers), like `MessagesZodState`.

To remove specific messages:

```typescript
import { RemoveMessage } from "@langchain/core/messages";

const deleteMessages = (state) => {
  const messages = state.messages;
  if (messages.length > 2) {
    // remove the earliest two messages
    return {
      messages: messages
        .slice(0, 2)
        .map((m) => new RemoveMessage({ id: m.id })),
    };
  }
};
```

:::

!!! warning

    When deleting messages, **make sure** that the resulting message history is valid. Check the limitations of the LLM provider you're using. For example:

    * some providers expect message history to start with a `user` message
    * most providers require `assistant` messages with tool calls to be followed by corresponding `tool` result messages.

??? example "Full example: delete messages"

    :::python
    ```python
    # highlight-next-line
    from langchain_core.messages import RemoveMessage

    def delete_messages(state):
        messages = state["messages"]
        if len(messages) > 2:
            # remove the earliest two messages
            # highlight-next-line
            return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_sequence([call_model, delete_messages])
    builder.add_edge(START, "call_model")

    checkpointer = InMemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    for event in app.stream(
        {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
        config,
        stream_mode="values"
    ):
        print([(message.type, message.content) for message in event["messages"]])

    for event in app.stream(
        {"messages": [{"role": "user", "content": "what's my name?"}]},
        config,
        stream_mode="values"
    ):
        print([(message.type, message.content) for message in event["messages"]])
    ```

    ```
    [('human', "hi! I'm bob")]
    [('human', "hi! I'm bob"), ('ai', 'Hi Bob! How are you doing today? Is there anything I can help you with?')]
    [('human', "hi! I'm bob"), ('ai', 'Hi Bob! How are you doing today? Is there anything I can help you with?'), ('human', "what's my name?")]
    [('human', "hi! I'm bob"), ('ai', 'Hi Bob! How are you doing today? Is there anything I can help you with?'), ('human', "what's my name?"), ('ai', 'Your name is Bob.')]
    [('human', "what's my name?"), ('ai', 'Your name is Bob.')]
    ```
    :::

    :::js
    ```typescript
    import { RemoveMessage } from "@langchain/core/messages";
    import { ChatAnthropic } from "@langchain/anthropic";
    import { StateGraph, START, MessagesZodState, MemorySaver } from "@langchain/langgraph";
    import { z } from "zod";

    const model = new ChatAnthropic({ model: "claude-3-5-sonnet-20241022" });

    const deleteMessages = (state: z.infer<typeof MessagesZodState>) => {
      const messages = state.messages;
      if (messages.length > 2) {
        // remove the earliest two messages
        return { messages: messages.slice(0, 2).map(m => new RemoveMessage({ id: m.id })) };
      }
      return {};
    };

    const callModel = async (state: z.infer<typeof MessagesZodState>) => {
      const response = await model.invoke(state.messages);
      return { messages: [response] };
    };

    const builder = new StateGraph(MessagesZodState)
      .addNode("call_model", callModel)
      .addNode("delete_messages", deleteMessages)
      .addEdge(START, "call_model")
      .addEdge("call_model", "delete_messages");

    const checkpointer = new MemorySaver();
    const app = builder.compile({ checkpointer });

    const config = { configurable: { thread_id: "1" } };

    for await (const event of await app.stream(
      { messages: [{ role: "user", content: "hi! I'm bob" }] },
      { ...config, streamMode: "values" }
    )) {
      console.log(event.messages.map(message => [message.getType(), message.content]));
    }

    for await (const event of await app.stream(
      { messages: [{ role: "user", content: "what's my name?" }] },
      { ...config, streamMode: "values" }
    )) {
      console.log(event.messages.map(message => [message.getType(), message.content]));
    }
    ```

    ```
    [['human', "hi! I'm bob"]]
    [['human', "hi! I'm bob"], ['ai', 'Hi Bob! How are you doing today? Is there anything I can help you with?']]
    [['human', "hi! I'm bob"], ['ai', 'Hi Bob! How are you doing today? Is there anything I can help you with?'], ['human', "what's my name?"]]
    [['human', "hi! I'm bob"], ['ai', 'Hi Bob! How are you doing today? Is there anything I can help you with?'], ['human', "what's my name?"], ['ai', 'Your name is Bob.']]
    [['human', "what's my name?"], ['ai', 'Your name is Bob.']]
    ```
    :::

### Summarize messages

The problem with trimming or removing messages, as shown above, is that you may lose information from culling of the message queue. Because of this, some applications benefit from a more sophisticated approach of summarizing the message history using a chat model.

![](../../concepts/img/memory/summary.png)

=== "In an agent"

    :::python
    To summarize message history in an agent, use @[`pre_model_hook`][create_react_agent] with a prebuilt [`SummarizationNode`](https://langchain-ai.github.io/langmem/reference/short_term/#langmem.short_term.SummarizationNode) abstraction:

    ```python
    from langchain_anthropic import ChatAnthropic
    from langmem.short_term import SummarizationNode, RunningSummary
    from langchain_core.messages.utils import count_tokens_approximately
    from langgraph.prebuilt import create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from langgraph.checkpoint.memory import InMemorySaver
    from typing import Any

    model = ChatAnthropic(model="claude-3-7-sonnet-latest")

    summarization_node = SummarizationNode( # (1)!
        token_counter=count_tokens_approximately,
        model=model,
        max_tokens=384,
        max_summary_tokens=128,
        output_messages_key="llm_input_messages",
    )

    class State(AgentState):
        # NOTE: we're adding this key to keep track of previous summary information
        # to make sure we're not summarizing on every LLM call
        # highlight-next-line
        context: dict[str, RunningSummary]  # (2)!


    checkpointer = InMemorySaver() # (3)!

    agent = create_react_agent(
        model=model,
        tools=tools,
        # highlight-next-line
        pre_model_hook=summarization_node, # (4)!
        # highlight-next-line
        state_schema=State, # (5)!
        checkpointer=checkpointer,
    )
    ```

    1. The `InMemorySaver` is a checkpointer that stores the agent's state in memory. In a production setting, you would typically use a database or other persistent storage. Please review the [checkpointer documentation](../../reference/checkpoints.md) for more options. If you're deploying with **LangGraph Platform**, the platform will provide a production-ready checkpointer for you.
    2. The `context` key is added to the agent's state. The key contains book-keeping information for the summarization node. It is used to keep track of the last summary information and ensure that the agent doesn't summarize on every LLM call, which can be inefficient.
    3. The `checkpointer` is passed to the agent. This enables the agent to persist its state across invocations.
    4. The `pre_model_hook` is set to the `SummarizationNode`. This node will summarize the message history before sending it to the LLM. The summarization node will automatically handle the summarization process and update the agent's state with the new summary. You can replace this with a custom implementation if you prefer. Please see the @[create_react_agent][create_react_agent] API reference for more details.
    5. The `state_schema` is set to the `State` class, which is the custom state that contains an extra `context` key.
    :::

=== "In a workflow"

    :::python
    Prompting and orchestration logic can be used to summarize the message history. For example, in LangGraph you can extend the [`MessagesState`](../../concepts/low_level.md#working-with-messages-in-graph-state) to include a `summary` key:

    ```python
    from langgraph.graph import MessagesState
    class State(MessagesState):
        summary: str
    ```

    Then, you can generate a summary of the chat history, using any existing summary as context for the next summary. This `summarize_conversation` node can be called after some number of messages have accumulated in the `messages` state key.

    ```python
    def summarize_conversation(state: State):

        # First, we get any existing summary
        summary = state.get("summary", "")

        # Create our summarization prompt
        if summary:

            # A summary already exists
            summary_message = (
                f"This is a summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )

        else:
            summary_message = "Create a summary of the conversation above:"

        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = model.invoke(messages)

        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}
    ```
    :::

    :::js
    Prompting and orchestration logic can be used to summarize the message history. For example, in LangGraph you can extend the [`MessagesZodState`](../../concepts/low_level.md#working-with-messages-in-graph-state) to include a `summary` key:

    ```typescript
    import { MessagesZodState } from "@langchain/langgraph";
    import { z } from "zod";

    const State = MessagesZodState.merge(z.object({
      summary: z.string().optional(),
    }));
    ```

    Then, you can generate a summary of the chat history, using any existing summary as context for the next summary. This `summarizeConversation` node can be called after some number of messages have accumulated in the `messages` state key.

    ```typescript
    import { RemoveMessage, HumanMessage } from "@langchain/core/messages";

    const summarizeConversation = async (state: z.infer<typeof State>) => {
      // First, we get any existing summary
      const summary = state.summary || "";

      // Create our summarization prompt
      let summaryMessage: string;
      if (summary) {
        // A summary already exists
        summaryMessage =
          `This is a summary of the conversation to date: ${summary}\n\n` +
          "Extend the summary by taking into account the new messages above:";
      } else {
        summaryMessage = "Create a summary of the conversation above:";
      }

      // Add prompt to our history
      const messages = [
        ...state.messages,
        new HumanMessage({ content: summaryMessage })
      ];
      const response = await model.invoke(messages);

      // Delete all but the 2 most recent messages
      const deleteMessages = state.messages
        .slice(0, -2)
        .map(m => new RemoveMessage({ id: m.id }));

      return {
        summary: response.content,
        messages: deleteMessages
      };
    };
    ```
    :::

??? example "Full example: summarize messages"

    :::python
    ```python
    from typing import Any, TypedDict

    from langchain.chat_models import init_chat_model
    from langchain_core.messages import AnyMessage
    from langchain_core.messages.utils import count_tokens_approximately
    from langgraph.graph import StateGraph, START, MessagesState
    from langgraph.checkpoint.memory import InMemorySaver
    # highlight-next-line
    from langmem.short_term import SummarizationNode, RunningSummary

    model = init_chat_model("anthropic:claude-3-7-sonnet-latest")
    summarization_model = model.bind(max_tokens=128)

    class State(MessagesState):
        # highlight-next-line
        context: dict[str, RunningSummary]  # (1)!

    class LLMInputState(TypedDict):  # (2)!
        summarized_messages: list[AnyMessage]
        context: dict[str, RunningSummary]

    # highlight-next-line
    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=summarization_model,
        max_tokens=256,
        max_tokens_before_summary=256,
        max_summary_tokens=128,
    )

    # highlight-next-line
    def call_model(state: LLMInputState):  # (3)!
        response = model.invoke(state["summarized_messages"])
        return {"messages": [response]}

    checkpointer = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node(call_model)
    # highlight-next-line
    builder.add_node("summarize", summarization_node)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "call_model")
    graph = builder.compile(checkpointer=checkpointer)

    # Invoke the graph
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": "hi, my name is bob"}, config)
    graph.invoke({"messages": "write a short poem about cats"}, config)
    graph.invoke({"messages": "now do the same but for dogs"}, config)
    final_response = graph.invoke({"messages": "what's my name?"}, config)

    final_response["messages"][-1].pretty_print()
    print("\nSummary:", final_response["context"]["running_summary"].summary)
    ```

    1. We will keep track of our running summary in the `context` field
    (expected by the `SummarizationNode`).
    2. Define private state that will be used only for filtering
    the inputs to `call_model` node.
    3. We're passing a private input state here to isolate the messages returned by the summarization node

    ```
    ================================== Ai Message ==================================

    From our conversation, I can see that you introduced yourself as Bob. That's the name you shared with me when we began talking.

    Summary: In this conversation, I was introduced to Bob, who then asked me to write a poem about cats. I composed a poem titled "The Mystery of Cats" that captured cats' graceful movements, independent nature, and their special relationship with humans. Bob then requested a similar poem about dogs, so I wrote "The Joy of Dogs," which highlighted dogs' loyalty, enthusiasm, and loving companionship. Both poems were written in a similar style but emphasized the distinct characteristics that make each pet special.
    ```
    :::

    :::js
    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";
    import {
      SystemMessage,
      HumanMessage,
      RemoveMessage,
      type BaseMessage
    } from "@langchain/core/messages";
    import {
      MessagesZodState,
      StateGraph,
      START,
      END,
      MemorySaver,
    } from "@langchain/langgraph";
    import { z } from "zod";
    import { v4 as uuidv4 } from "uuid";

    const memory = new MemorySaver();

    // We will add a `summary` attribute (in addition to `messages` key,
    // which MessagesZodState already has)
    const GraphState = z.object({
      messages: MessagesZodState.shape.messages,
      summary: z.string().default(""),
    });

    // We will use this model for both the conversation and the summarization
    const model = new ChatAnthropic({ model: "claude-3-haiku-20240307" });

    // Define the logic to call the model
    const callModel = async (state: z.infer<typeof GraphState>) => {
      // If a summary exists, we add this in as a system message
      const { summary } = state;
      let { messages } = state;
      if (summary) {
        const systemMessage = new SystemMessage({
          id: uuidv4(),
          content: `Summary of conversation earlier: ${summary}`,
        });
        messages = [systemMessage, ...messages];
      }
      const response = await model.invoke(messages);
      // We return an object, because this will get added to the existing state
      return { messages: [response] };
    };

    // We now define the logic for determining whether to end or summarize the conversation
    const shouldContinue = (state: z.infer<typeof GraphState>) => {
      const messages = state.messages;
      // If there are more than six messages, then we summarize the conversation
      if (messages.length > 6) {
        return "summarize_conversation";
      }
      // Otherwise we can just end
      return END;
    };

    const summarizeConversation = async (state: z.infer<typeof GraphState>) => {
      // First, we summarize the conversation
      const { summary, messages } = state;
      let summaryMessage: string;
      if (summary) {
        // If a summary already exists, we use a different system prompt
        // to summarize it than if one didn't
        summaryMessage =
          `This is summary of the conversation to date: ${summary}\n\n` +
          "Extend the summary by taking into account the new messages above:";
      } else {
        summaryMessage = "Create a summary of the conversation above:";
      }

      const allMessages = [
        ...messages,
        new HumanMessage({ id: uuidv4(), content: summaryMessage }),
      ];

      const response = await model.invoke(allMessages);

      // We now need to delete messages that we no longer want to show up
      // I will delete all but the last two messages, but you can change this
      const deleteMessages = messages
        .slice(0, -2)
        .map((m) => new RemoveMessage({ id: m.id! }));

      if (typeof response.content !== "string") {
        throw new Error("Expected a string response from the model");
      }

      return { summary: response.content, messages: deleteMessages };
    };

    // Define a new graph
    const workflow = new StateGraph(GraphState)
      // Define the conversation node and the summarize node
      .addNode("conversation", callModel)
      .addNode("summarize_conversation", summarizeConversation)
      // Set the entrypoint as conversation
      .addEdge(START, "conversation")
      // We now add a conditional edge
      .addConditionalEdges(
        // First, we define the start node. We use `conversation`.
        // This means these are the edges taken after the `conversation` node is called.
        "conversation",
        // Next, we pass in the function that will determine which node is called next.
        shouldContinue,
      )
      // We now add a normal edge from `summarize_conversation` to END.
      // This means that after `summarize_conversation` is called, we end.
      .addEdge("summarize_conversation", END);

    // Finally, we compile it!
    const app = workflow.compile({ checkpointer: memory });
    ```
    :::

### Manage checkpoints

You can view and delete the information stored by the checkpointer.

#### View thread state (checkpoint)

:::python
=== "Graph/Functional API"

    ```python
    config = {
        "configurable": {
            # highlight-next-line
            "thread_id": "1",
            # optionally provide an ID for a specific checkpoint,
            # otherwise the latest checkpoint is shown
            # highlight-next-line
            # "checkpoint_id": "1f029ca3-1f5b-6704-8004-820c16b69a5a"

        }
    }
    # highlight-next-line
    graph.get_state(config)
    ```

    ```
    StateSnapshot(
        values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today?), HumanMessage(content="what's my name?"), AIMessage(content='Your name is Bob.')]}, next=(),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1f5b-6704-8004-820c16b69a5a'}},
        metadata={
            'source': 'loop',
            'writes': {'call_model': {'messages': AIMessage(content='Your name is Bob.')}},
            'step': 4,
            'parents': {},
            'thread_id': '1'
        },
        created_at='2025-05-05T16:01:24.680462+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
        tasks=(),
        interrupts=()
    )
    ```

=== "Checkpointer API"

    ```python
    config = {
        "configurable": {
            # highlight-next-line
            "thread_id": "1",
            # optionally provide an ID for a specific checkpoint,
            # otherwise the latest checkpoint is shown
            # highlight-next-line
            # "checkpoint_id": "1f029ca3-1f5b-6704-8004-820c16b69a5a"

        }
    }
    # highlight-next-line
    checkpointer.get_tuple(config)
    ```

    ```
    CheckpointTuple(
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1f5b-6704-8004-820c16b69a5a'}},
        checkpoint={
            'v': 3,
            'ts': '2025-05-05T16:01:24.680462+00:00',
            'id': '1f029ca3-1f5b-6704-8004-820c16b69a5a',
            'channel_versions': {'__start__': '00000000000000000000000000000005.0.5290678567601859', 'messages': '00000000000000000000000000000006.0.3205149138784782', 'branch:to:call_model': '00000000000000000000000000000006.0.14611156755133758'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000004.0.5736472536395331'}, 'call_model': {'branch:to:call_model': '00000000000000000000000000000005.0.1410174088651449'}},
            'channel_values': {'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today?), HumanMessage(content="what's my name?"), AIMessage(content='Your name is Bob.')]},
        },
        metadata={
            'source': 'loop',
            'writes': {'call_model': {'messages': AIMessage(content='Your name is Bob.')}},
            'step': 4,
            'parents': {},
            'thread_id': '1'
        },
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
        pending_writes=[]
    )
    ```

:::

:::js

```typescript
const config = {
  configurable: {
    thread_id: "1",
    // optionally provide an ID for a specific checkpoint,
    // otherwise the latest checkpoint is shown
    // checkpoint_id: "1f029ca3-1f5b-6704-8004-820c16b69a5a"
  },
};
await graph.getState(config);
```

```
{
  values: { messages: [HumanMessage(...), AIMessage(...), HumanMessage(...), AIMessage(...)] },
  next: [],
  config: { configurable: { thread_id: '1', checkpoint_ns: '', checkpoint_id: '1f029ca3-1f5b-6704-8004-820c16b69a5a' } },
  metadata: {
    source: 'loop',
    writes: { call_model: { messages: AIMessage(...) } },
    step: 4,
    parents: {},
    thread_id: '1'
  },
  createdAt: '2025-05-05T16:01:24.680462+00:00',
  parentConfig: { configurable: { thread_id: '1', checkpoint_ns: '', checkpoint_id: '1f029ca3-1790-6b0a-8003-baf965b6a38f' } },
  tasks: [],
  interrupts: []
}
```

:::

#### View the history of the thread (checkpoints)

:::python
=== "Graph/Functional API"

    ```python
    config = {
        "configurable": {
            # highlight-next-line
            "thread_id": "1"
        }
    }
    # highlight-next-line
    list(graph.get_state_history(config))
    ```

    ```
    [
        StateSnapshot(
            values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?'), HumanMessage(content="what's my name?"), AIMessage(content='Your name is Bob.')]},
            next=(),
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1f5b-6704-8004-820c16b69a5a'}},
            metadata={'source': 'loop', 'writes': {'call_model': {'messages': AIMessage(content='Your name is Bob.')}}, 'step': 4, 'parents': {}, 'thread_id': '1'},
            created_at='2025-05-05T16:01:24.680462+00:00',
            parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
            tasks=(),
            interrupts=()
        ),
        StateSnapshot(
            values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?'), HumanMessage(content="what's my name?")]},
            next=('call_model',),
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
            metadata={'source': 'loop', 'writes': None, 'step': 3, 'parents': {}, 'thread_id': '1'},
            created_at='2025-05-05T16:01:23.863421+00:00',
            parent_config={...}
            tasks=(PregelTask(id='8ab4155e-6b15-b885-9ce5-bed69a2c305c', name='call_model', path=('__pregel_pull', 'call_model'), error=None, interrupts=(), state=None, result={'messages': AIMessage(content='Your name is Bob.')}),),
            interrupts=()
        ),
        StateSnapshot(
            values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')]},
            next=('__start__',),
            config={...},
            metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': "what's my name?"}]}}, 'step': 2, 'parents': {}, 'thread_id': '1'},
            created_at='2025-05-05T16:01:23.863173+00:00',
            parent_config={...}
            tasks=(PregelTask(id='24ba39d6-6db1-4c9b-f4c5-682aeaf38dcd', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'messages': [{'role': 'user', 'content': "what's my name?"}]}),),
            interrupts=()
        ),
        StateSnapshot(
            values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')]},
            next=(),
            config={...},
            metadata={'source': 'loop', 'writes': {'call_model': {'messages': AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')}}, 'step': 1, 'parents': {}, 'thread_id': '1'},
            created_at='2025-05-05T16:01:23.862295+00:00',
            parent_config={...}
            tasks=(),
            interrupts=()
        ),
        StateSnapshot(
            values={'messages': [HumanMessage(content="hi! I'm bob")]},
            next=('call_model',),
            config={...},
            metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': '1'},
            created_at='2025-05-05T16:01:22.278960+00:00',
            parent_config={...}
            tasks=(PregelTask(id='8cbd75e0-3720-b056-04f7-71ac805140a0', name='call_model', path=('__pregel_pull', 'call_model'), error=None, interrupts=(), state=None, result={'messages': AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')}),),
            interrupts=()
        ),
        StateSnapshot(
            values={'messages': []},
            next=('__start__',),
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-0870-6ce2-bfff-1f3f14c3e565'}},
            metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': "hi! I'm bob"}]}}, 'step': -1, 'parents': {}, 'thread_id': '1'},
            created_at='2025-05-05T16:01:22.277497+00:00',
            parent_config=None,
            tasks=(PregelTask(id='d458367b-8265-812c-18e2-33001d199ce6', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'messages': [{'role': 'user', 'content': "hi! I'm bob"}]}),),
            interrupts=()
        )
    ]
    ```

=== "Checkpointer API"

    ```python
    config = {
        "configurable": {
            # highlight-next-line
            "thread_id": "1"
        }
    }
    # highlight-next-line
    list(checkpointer.list(config))
    ```

    ```
    [
        CheckpointTuple(
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1f5b-6704-8004-820c16b69a5a'}},
            checkpoint={
                'v': 3,
                'ts': '2025-05-05T16:01:24.680462+00:00',
                'id': '1f029ca3-1f5b-6704-8004-820c16b69a5a',
                'channel_versions': {'__start__': '00000000000000000000000000000005.0.5290678567601859', 'messages': '00000000000000000000000000000006.0.3205149138784782', 'branch:to:call_model': '00000000000000000000000000000006.0.14611156755133758'},
                'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000004.0.5736472536395331'}, 'call_model': {'branch:to:call_model': '00000000000000000000000000000005.0.1410174088651449'}},
                'channel_values': {'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?'), HumanMessage(content="what's my name?"), AIMessage(content='Your name is Bob.')]},
            },
            metadata={'source': 'loop', 'writes': {'call_model': {'messages': AIMessage(content='Your name is Bob.')}}, 'step': 4, 'parents': {}, 'thread_id': '1'},
            parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
            pending_writes=[]
        ),
        CheckpointTuple(
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
            checkpoint={
                'v': 3,
                'ts': '2025-05-05T16:01:23.863421+00:00',
                'id': '1f029ca3-1790-6b0a-8003-baf965b6a38f',
                'channel_versions': {'__start__': '00000000000000000000000000000005.0.5290678567601859', 'messages': '00000000000000000000000000000006.0.3205149138784782', 'branch:to:call_model': '00000000000000000000000000000006.0.14611156755133758'},
                'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000004.0.5736472536395331'}, 'call_model': {'branch:to:call_model': '00000000000000000000000000000005.0.1410174088651449'}},
                'channel_values': {'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?'), HumanMessage(content="what's my name?")], 'branch:to:call_model': None}
            },
            metadata={'source': 'loop', 'writes': None, 'step': 3, 'parents': {}, 'thread_id': '1'},
            parent_config={...},
            pending_writes=[('8ab4155e-6b15-b885-9ce5-bed69a2c305c', 'messages', AIMessage(content='Your name is Bob.'))]
        ),
        CheckpointTuple(
            config={...},
            checkpoint={
                'v': 3,
                'ts': '2025-05-05T16:01:23.863173+00:00',
                'id': '1f029ca3-1790-616e-8002-9e021694a0cd',
                'channel_versions': {'__start__': '00000000000000000000000000000004.0.5736472536395331', 'messages': '00000000000000000000000000000003.0.7056767754077798', 'branch:to:call_model': '00000000000000000000000000000003.0.22059023329132854'},
                'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.7040775356287469'}, 'call_model': {'branch:to:call_model': '00000000000000000000000000000002.0.9300422176788571'}},
                'channel_values': {'__start__': {'messages': [{'role': 'user', 'content': "what's my name?"}]}, 'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')]}
            },
            metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': "what's my name?"}]}}, 'step': 2, 'parents': {}, 'thread_id': '1'},
            parent_config={...},
            pending_writes=[('24ba39d6-6db1-4c9b-f4c5-682aeaf38dcd', 'messages', [{'role': 'user', 'content': "what's my name?"}]), ('24ba39d6-6db1-4c9b-f4c5-682aeaf38dcd', 'branch:to:call_model', None)]
        ),
        CheckpointTuple(
            config={...},
            checkpoint={
                'v': 3,
                'ts': '2025-05-05T16:01:23.862295+00:00',
                'id': '1f029ca3-178d-6f54-8001-d7b180db0c89',
                'channel_versions': {'__start__': '00000000000000000000000000000002.0.18673090920108737', 'messages': '00000000000000000000000000000003.0.7056767754077798', 'branch:to:call_model': '00000000000000000000000000000003.0.22059023329132854'},
                'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.7040775356287469'}, 'call_model': {'branch:to:call_model': '00000000000000000000000000000002.0.9300422176788571'}},
                'channel_values': {'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')]}
            },
            metadata={'source': 'loop', 'writes': {'call_model': {'messages': AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')}}, 'step': 1, 'parents': {}, 'thread_id': '1'},
            parent_config={...},
            pending_writes=[]
        ),
        CheckpointTuple(
            config={...},
            checkpoint={
                'v': 3,
                'ts': '2025-05-05T16:01:22.278960+00:00',
                'id': '1f029ca3-0874-6612-8000-339f2abc83b1',
                'channel_versions': {'__start__': '00000000000000000000000000000002.0.18673090920108737', 'messages': '00000000000000000000000000000002.0.30296526818059655', 'branch:to:call_model': '00000000000000000000000000000002.0.9300422176788571'},
                'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.7040775356287469'}},
                'channel_values': {'messages': [HumanMessage(content="hi! I'm bob")], 'branch:to:call_model': None}
            },
            metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': '1'},
            parent_config={...},
            pending_writes=[('8cbd75e0-3720-b056-04f7-71ac805140a0', 'messages', AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?'))]
        ),
        CheckpointTuple(
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-0870-6ce2-bfff-1f3f14c3e565'}},
            checkpoint={
                'v': 3,
                'ts': '2025-05-05T16:01:22.277497+00:00',
                'id': '1f029ca3-0870-6ce2-bfff-1f3f14c3e565',
                'channel_versions': {'__start__': '00000000000000000000000000000001.0.7040775356287469'},
                'versions_seen': {'__input__': {}},
                'channel_values': {'__start__': {'messages': [{'role': 'user', 'content': "hi! I'm bob"}]}}
            },
            metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': "hi! I'm bob"}]}}, 'step': -1, 'parents': {}, 'thread_id': '1'},
            parent_config=None,
            pending_writes=[('d458367b-8265-812c-18e2-33001d199ce6', 'messages', [{'role': 'user', 'content': "hi! I'm bob"}]), ('d458367b-8265-812c-18e2-33001d199ce6', 'branch:to:call_model', None)]
        )
    ]
    ```

:::

:::js

```typescript
const config = {
  configurable: {
    thread_id: "1",
  },
};

const history = [];
for await (const state of graph.getStateHistory(config)) {
  history.push(state);
}
```

:::

#### Delete all checkpoints for a thread

:::python

```python
thread_id = "1"
checkpointer.delete_thread(thread_id)
```

:::

:::js

```typescript
const threadId = "1";
await checkpointer.deleteThread(threadId);
```

:::

:::python

## Prebuilt memory tools

**LangMem** is a LangChain-maintained library that offers tools for managing long-term memories in your agent. See the [LangMem documentation](https://langchain-ai.github.io/langmem/) for usage examples.

:::

