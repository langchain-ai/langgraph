# Context

**Context engineering** is the practice of building dynamic systems that provide the right information and tools, in the right format, so that an AI application can accomplish a task. Context can be characterized along two key dimensions:

1. By **mutability**:
    - **Static context**: Immutable data that doesn't change during execution (e.g., user metadata, database connections, tools)
    - **Dynamic context**: Mutable data that evolves as the application runs (e.g., conversation history, intermediate results, tool call observations)
2. By **lifetime**:
    - **Runtime context**: Data scoped to a single run or invocation
    - **Cross-conversation context**: Data that persists across multiple conversations or sessions

!!! tip "Runtime context vs LLM context"

    Runtime context refers to local context: data and dependencies your code needs to run. It does **not** refer to:

    * The LLM context, which is the data passed into the LLM's prompt.
    * The "context window", which is the maximum number of tokens that can be passed to the LLM.

    Runtime context can be used to optimize the LLM context. For example, you can use user metadata
    in the runtime context to fetch user preferences and feed them into the context window.

LangGraph provides three ways to manage context, which combines the mutability and lifetime dimensions:

:::python

| Context type                                                                                | Description                                            | Mutability | Lifetime           | Access method                           |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ---------- | ------------------ | --------------------------------------- |
| [**Static runtime context**](#static-runtime-context)                                       | User metadata, tools, db connections passed at startup | Static     | Single run         | `context` argument to `invoke`/`stream` |
| [**Dynamic runtime context (state)**](#dynamic-runtime-context-state)                       | Mutable data that evolves during a single run          | Dynamic    | Single run         | LangGraph state object                  |
| [**Dynamic cross-conversation context (store)**](#dynamic-cross-conversation-context-store) | Persistent data shared across conversations            | Dynamic    | Cross-conversation | LangGraph store                         |

## Static runtime context

**Static runtime context** represents immutable data like user metadata, tools, and database connections that are passed to an application at the start of a run via the `context` argument to `invoke`/`stream`. This data does not change during execution.

!!! version-added "Added in version 0.6.0: `context` replaces `config['configurable']`"

    Runtime context is now passed to the `context` argument of `invoke`/`stream`,
    which replaces the previous pattern of passing application configuration to `config['configurable']`.

```python
@dataclass
class ContextSchema:
    user_name: str

graph.invoke( # (1)!
    {"messages": [{"role": "user", "content": "hi!"}]}, # (2)!
    # highlight-next-line
    context={"user_name": "John Smith"} # (3)!
)
```

1. This is the invocation of the agent or graph. The `invoke` method runs the underlying graph with the provided input.
2. This example uses messages as an input, which is common, but your application may use different input structures.
3. This is where you pass the runtime data. The `context` parameter allows you to provide additional dependencies that the agent can use during its execution.

=== "Agent prompt"

    ```python
    from langchain_core.messages import AnyMessage
    from langgraph.runtime import get_runtime
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from langgraph.prebuilt import create_react_agent

    # highlight-next-line
    def prompt(state: AgentState) -> list[AnyMessage]:
        runtime = get_runtime(ContextSchema)
        system_msg = f"You are a helpful assistant. Address the user as {runtime.context.user_name}."
        return [{"role": "system", "content": system_msg}] + state["messages"]

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        prompt=prompt,
        context_schema=ContextSchema
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        context={"user_name": "John Smith"}
    )
    ```

    * See [Agents](../agents/agents.md) for details.

=== "Workflow node"

    ```python
    from langgraph.runtime import Runtime

    # highlight-next-line
    def node(state: State, runtime: Runtime[ContextSchema]):
        user_name = runtime.context.user_name
        ...
    ```

    * See [the Graph API](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#add-runtime-configuration) for details.

=== "In a tool"

    ```python
    from langgraph.runtime import get_runtime

    @tool
    # highlight-next-line
    def get_user_email() -> str:
        """Retrieve user information based on user ID."""
        # simulate fetching user info from a database
        runtime = get_runtime(ContextSchema)
        email = get_user_email_from_db(runtime.context.user_name)
        return email
    ```

    See the [tool calling guide](../how-tos/tool-calling.md#configuration) for details.

!!! tip

    The `Runtime` object can be used to access static context and other utilities like the active store and stream writer.
    See the [Runtime][langgraph.runtime.Runtime] documentation for details.

:::

:::js

| Context type                                                                                | Description                                   | Mutability | Lifetime           |
| ------------------------------------------------------------------------------------------- | --------------------------------------------- | ---------- | ------------------ |
| [**Config**](#config-static-context)                                                        | data passed at the start of a run             | Static     | Single run         |
| [**Dynamic runtime context (state)**](#dynamic-runtime-context-state)                       | Mutable data that evolves during a single run | Dynamic    | Single run         |
| [**Dynamic cross-conversation context (store)**](#dynamic-cross-conversation-context-store) | Persistent data shared across conversations   | Dynamic    | Cross-conversation |

## Config (static context)

Config is for immutable data like user metadata or API keys. Use this when you have values that don't change mid-run.

Specify configuration using a key called **"configurable"** which is reserved for this purpose.

```typescript
await graph.invoke(
  // (1)!
  { messages: [{ role: "user", content: "hi!" }] }, // (2)!
  // highlight-next-line
  { configurable: { user_id: "user_123" } } // (3)!
);
```

:::

## Dynamic runtime context (state)

**Dynamic runtime context** represents mutable data that can evolve during a single run and is managed through the LangGraph state object. This includes conversation history, intermediate results, and values derived from tools or LLM outputs. In LangGraph, the state object acts as [short-term memory](../concepts/memory.md) during a run.

=== "In an agent"

    Example shows how to incorporate state into an agent **prompt**.

    State can also be accessed by the agent's **tools**, which can read or update the state as needed. See [tool calling guide](../how-tos/tool-calling.md#short-term-memory) for details.

    :::python
    ```python
    from langchain_core.messages import AnyMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt import create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState

    # highlight-next-line
    class CustomState(AgentState): # (1)!
        user_name: str

    def prompt(
        # highlight-next-line
        state: CustomState
    ) -> list[AnyMessage]:
        user_name = state["user_name"]
        system_msg = f"You are a helpful assistant. User's name is {user_name}"
        return [{"role": "system", "content": system_msg}] + state["messages"]

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[...],
        # highlight-next-line
        state_schema=CustomState, # (2)!
        prompt=prompt
    )

    agent.invoke({
        "messages": "hi!",
        "user_name": "John Smith"
    })
    ```

    1. Define a custom state schema that extends `AgentState` or `MessagesState`.
    2. Pass the custom state schema to the agent. This allows the agent to access and modify the state during execution.
    :::

    :::js
    ```typescript
    import type { BaseMessage } from "@langchain/core/messages";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { MessagesZodState } from "@langchain/langgraph";
    import { z } from "zod";

    // highlight-next-line
    const CustomState = z.object({ // (1)!
      messages: MessagesZodState.shape.messages,
      userName: z.string(),
    });

    const prompt = (
      // highlight-next-line
      state: z.infer<typeof CustomState>
    ): BaseMessage[] => {
      const userName = state.userName;
      const systemMsg = `You are a helpful assistant. User's name is ${userName}`;
      return [{ role: "system", content: systemMsg }, ...state.messages];
    };

    const agent = createReactAgent({
      llm: model,
      tools: [...],
      // highlight-next-line
      stateSchema: CustomState, // (2)!
      stateModifier: prompt,
    });

    await agent.invoke({
      messages: [{ role: "user", content: "hi!" }],
      userName: "John Smith",
    });
    ```

    1. Define a custom state schema that extends `MessagesZodState` or creates a new schema.
    2. Pass the custom state schema to the agent. This allows the agent to access and modify the state during execution.
    :::

=== "In a workflow"

    :::python
    ```python
    from typing_extensions import TypedDict
    from langchain_core.messages import AnyMessage
    from langgraph.graph import StateGraph

    # highlight-next-line
    class CustomState(TypedDict): # (1)!
        messages: list[AnyMessage]
        extra_field: int

    # highlight-next-line
    def node(state: CustomState): # (2)!
        messages = state["messages"]
        ...
        return { # (3)!
            # highlight-next-line
            "extra_field": state["extra_field"] + 1
        }

    builder = StateGraph(State)
    builder.add_node(node)
    builder.set_entry_point("node")
    graph = builder.compile()
    ```

    1. Define a custom state
    2. Access the state in any node or tool
    3. The Graph API is designed to work as easily as possible with state. The return value of a node represents a requested update to the state.
    :::

    :::js
    ```typescript
    import type { BaseMessage } from "@langchain/core/messages";
    import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";
    import { z } from "zod";

    // highlight-next-line
    const CustomState = z.object({ // (1)!
      messages: MessagesZodState.shape.messages,
      extraField: z.number(),
    });

    const builder = new StateGraph(CustomState)
      .addNode("node", async (state) => { // (2)!
        const messages = state.messages;
        // ...
        return { // (3)!
          // highlight-next-line
          extraField: state.extraField + 1,
        };
      })
      .addEdge(START, "node");

    const graph = builder.compile();
    ```

    1. Define a custom state
    2. Access the state in any node or tool
    3. The Graph API is designed to work as easily as possible with state. The return value of a node represents a requested update to the state.
    :::

!!! tip "Turning on memory"

    Please see the [memory guide](../how-tos/memory/add-memory.md) for more details on how to enable memory. This is a powerful feature that allows you to persist the agent's state across multiple invocations. Otherwise, the state is scoped only to a single run.

## Dynamic cross-conversation context (store)

**Dynamic cross-conversation context** represents persistent, mutable data that spans across multiple conversations or sessions and is managed through the LangGraph store. This includes user profiles, preferences, and historical interactions. The LangGraph store acts as [long-term memory](../concepts/memory.md#long-term-memory) across multiple runs. This can be used to read or update persistent facts (e.g., user profiles, preferences, prior interactions).

For more information, see the [Memory guide](../how-tos/memory/add-memory.md).
