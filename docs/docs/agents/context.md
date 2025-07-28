# Context

**Context engineering** is the [art and science of filling the context window with just the right information](https://x.com/karpathy/status/1937902205765607626) so that an AI application can accomplish a task. Context can be characterized along two key dimensions:

**By mutability:**

- **Static context**: Immutable data that doesn't change during execution (e.g., user metadata, database connections, tools)
- **Dynamic context**: Mutable data that evolves as the application runs (e.g., conversation history, intermediate results, tool call observations)

**By lifetime:**

- **Runtime context**: Data scoped to a single run or invocation
- **Cross-conversation context**: Data that persists across multiple conversations or sessions

LangGraph provides three ways to manage context, combining the mutability and lifetime dimensions:

| Context Type                                                                 | Description                                            | Mutability | Lifetime                | Access Method                    |
|------------------------------------------------------------------------------|--------------------------------------------------------|------------|-------------------------|-----------------------------------|
| [**Static Runtime Context**](#static-runtime-context)                        | User metadata, tools, db connections passed at startup | Static     | Single run              | `context` argument to `invoke`/`stream` |
| [**Dynamic Runtime Context (State)**](#dynamic-runtime-context-state)        | Mutable data that evolves during a single run         | Dynamic    | Single run              | LangGraph state object           |
| [**Dynamic Cross-Conversation Context (Store)**](#dynamic-cross-conversation-context-store) | Persistent data shared across conversations            | Dynamic    | Cross-conversation      | LangGraph store                  |

## Static Runtime Context

**Static runtime context** represents immutable data like user metadata, tools, and database connections that's passed to an application at the start of a run via the `context` argument to `invoke`/`stream`. This data doesn't change during execution.

!!! version-added "New in LangGraph v0.6: `Runtime.context` replaces `config['configurable']`"

    The `Runtime` object is recommended to access static context and other utilities like the active store and stream writer.

!!! note "Application configuration vs. LLM Context"

    Runtime context can include data that will be passed to the LLM (e.g., system prompt, tools) as well as application configuration (e.g., model settings, temperature, API keys, database connections) that governs application behavior but is not explicitly passed to the LLM.
            
Application configuration can be passed via the `context` argument, which replaces `config['configurable']`. And, as before, any context you want to write to state for use in the application can be passed directly as a dictionary to `invoke` / `stream`. 

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
    def node(state: State, config: Runtime[ContextSchema]):
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

## Dynamic Runtime Context (State)

**Dynamic runtime context** represents mutable data that can evolve during a single run and is managed through the LangGraph state object. This includes conversation history, intermediate results, and values derived from tools or LLM outputs. In LangGraph, the state object acts as [short-term memory](../concepts/memory.md) during a run.

=== "In an agent"

    Example shows how to incorporate state into an agent **prompt**.

    State can also be accessed by the agent's **tools**, which can read or update the state as needed. See [tool calling guide](../how-tos/tool-calling.md#short-term-memory) for details.

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


=== "In a workflow"

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


!!! tip "Turning on memory"

    Please see the [memory guide](../how-tos/memory/add-memory.md) for more details on how to enable memory. This is a powerful feature that allows you to persist the agent's state across multiple invocations. Otherwise, the state is scoped only to a single run.

## Dynamic Cross-Conversation Context (Store)

**Dynamic cross-conversation context** represents persistent, mutable data that spans across multiple conversations or sessions and is managed through the LangGraph store. This includes user profiles, preferences, and historical interactions. The LangGraph store acts as [**long-term memory**](../concepts/memory.md#long-term-memory) across multiple runs. This can be used to read or update persistent facts (e.g., user profiles, preferences, prior interactions). 

For more information, see the [Memory guide](../how-tos/memory/add-memory.md).