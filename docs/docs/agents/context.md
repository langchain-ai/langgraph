# Context

**Context engineering** is the practice of building dynamic systems that provide the right information and tools, in the right format, so that a language model can plausibly accomplish a task.

Context includes *any* data outside the message list that can shape behavior. This can be:

- Information passed at runtime, like a `user_id` or API credentials.
- Internal state updated during a multi-step reasoning process.
- Persistent memory or facts from previous interactions.

LangGraph provides **three** primary ways to supply context:

| Type                                                                         | Description                                   | Mutable? | Lifetime                |
|------------------------------------------------------------------------------|-----------------------------------------------|----------|-------------------------|
| [**Config**](#config-static-context)                                         | data passed at the start of a run             | ❌        | per run                 |
| [**Short-term memory (State)**](#short-term-memory-mutable-context)          | dynamic data that can change during execution | ✅        | per run or conversation |
| [**Long-term memory (Store)**](#long-term-memory-cross-conversation-context) | data that can be shared between conversations | ✅        | across conversations    |

## Provide runtime context

### Config (static context)

Config is for immutable data like user metadata or API keys. Use
when you have values that don't change mid-run.

Specify configuration using a key called **"configurable"** which is reserved
for this purpose:

```python
graph.invoke( # (1)!
    {"messages": [{"role": "user", "content": "hi!"}]}, # (2)!
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}} # (3)!
)
```

1. This is the invocation of the agent or graph. The `invoke` method runs the underlying graph with the provided input.
2. This example uses messages as an input, which is common, but your application may use different input structures.
3. This is where you pass the configuration data. The `config` parameter allows you to provide additional context that the agent can use during its execution.

=== "Agent prompt"

    ```python
    from langchain_core.messages import AnyMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from langgraph.prebuilt import create_react_agent

    # highlight-next-line
    def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
        user_name = config["configurable"].get("user_name")
        system_msg = f"You are a helpful assistant. Address the user as {user_name}."
        return [{"role": "system", "content": system_msg}] + state["messages"]

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        prompt=prompt
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        config={"configurable": {"user_name": "John Smith"}}
    )
    ```

    * See [Agents](../agents/agents.md) for details.

=== "Workflow node"

    ```python
    from langchain_core.runnables import RunnableConfig

    # highlight-next-line
    def node(state: State, config: RunnableConfig):
        user_name = config["configurable"].get("user_name")
        ...
    ```

    * See [the Graph API](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#add-runtime-configuration) for details.

=== "In a tool"

    ```python
    from langchain_core.runnables import RunnableConfig

    @tool
    # highlight-next-line
    def get_user_info(config: RunnableConfig) -> str:
        """Retrieve user information based on user ID."""
        user_id = config["configurable"].get("user_id")
        return "User is John Smith" if user_id == "user_123" else "Unknown user"
    ```

    See the [tool calling guide](../how-tos/tool-calling.md#configuration) for details.

### Short-term memory (mutable context)

State acts as [short-term memory](../concepts/memory.md) during a run. It holds dynamic data that can evolve during execution, such as values derived from tools or LLM outputs.

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

### Long-term memory (cross-conversation context)

For context that spans *across* conversations or sessions, LangGraph allows access to **long-term memory** via a `store`. This can be used to read or update persistent facts (e.g., user profiles, preferences, prior interactions). 

For more information, see the [Memory guide](../how-tos/memory/add-memory.md).