# Running agents

You can run agents by calling `agent.invoke()` or or use [streaming](./streaming.md).

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(...)
# highlight-next-line
agent.invoke({"messages": "what is the weather in sf"})
```

## Inputs and outputs

Agents use a language model that expects a list of messages as an input. Therefore, agent inputs and outputs are stored as a list of messages under the `messages` key in the agent [state](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state).

### Inputs

To invoke an agent you need to provide a list of messages as an input. By default, `create_react_agent` expects a list of [LangChain messages](https://python.langchain.com/docs/concepts/messages/#langchain-messages). For convenience, you can use the following formats:

- a message string: `{"messages": "My input"}` — automatically converted to LangChain [`HumanMessage`](https://python.langchain.com/docs/concepts/messages/#humanmessage)
- a message: `{"messages": {"role": "user", "content": "hi"}}`
- a list of messages: `{"messages": [{"role": "user", "content": "hi"}]}`

!!! Note

    This behavior is different from the `prompt` parameter in `create_react_agent`. When you pass a string as a `prompt`, it is automatically converted into a system message.

### Outputs

When you invoke the agent with `agent.invoke({"messages": ...})`, the agent will return `{"messages": [...list of messages]}` that contains:

- original input messages
- messages from the tool-calling loop — assistant messages with tool calls and tool messages with tool results
- final agent response (assistant message)

## Streaming

[Streaming](./streaming.md) is key to building responsive applications. With LangGraph agents you can stream [agent progress](./streaming.md#agent-progress), [LLM tokens](./streaming.md#llm-tokens), [tool updates](./streaming.md#tool-updates) and more.

## Max iterations

To abort an agent run that exceeds a specified number of iterations you can set `recursion_limit`, which controls maximum number of LangGraph steps an agent can take before raising a `GraphRecursionError`. You can configure `recursion_limit` at runtime or when defining agent via `.with_config()`:

=== "Runtime"

    ```python
    from langgraph.errors import GraphRecursionError
    from langgraph.prebuilt import create_react_agent

    max_iterations = 3
    # highlight-next-line
    recursion_limit = 2 * max_iterations + 1
    agent = create_react_agent(
        model="anthropic:claude-3-5-haiku-latest",
        tools=[get_weather]
    )

    try:
        response = agent.invoke(
            {"messages": "what's the weather in sf"},
            # highlight-next-line
            {"recursion_limit": recursion_limit},
        )
    except GraphRecursionError:
        print("Agent stopped due to max iterations.")
    ```

=== "`.with_config()`"

    ```python
    from langgraph.errors import GraphRecursionError
    from langgraph.prebuilt import create_react_agent

    max_iterations = 3
    # highlight-next-line
    recursion_limit = 2 * max_iterations + 1
    agent = create_react_agent(
        model="anthropic:claude-3-5-haiku-latest",
        tools=[get_weather]
    )
    # highlight-next-line
    agent_with_recursion_limit = agent.with_config(recursion_limit=recursion_limit)

    try:
        response = agent_with_recursion_limit.invoke(
            {"messages": "what's the weather in sf"},
        )
    except GraphRecursionError:
        print("Agent stopped due to max iterations.")
    ```