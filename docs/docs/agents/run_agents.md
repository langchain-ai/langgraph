---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Running agents

Agents support both synchronous and asynchronous execution using either `.invoke()` / `await .ainvoke()` for full responses, or `.stream()` / `.astream()` for **incremental** [streaming](../how-tos/streaming.md) output. This section explains how to provide input, interpret output, enable streaming, and control execution limits.

## Basic usage

Agents can be executed in two primary modes:

:::python

- **Synchronous** using `.invoke()` or `.stream()`
- **Asynchronous** using `await .ainvoke()` or `async for` with `.astream()`
  :::

:::js

- **Synchronous** using `.invoke()` or `.stream()`
- **Asynchronous** using `await .invoke()` or `for await` with `.stream()`
  :::

:::python
=== "Sync invocation"

    ```python
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(...)

    # highlight-next-line
    response = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
    ```

=== "Async invocation"

    ```python
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(...)
    # highlight-next-line
    response = await agent.ainvoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
    ```

:::

:::js

```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const agent = createReactAgent(...);
// highlight-next-line
const response = await agent.invoke({
    "messages": [
        { "role": "user", "content": "what is the weather in sf" }
    ]
});
```

:::

## Inputs and outputs

Agents use a language model that expects a list of `messages` as an input. Therefore, agent inputs and outputs are stored as a list of `messages` under the `messages` key in the agent [state](../concepts/low_level.md#working-with-messages-in-graph-state).

## Input format

Agent input must be a dictionary with a `messages` key. Supported formats are:

:::python
| Format | Example |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------|
| String | `{"messages": "Hello"}` — Interpreted as a [HumanMessage](https://python.langchain.com/docs/concepts/messages/#humanmessage) |
| Message dictionary | `{"messages": {"role": "user", "content": "Hello"}}` |
| List of messages | `{"messages": [{"role": "user", "content": "Hello"}]}` |
| With custom state | `{"messages": [{"role": "user", "content": "Hello"}], "user_name": "Alice"}` — If using a custom `state_schema` |
:::

:::js
| Format | Example |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------|
| String | `{"messages": "Hello"}` — Interpreted as a [HumanMessage](https://js.langchain.com/docs/concepts/messages/#humanmessage) |
| Message dictionary | `{"messages": {"role": "user", "content": "Hello"}}` |
| List of messages | `{"messages": [{"role": "user", "content": "Hello"}]}` |
| With custom state | `{"messages": [{"role": "user", "content": "Hello"}], "user_name": "Alice"}` — If using a custom state definition |
:::

:::python
Messages are automatically converted into LangChain's internal message format. You can read
more about [LangChain messages](https://python.langchain.com/docs/concepts/messages/#langchain-messages) in the LangChain documentation.
:::

:::js
Messages are automatically converted into LangChain's internal message format. You can read
more about [LangChain messages](https://js.langchain.com/docs/concepts/messages/#langchain-messages) in the LangChain documentation.
:::

!!! tip "Using custom agent state"

    :::python
    You can provide additional fields defined in your agent's state schema directly in the input dictionary. This allows dynamic behavior based on runtime data or prior tool outputs.
    See the [context guide](./context.md) for full details.
    :::

    :::js
    You can provide additional fields defined in your agent's state directly in the state definition. This allows dynamic behavior based on runtime data or prior tool outputs.
    See the [context guide](./context.md) for full details.
    :::

!!! note

    :::python
    A string input for `messages` is converted to a [HumanMessage](https://python.langchain.com/docs/concepts/messages/#humanmessage). This behavior differs from the `prompt` parameter in `create_react_agent`, which is interpreted as a [SystemMessage](https://python.langchain.com/docs/concepts/messages/#systemmessage) when passed as a string.
    :::

    :::js
    A string input for `messages` is converted to a [HumanMessage](https://js.langchain.com/docs/concepts/messages/#humanmessage). This behavior differs from the `prompt` parameter in `createReactAgent`, which is interpreted as a [SystemMessage](https://js.langchain.com/docs/concepts/messages/#systemmessage) when passed as a string.
    :::

## Output format

:::python
Agent output is a dictionary containing:

- `messages`: A list of all messages exchanged during execution (user input, assistant replies, tool invocations).
- Optionally, `structured_response` if [structured output](./agents.md#6-configure-structured-output) is configured.
- If using a custom `state_schema`, additional keys corresponding to your defined fields may also be present in the output. These can hold updated state values from tool execution or prompt logic.
:::

:::js
Agent output is a dictionary containing:

- `messages`: A list of all messages exchanged during execution (user input, assistant replies, tool invocations).
- Optionally, `structuredResponse` if [structured output](./agents.md#6-configure-structured-output) is configured.
- If using a custom state definition, additional keys corresponding to your defined fields may also be present in the output. These can hold updated state values from tool execution or prompt logic.
:::

See the [context guide](./context.md) for more details on working with custom state schemas and accessing context.

## Streaming output

Agents support streaming responses for more responsive applications. This includes:

- **Progress updates** after each step
- **LLM tokens** as they're generated
- **Custom tool messages** during execution

Streaming is available in both sync and async modes:

:::python
=== "Sync streaming"

    ```python
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        stream_mode="updates"
    ):
        print(chunk)
    ```

=== "Async streaming"

    ```python
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        stream_mode="updates"
    ):
        print(chunk)
    ```

:::

:::js

```typescript
for await (const chunk of agent.stream(
  { messages: [{ role: "user", content: "what is the weather in sf" }] },
  { streamMode: "updates" }
)) {
  console.log(chunk);
}
```

:::

!!! tip

    For full details, see the [streaming guide](../how-tos/streaming.md).

## Max iterations

:::python
To control agent execution and avoid infinite loops, set a recursion limit. This defines the maximum number of steps the agent can take before raising a `GraphRecursionError`. You can configure `recursion_limit` at runtime or when defining agent via `.with_config()`:
:::

:::js
To control agent execution and avoid infinite loops, set a recursion limit. This defines the maximum number of steps the agent can take before raising a `GraphRecursionError`. You can configure `recursionLimit` at runtime or when defining agent via `.withConfig()`:
:::

:::python
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
            {"messages": [{"role": "user", "content": "what's the weather in sf"}]},
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
            {"messages": [{"role": "user", "content": "what's the weather in sf"}]},
        )
    except GraphRecursionError:
        print("Agent stopped due to max iterations.")
    ```

:::

:::js
=== "Runtime"

    ```typescript
    import { GraphRecursionError } from "@langchain/langgraph";
    import { ChatAnthropic } from "@langchain/langgraph/prebuilt";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";

    const maxIterations = 3;
    // highlight-next-line
    const recursionLimit = 2 * maxIterations + 1;
    const agent = createReactAgent({
        llm: new ChatAnthropic({ model: "claude-3-5-haiku-latest" }),
        tools: [getWeather]
    });

    try {
        const response = await agent.invoke(
            {"messages": [{"role": "user", "content": "what's the weather in sf"}]},
            // highlight-next-line
            { recursionLimit }
        );
    } catch (error) {
        if (error instanceof GraphRecursionError) {
            console.log("Agent stopped due to max iterations.");
        }
    }
    ```

=== "`.withConfig()`"

    ```typescript
    import { GraphRecursionError } from "@langchain/langgraph";
    import { ChatAnthropic } from "@langchain/langgraph/prebuilt";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";

    const maxIterations = 3;
    // highlight-next-line
    const recursionLimit = 2 * maxIterations + 1;
    const agent = createReactAgent({
        llm: new ChatAnthropic({ model: "claude-3-5-haiku-latest" }),
        tools: [getWeather]
    });
    // highlight-next-line
    const agentWithRecursionLimit = agent.withConfig({ recursionLimit });

    try {
        const response = await agentWithRecursionLimit.invoke(
            {"messages": [{"role": "user", "content": "what's the weather in sf"}]},
        );
    } catch (error) {
        if (error instanceof GraphRecursionError) {
            console.log("Agent stopped due to max iterations.");
        }
    }
    ```

:::

:::python

## Additional Resources

- [Async programming in LangChain](https://python.langchain.com/docs/concepts/async)
  :::
