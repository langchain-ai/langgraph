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