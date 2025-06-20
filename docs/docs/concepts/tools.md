# Tools

Many AI applications interact with users via natural language. However, some use cases require models to interact directly with external systems—such as APIs, databases, or file systems—using structured input. In these cases, **tool calling** enables models to generate structured outputs conforming to a predefined input schema.

[Tools](https://python.langchain.com/docs/concepts/tools/) encapsulate a callable function and its input schema. These can be passed to compatible [chat models](https://python.langchain.com/docs/concepts/chat_models), allowing the model to invoke tools with specific arguments when appropriate.

## Tool calling

![Diagram of a tool call by a model](./img/tool_call.png)

Tool calling is usually **conditional**: the model decides whether to call a tool based on the input. If the input is unrelated to any tool, the model returns a natural language response:

```python
llm_with_tools.invoke("Hello world!")  # → AIMessage(content="Hello!")
```

If the input is relevant, the model generates a `tool_call`:

```python
llm_with_tools.invoke("What is 2 multiplied by 3?")
# → AIMessage(tool_calls=[{'name': 'multiply', 'args': {'a': 2, 'b': 3}, ...}])
```

See the [tool calling guide](../how-tos/tool-calling.md) for more details.

## Custom tools

You can define custom tools using the `@tool` decorator or plain Python functions. For example:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

See the [tool calling guide](../how-tos/tool-calling.md) for more details.

## Prebuilt tools

LangChain offers prebuilt tool integrations for APIs, databases, file systems, and web data. 

Browse the [integrations directory](https://python.langchain.com/docs/integrations/tools/) for available tools.

Common categories:

* **Search**: Bing, SerpAPI, Tavily
* **Code execution**: Python REPL, Node.js REPL
* **Databases**: SQL, MongoDB, Redis
* **Web data**: Scraping and browsing
* **APIs**: OpenWeatherMap, NewsAPI, etc.

## Tool execution

LangGraph provides prebuilt components to handle tool invocation:

* [`ToolNode`][oolNode]: Executes tools based on AI tool calls.
* [`create_react_agent`][create_react_agent]: Constructs a full agent that manages tool calling automatically.
