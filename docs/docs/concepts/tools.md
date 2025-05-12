# Tools

Many AI applications interact directly with humans. In these cases, it is appropriate for models to respond in natural language.
But what about cases where we want a model to also interact *directly* with systems, such as databases or an API?
These systems often have a particular input schema; for example, APIs frequently have a required payload structure. You can use [tool calling](https://platform.openai.com/docs/guides/function-calling/example-use-cases) to request model responses that match a particular schema.

[Tools](https://python.langchain.com/docs/concepts/tools/) are a way to encapsulate a function and its input schema in a way that can be passed to a chat model that supports tool calling. This allows the model to request the execution of this function with specific inputs.

**Tools** can be passed to [chat models](https://python.langchain.com/docs/concepts/chat_models) that support [tool calling](https://python.langchain.com/docs/concepts/tool_calling) allowing the model to request the execution of a specific function with specific inputs.

You can [create custom tools](https://python.langchain.com/docs/how_to/custom_tools/) or use [prebuilt](#prebuilt-tools) tools.

## Tool calling

![Diagram of a tool call by a model](./img/tool_call.png)

A key principle of tool calling is that the model decides when to use a tool based on the input's relevance. The model doesn't always need to call a tool.
For example, given an input that is *irrelevant to the tool*, the model would not call the tool:

```python
result = llm_with_tools.invoke("Hello world!")
```

The result would be an `AIMessage` containing the model's response in natural language (e.g., "Hello!").
However, if we pass an input *relevant to the tool*, the model should choose to call it:

```python
result = llm_with_tools.invoke("What is 2 multiplied by 3?")
```

As before, the output `result` will be an `AIMessage`. 
But, if the tool was called, `result` will have a `tool_calls` attribute.
This attribute includes everything needed to execute the tool, including the tool name and input arguments:

```
result.tool_calls
{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'xxx', 'type': 'tool_call'}
```

For more details on usage, see the [how-to guide](../how-tos/tool-calling.ipynb).

## Execute tools

LangGraph offers pre-built components — [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] and [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] — that invoke the tools on behalf of the user.

See this [how-to guide](../how-tos/tool-calling.ipynb#use-prebuilt-toolnode) on tool calling.

## Prebuilt tools

LangChain supports a wide range of prebuilt tool integrations for interacting with APIs, databases, file systems, web data, and more. These tools extend the functionality of agents and enable rapid development.

You can browse the full list of available integrations in the [LangChain integrations directory](https://python.langchain.com/docs/integrations/tools/).

Some commonly used tool categories include:

- **Search**: Bing, SerpAPI, Tavily
- **Code interpreters**: Python REPL, Node.js REPL
- **Databases**: SQL, MongoDB, Redis
- **Web data**: Web scraping and browsing
- **APIs**: OpenWeatherMap, NewsAPI, and others

These integrations can be configured and added to your agents using the same `tools` parameter shown in the examples above.