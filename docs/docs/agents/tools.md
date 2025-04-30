---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Tools

[Tools](https://python.langchain.com/docs/concepts/tools/) are a way to encapsulate a function and its input schema in a way that can be passed to a chat model that supports tool calling. This allows the model to request the execution of this function with specific inputs.

You can either [define your own tools](#define-simple-tools) or use [prebuilt integrations](#prebuilt-tools) that LangChain provides.

## Define simple tools

You can pass a vanilla function to `create_react_agent` to use as a tool:

```python
from langgraph.prebuilt import create_react_agent

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

create_react_agent(
    model="anthropic:claude-3-7-sonnet",
    tools=[multiply]
)
```

`create_react_agent` automatically converts vanilla functions to [LangChain tools](https://python.langchain.com/docs/concepts/tools/#tool-interface).

## Customize tools

For more control over tool behavior, use the `@tool` decorator:

```python
# highlight-next-line
from langchain_core.tools import tool

# highlight-next-line
@tool("multiply_tool", parse_docstring=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a * b
```

You can also define a custom input schema using Pydantic:

```python
from pydantic import BaseModel, Field

class MultiplyInputSchema(BaseModel):
    """Multiply two numbers"""
    a: int = Field(description="First operand")
    b: int = Field(description="Second operand")

# highlight-next-line
@tool("multiply_tool", args_schema=MultiplyInputSchema)
def multiply(a: int, b: int) -> int:
   return a * b
```

For additional customization, refer to the [custom tools guide](https://python.langchain.com/docs/how_to/custom_tools/).

## Hide arguments from the model

Some tools require runtime-only arguments (e.g., user ID or session context) that should not be controllable by the model.

You can put these arguments in the `state` or `config` of the agent, and access
this information inside the tool:

```python
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

def my_tool(
    # This will be populated by an LLM
    tool_arg: str,
    # access information that's dynamically updated inside the agent
    # highlight-next-line
    state: Annotated[AgentState, InjectedState],
    # access static data that is passed at agent invocation
    # highlight-next-line
    config: RunnableConfig,
) -> str:
    """My tool."""
    do_something_with_state(state["messages"])
    do_something_with_config(config)
    ...
```

## Disable parallel tool calling

Some model providers support executing multiple tools in parallel, but
allow users to disable this feature.

For supported providers, you can disable parallel tool calling by setting `parallel_tool_calls=False` via the `model.bind_tools()` method:

```python
from langchain.chat_models import init_chat_model

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

model = init_chat_model("anthropic:claude-3-5-sonnet-latest", temperature=0)
tools = [add, multiply]
agent = create_react_agent(
    # disable parallel tool calls
    # highlight-next-line
    model=model.bind_tools(tools, parallel_tool_calls=False),
    tools=tools
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what's 3 + 5 and 4 * 7?"}]}
)
```

## Return tool results directly

Use `return_direct=True` to return tool results immediately and stop the agent loop:

```python
from langchain_core.tools import tool

# highlight-next-line
@tool(return_direct=True)
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[add]
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what's 3 + 5?"}]}
)
```

## Force tool use

To force the agent to use specific tools, you can set the `tool_choice` option in `model.bind_tools()`:

```python
from langchain_core.tools import tool

# highlight-next-line
@tool(return_direct=True)
def greet(user_name: str) -> int:
    """Greet user."""
    return f"Hello {user_name}!"

tools = [greet]

agent = create_react_agent(
    # highlight-next-line
    model=model.bind_tools(tools, tool_choice={"type": "tool", "name": "greet"}),
    tools=tools
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi, I am Bob"}]}
)
```

!!! Warning "Avoid infinite loops"

    Forcing tool usage without stopping conditions can create infinite loops. Use one of the following safeguards:

    - Mark the tool with [`return_direct=True`](#return-tool-results-directly) to end the loop after execution.
    - Set [`recursion_limit`](../concepts/low_level.md#recursion-limit) to restrict the number of execution steps.

## Handle tool errors

By default, the agent will catch all exceptions raised during tool calls and will pass those as tool messages to the LLM. To control how the errors are handled, you can use the prebuilt [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] — the node that executes tools inside `create_react_agent` — via its `handle_tool_errors` parameter:

=== "Enable error handling (default)"

    ```python
    from langgraph.prebuilt import create_react_agent

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        if a == 42:
            raise ValueError("The ultimate error")
        return a * b

    # Run with error handling (default)
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[multiply]
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "what's 42 x 7?"}]}
    )
    ```

=== "Disable error handling"

    ```python
    from langgraph.prebuilt import create_react_agent, ToolNode

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        if a == 42:
            raise ValueError("The ultimate error")
        return a * b

    # highlight-next-line
    tool_node = ToolNode(
        [multiply],
        # highlight-next-line
        handle_tool_errors=False  # (1)!
    )
    agent_no_error_handling = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=tool_node
    )
    agent_no_error_handling.invoke(
        {"messages": [{"role": "user", "content": "what's 42 x 7?"}]}
    )
    ```

    1. This disables error handling (enabled by default). See all available strategies in the [API reference][langgraph.prebuilt.tool_node.ToolNode].

=== "Custom error handling"

    ```python
    from langgraph.prebuilt import create_react_agent, ToolNode

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        if a == 42:
            raise ValueError("The ultimate error")
        return a * b

    # highlight-next-line
    tool_node = ToolNode(
        [multiply],
        # highlight-next-line
        handle_tool_errors=(
            "Can't use 42 as a first operand, you must switch operands!"  # (1)!
        )
    )
    agent_custom_error_handling = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=tool_node
    )
    agent_custom_error_handling.invoke(
        {"messages": [{"role": "user", "content": "what's 42 x 7?"}]}
    )
    ```

    1. This provides a custom message to send to the LLM in case of an exception. See all available strategies in the [API reference][langgraph.prebuilt.tool_node.ToolNode].

See [API reference][langgraph.prebuilt.tool_node.ToolNode] for more information on different tool error handling options.

## Working with memory

LangGraph allows access to short-term and long-term memory from tools. See [Memory](./memory.md) guide for more information on:

* how to [read](./memory.md#read-short-term) from and [write](./memory.md#write-short-term) to **short-term** memory
* how to [read](./memory.md#read-long-term) from and [write](./memory.md#write-long-term) to **long-term** memory

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

