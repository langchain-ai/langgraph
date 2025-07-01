# Call tools

[Tools](../concepts/tools.md) encapsulate a callable function and its input schema. These can be passed to compatible [chat models](https://python.langchain.com/docs/concepts/chat_models), allowing the model to decide whether to invoke a tool and determine the appropriate arguments.

You can [define your own tools](#define-a-tool) or use [prebuilt tools](#prebuilt-tools)

## Define a tool

Define a basic tool with the [@tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) decorator:

```python
from langchain_core.tools import tool

# highlight-next-line
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

## Run a tool

Tools conform to the [Runnable interface](https://python.langchain.com/docs/concepts/runnables/), which means you can run a tool using the `invoke` method:

```python
multiply.invoke({"a": 6, "b": 7})  # returns 42
```

If the tool is invoked with `type="tool_call"`, it will return a [ToolMessage](https://python.langchain.com/docs/concepts/messages/#toolmessage):

```python
tool_call = {
    "type": "tool_call",
    "id": "1",
    "args": {"a": 42, "b": 7}
}
multiply.invoke(tool_call) # returns a ToolMessage object
```

Output:

```pycon
ToolMessage(content='294', name='multiply', tool_call_id='1')
```


## Use in an agent

To create a tool-calling agent, you can use the prebuilt [create_react_agent][langgraph.prebuilt.chat_agent_executor.create_react_agent]:

```python
from langchain_core.tools import tool
# highlight-next-line
from langgraph.prebuilt import create_react_agent

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# highlight-next-line
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet",
    tools=[multiply]
)
agent.invoke({"messages": [{"role": "user", "content": "what's 42 x 7?"}]})
```

## Use in a workflow

If you are writing a custom workflow, you will need to:

1. register the tools with the chat model
2. call the tool if the model decides to use it

Use `model.bind_tools()` to register the tools with the model. 

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(model="claude-3-5-haiku-latest")

# highlight-next-line
model_with_tools = model.bind_tools([multiply])
```

LLMs automatically determine if a tool invocation is necessary and handle calling the tool with the appropriate arguments.

??? example "Extended example: attach tools to a chat model"

    ```python
    from langchain_core.tools import tool
    from langchain.chat_models import init_chat_model

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    model = init_chat_model(model="claude-3-5-haiku-latest")
    # highlight-next-line
    model_with_tools = model.bind_tools([multiply])

    response_message = model_with_tools.invoke("what's 42 x 7?")
    tool_call = response_message.tool_calls[0]

    multiply.invoke(tool_call)
    ```

    ```pycon
    ToolMessage(
        content='294',
        name='multiply',
        tool_call_id='toolu_0176DV4YKSD8FndkeuuLj36c'
    )
    ```
#### ToolNode

To execute tools in custom workflows, use the prebuilt [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] or implement your own custom node.

`ToolNode` is a specialized node for executing tools in a workflow. It provides the following features:

* Supports both synchronous and asynchronous tools.
* Executes multiple tools concurrently.
* Handles errors during tool execution (`handle_tool_errors=True`, enabled by default). See [handling tool errors](#handle-errors) for more details.

`ToolNode` operates on [`MessagesState`](../concepts/low_level.md#messagesstate):

* **Input**: `MessagesState`, where the last message is an `AIMessage` containing the `tool_calls` parameter.
* **Output**: `MessagesState` updated with the resulting [`ToolMessage`](https://python.langchain.com/docs/concepts/messages/#toolmessage) from executed tools.


```python
# highlight-next-line
from langgraph.prebuilt import ToolNode

def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."

def get_coolest_cities():
    """Get a list of coolest cities"""
    return "nyc, sf"

# highlight-next-line
tool_node = ToolNode([get_weather, get_coolest_cities])
tool_node.invoke({"messages": [...]})
```

??? example "Single tool call"

    ```python
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode
    
    # Define tools
    @tool
    def get_weather(location: str):
        """Call to get the current weather."""
        if location.lower() in ["sf", "san francisco"]:
            return "It's 60 degrees and foggy."
        else:
            return "It's 90 degrees and sunny."
    
    # highlight-next-line
    tool_node = ToolNode([get_weather])
    
    message_with_single_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"location": "sf"},
                "id": "tool_call_id",
                "type": "tool_call",
            }
        ],
    )
    
    tool_node.invoke({"messages": [message_with_single_tool_call]})
    ```
    
    ```
    {'messages': [ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='tool_call_id')]}
    ```

??? example "Multiple tool calls"

    ```python
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode
    
    # Define tools
    
    def get_weather(location: str):
        """Call to get the current weather."""
        if location.lower() in ["sf", "san francisco"]:
            return "It's 60 degrees and foggy."
        else:
            return "It's 90 degrees and sunny."
    
    def get_coolest_cities():
        """Get a list of coolest cities"""
        return "nyc, sf"
    
    # highlight-next-line
    tool_node = ToolNode([get_weather, get_coolest_cities])

    message_with_multiple_tool_calls = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_coolest_cities",
                "args": {},
                "id": "tool_call_id_1",
                "type": "tool_call",
            },
            {
                "name": "get_weather",
                "args": {"location": "sf"},
                "id": "tool_call_id_2",
                "type": "tool_call",
            },
        ],
    )

    # highlight-next-line
    tool_node.invoke({"messages": [message_with_multiple_tool_calls]})  # (1)!
    ```

    1. `ToolNode` will execute both tools in parallel

    ```
    {
        'messages': [
            ToolMessage(content='nyc, sf', name='get_coolest_cities', tool_call_id='tool_call_id_1'),
            ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='tool_call_id_2')
        ]
    }
    ```



??? example "Use with a chat model"

    ```python
    from langchain.chat_models import init_chat_model
    from langgraph.prebuilt import ToolNode
    
    def get_weather(location: str):
        """Call to get the current weather."""
        if location.lower() in ["sf", "san francisco"]:
            return "It's 60 degrees and foggy."
        else:
            return "It's 90 degrees and sunny."
    
    # highlight-next-line
    tool_node = ToolNode([get_weather])
    
    model = init_chat_model(model="claude-3-5-haiku-latest")
    # highlight-next-line
    model_with_tools = model.bind_tools([get_weather])  # (1)!
    
    
    # highlight-next-line
    response_message = model_with_tools.invoke("what's the weather in sf?")
    tool_node.invoke({"messages": [response_message]})
    ```

    1. Use `.bind_tools()` to attach the tool schema to the chat model

    ```
    {'messages': [ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='toolu_01Pnkgw5JeTRxXAU7tyHT4UW')]}
    ```

??? example "Use in a tool-calling agent"

    This is an example of creating a tool-calling agent from scratch using `ToolNode`. You can also use LangGraph's prebuilt [agent](../agents/agents.md).

    ```python
    from langchain.chat_models import init_chat_model
    from langgraph.prebuilt import ToolNode
    from langgraph.graph import StateGraph, MessagesState, START, END
    
    def get_weather(location: str):
        """Call to get the current weather."""
        if location.lower() in ["sf", "san francisco"]:
            return "It's 60 degrees and foggy."
        else:
            return "It's 90 degrees and sunny."
    
    # highlight-next-line
    tool_node = ToolNode([get_weather])
    
    model = init_chat_model(model="claude-3-5-haiku-latest")
    # highlight-next-line
    model_with_tools = model.bind_tools([get_weather])
    
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    def call_model(state: MessagesState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    builder = StateGraph(MessagesState)
    
    # Define the two nodes we will cycle between
    builder.add_node("call_model", call_model)
    # highlight-next-line
    builder.add_node("tools", tool_node)
    
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    
    graph = builder.compile()
    
    graph.invoke({"messages": [{"role": "user", "content": "what's the weather in sf?"}]})
    ```
    
    ```
    {
        'messages': [
            HumanMessage(content="what's the weather in sf?"),
            AIMessage(
                content=[{'text': "I'll help you check the weather in San Francisco right now.", 'type': 'text'}, {'id': 'toolu_01A4vwUEgBKxfFVc5H3v1CNs', 'input': {'location': 'San Francisco'}, 'name': 'get_weather', 'type': 'tool_use'}],
                tool_calls=[{'name': 'get_weather', 'args': {'location': 'San Francisco'}, 'id': 'toolu_01A4vwUEgBKxfFVc5H3v1CNs', 'type': 'tool_call'}]
            ),
            ToolMessage(content="It's 60 degrees and foggy."),
            AIMessage(content="The current weather in San Francisco is 60 degrees and foggy. Typical San Francisco weather with its famous marine layer!")
        ]
    }
    ```


## Tool customization

### Parameter descriptions

Auto-generate descriptions from docstrings:

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

### Explicit input schema

Define schemas using `args_schema`:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class MultiplyInputSchema(BaseModel):
    """Multiply two numbers"""
    a: int = Field(description="First operand")
    b: int = Field(description="Second operand")

# highlight-next-line
@tool("multiply_tool", args_schema=MultiplyInputSchema)
def multiply(a: int, b: int) -> int:
    return a * b
```

### Tool name

Override the default tool name (function name) using the first argument:

```python
from langchain_core.tools import tool

# highlight-next-line
@tool("multiply_tool")
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

## Context management

Tools within LangGraph sometimes require context data, such as runtime-only arguments (e.g., user IDs or session details), that should not be controlled by the model. LangGraph provides three methods for managing such context:

| Type                                    | Usage Scenario                           | Mutable | Lifetime                 |
|-----------------------------------------|------------------------------------------|---------|--------------------------|
| [Configuration](#configuration)         | Static, immutable runtime data           | ❌       | Single invocation        |
| [Short-term memory](#short-term-memory) | Dynamic, changing data during invocation | ✅       | Single invocation        |
| [Long-term memory](#long-term-memory)   | Persistent, cross-session data           | ✅       | Across multiple sessions | 

### Configuration

Use configuration when you have **immutable** runtime data that tools require, such as user identifiers. You pass these arguments via [`RunnableConfig`](https://python.langchain.com/docs/concepts/runnables/#runnableconfig) at invocation and access them in the tool:

```python
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

@tool
# highlight-next-line
def get_user_info(config: RunnableConfig) -> str:
    """Retrieve user information based on user ID."""
    user_id = config["configurable"].get("user_id")
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

# Invocation example with an agent
agent.invoke(
    {"messages": [{"role": "user", "content": "look up user info"}]},
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}}
)
```

??? example "Extended example: Access config in tools"

    ```python
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    
    def get_user_info(
        # highlight-next-line
        config: RunnableConfig,
    ) -> str:
        """Look up user info."""
        # highlight-next-line
        user_id = config["configurable"].get("user_id")
        return "User is John Smith" if user_id == "user_123" else "Unknown user"
    
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_user_info],
    )
    
    agent.invoke(
        {"messages": [{"role": "user", "content": "look up user information"}]},
        # highlight-next-line
        config={"configurable": {"user_id": "user_123"}}
    )
    ```

### Short-term memory

Short-term memory maintains **dynamic** state that changes during a single execution. 

To **access** (read) the graph state inside the tools, you can use a special parameter **annotation** — [`InjectedState`][langgraph.prebuilt.InjectedState]: 

```python
from typing import Annotated, NotRequired
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

class CustomState(AgentState):
    # The user_name field in short-term state
    user_name: NotRequired[str]

@tool
def get_user_name(
    # highlight-next-line
    state: Annotated[CustomState, InjectedState]
) -> str:
    """Retrieve the current user-name from state."""
    # Return stored name or a default if not set
    return state.get("user_name", "Unknown user")

# Example agent setup
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_user_name],
    state_schema=CustomState,
)

# Invocation: reads the name from state (initially empty)
agent.invoke({"messages": "what's my name?"})
```

Use a tool that returns a `Command` to **update** `user_name` and append a confirmation message:

```python
from typing import Annotated
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId

@tool
def update_user_name(
    new_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Update user-name in short-term memory."""
    # highlight-next-line
    return Command(update={
        # highlight-next-line
        "user_name": new_name,
        # highlight-next-line
        "messages": [
            # highlight-next-line
            ToolMessage(f"Updated user name to {new_name}", tool_call_id=tool_call_id)
            # highlight-next-line
        ]
        # highlight-next-line
    })
```

!!! important

    If you want to use tools that return `Command` and update graph state, you can either use prebuilt [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] / [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] components, or implement your own tool-executing node that collects `Command` objects returned by the tools and returns a list of them, e.g.:
    
    ```python
    def call_tools(state):
        ...
        commands = [tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls]
        return commands
    ```


### Long-term memory

Use [long-term memory](../concepts/memory.md#long-term-memory) to store user-specific or application-specific data across conversations. This is useful for applications like chatbots, where you want to remember user preferences or other information.

To use long-term memory, you need to:

1. [Configure a store](memory/add-memory.md#add-long-term-memory) to persist data across invocations.
2. Use the [`get_store`][langgraph.config.get_store] function to access the store from within tools or prompts.

To **access** information in the store:

```python
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph
# highlight-next-line
from langgraph.config import get_store

@tool
def get_user_info(config: RunnableConfig) -> str:
    """Look up user info."""
    # Same as that provided to `builder.compile(store=store)` 
    # or `create_react_agent`
    # highlight-next-line
    store = get_store()
    user_id = config["configurable"].get("user_id")
    # highlight-next-line
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

builder = StateGraph(...)
...
graph = builder.compile(store=store)
```

??? example "Access long-term memory"

    ```python
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import tool
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

    @tool
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
    
    1. The `InMemoryStore` is a store that stores data in memory. In a production setting, you would typically use a database or other persistent storage. Please review the [store documentation][../reference/store.md) for more options. If you're deploying with **LangGraph Platform**, the platform will provide a production-ready store for you.
    2. For this example, we write some sample data to the store using the `put` method. Please see the [BaseStore.put][langgraph.store.base.BaseStore.put] API reference for more details.
    3. The first argument is the namespace. This is used to group related data together. In this case, we are using the `users` namespace to group user data.
    4. A key within the namespace. This example uses a user ID for the key.
    5. The data that we want to store for the given user.
    6. The `get_store` function is used to access the store. You can call it from anywhere in your code, including tools and prompts. This function returns the store that was passed to the agent when it was created.
    7. The `get` method is used to retrieve data from the store. The first argument is the namespace, and the second argument is the key. This will return a `StoreValue` object, which contains the value and metadata about the value.
    8. The `store` is passed to the agent. This enables the agent to access the store when running tools. You can also use the `get_store` function to access the store from anywhere in your code.

To **update** information in the store:

```python
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph
# highlight-next-line
from langgraph.config import get_store

@tool
def save_user_info(user_info: str, config: RunnableConfig) -> str:
    """Save user info."""
    # Same as that provided to `builder.compile(store=store)` 
    # or `create_react_agent`
    # highlight-next-line
    store = get_store()
    user_id = config["configurable"].get("user_id")
    # highlight-next-line
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

builder = StateGraph(...)
...
graph = builder.compile(store=store)
```

??? example "Update long-term memory"

    ```python
    from typing_extensions import TypedDict

    from langchain_core.tools import tool
    from langgraph.config import get_store
    from langgraph.prebuilt import create_react_agent
    from langgraph.store.memory import InMemoryStore
    
    store = InMemoryStore() # (1)!
    
    class UserInfo(TypedDict): # (2)!
        name: str

    @tool
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
    
    1. The `InMemoryStore` is a store that stores data in memory. In a production setting, you would typically use a database or other persistent storage. Please review the [store documentation](../reference/store.md) for more options. If you're deploying with **LangGraph Platform**, the platform will provide a production-ready store for you.
    2. The `UserInfo` class is a `TypedDict` that defines the structure of the user information. The LLM will use this to format the response according to the schema.
    3. The `save_user_info` function is a tool that allows an agent to update user information. This could be useful for a chat application where the user wants to update their profile information.
    4. The `get_store` function is used to access the store. You can call it from anywhere in your code, including tools and prompts. This function returns the store that was passed to the agent when it was created.
    5. The `put` method is used to store data in the store. The first argument is the namespace, and the second argument is the key. This will store the user information in the store.
    6. The `user_id` is passed in the config. This is used to identify the user whose information is being updated.

## Advanced tool features

### Immediate return

Use `return_direct=True` to immediately return a tool's result without executing additional logic.

This is useful for tools that should not trigger further processing or tool calls, allowing you to return results directly to the user.

```python
# highlight-next-line
@tool(return_direct=True)
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
```

??? example "Extended example: Using return_direct in a prebuilt agent"

    ```python
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

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


!!! important "Using without prebuilt components"

    If you are building a custom workflow and are not relying on `create_react_agent` or `ToolNode`, you will also
    need to implement the control flow to handle `return_direct=True`.

### Force tool use

If you need to force a specific tool to be used, you will need to configure this
at the **model** level using the `tool_choice` parameter in the `bind_tools` method.

Force specific tool usage via tool_choice:

```python
@tool(return_direct=True)
def greet(user_name: str) -> int:
    """Greet user."""
    return f"Hello {user_name}!"

tools = [greet]

configured_model = model.bind_tools(
    tools,
    # Force the use of the 'greet' tool
    # highlight-next-line
    tool_choice={"type": "tool", "name": "greet"}
)

```

??? example "Extended example: Force tool usage in an agent"

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

    - Mark the tool with [`return_direct=True`](#immediate-return to end the loop after execution.
    - Set [`recursion_limit`](../concepts/low_level.md#recursion-limit) to restrict the number of execution steps.


!!! tip "Tool choice configuration"

    The `tool_choice` parameter is used to configure which tool should be used by the model when it decides to call a tool. This is useful when you want to ensure that a specific tool is always called for a particular task or when you want to override the model's default behavior of choosing a tool based on its internal logic.

    Note that not all models support this feature, and the exact configuration may vary depending on the model you are using.

### Disable parallel calls

For supported providers, you can disable parallel tool calling by setting `parallel_tool_calls=False` via the `model.bind_tools()` method:

```python
model.bind_tools(
    tools, 
    # highlight-next-line
    parallel_tool_calls=False
)
```

??? example "Extended example: disable parallel tool calls in a prebuilt agent"

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

### Handle errors

LangGraph provides built-in error handling for tool execution through the prebuilt [ToolNode][langgraph.prebuilt.tool_node.ToolNode] component, used both independently and in prebuilt agents.

By **default**, `ToolNode` catches exceptions raised during tool execution and returns them as `ToolMessage` objects with a status indicating an error.

```python
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

def multiply(a: int, b: int) -> int:
    if a == 42:
        raise ValueError("The ultimate error")
    return a * b

# Default error handling (enabled by default)
tool_node = ToolNode([multiply])

message = AIMessage(
    content="",
    tool_calls=[{
        "name": "multiply",
        "args": {"a": 42, "b": 7},
        "id": "tool_call_id",
        "type": "tool_call"
    }]
)

result = tool_node.invoke({"messages": [message]})
```

Output:

```pycon
{'messages': [
    ToolMessage(
        content="Error: ValueError('The ultimate error')\n Please fix your mistakes.",
        name='multiply',
        tool_call_id='tool_call_id',
        status='error'
    )
]}
```

#### Disable error handling

To propagate exceptions directly, disable error handling:

```python
tool_node = ToolNode([multiply], handle_tool_errors=False)
```

With error handling disabled, exceptions raised by tools will propagate up, requiring explicit management.

#### Custom error messages

Provide a custom error message by setting `handle_tool_errors` to a string:

```python
tool_node = ToolNode(
    [multiply],
    handle_tool_errors="Can't use 42 as the first operand, please switch operands!"
)
```

Example output:

```python
{'messages': [
    ToolMessage(
        content="Can't use 42 as the first operand, please switch operands!",
        name='multiply',
        tool_call_id='tool_call_id',
        status='error'
    )
]}
```

#### Error handling in agents

Error handling in prebuilt agents (`create_react_agent`) leverages `ToolNode`:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[multiply]
)

# Default error handling
agent.invoke({"messages": [{"role": "user", "content": "what's 42 x 7?"}]})
```

To disable or customize error handling in prebuilt agents, explicitly pass a configured `ToolNode`:

```python
custom_tool_node = ToolNode(
    [multiply],
    handle_tool_errors="Cannot use 42 as a first operand!"
)

agent_custom = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=custom_tool_node
)

agent_custom.invoke({"messages": [{"role": "user", "content": "what's 42 x 7?"}]})
```

### Handle large numbers of tools

As the number of available tools grows, you may want to limit the scope of the LLM's selection, to decrease token consumption and to help manage sources of error in LLM reasoning.

To address this, you can dynamically adjust the tools available to a model by retrieving relevant tools at runtime using semantic search.

See [`langgraph-bigtool`](https://github.com/langchain-ai/langgraph-bigtool) prebuilt library for a ready-to-use implementation.

## Prebuilt tools

### LLM provider tools

You can use prebuilt tools from model providers by passing a dictionary with tool specs to the `tools` parameter of `create_react_agent`. For example, to use the `web_search_preview` tool from OpenAI:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="openai:gpt-4o-mini", 
    tools=[{"type": "web_search_preview"}]
)
response = agent.invoke(
    {"messages": ["What was a positive news story from today?"]}
)
```

Please consult the documentation for the specific model you are using to see which tools are available and how to use them.

### LangChain tools

Additionally, LangChain supports a wide range of prebuilt tool integrations for interacting with APIs, databases, file systems, web data, and more. These tools extend the functionality of agents and enable rapid development.

You can browse the full list of available integrations in the [LangChain integrations directory](https://python.langchain.com/docs/integrations/tools/).

Some commonly used tool categories include:

- **Search**: Bing, SerpAPI, Tavily
- **Code interpreters**: Python REPL, Node.js REPL
- **Databases**: SQL, MongoDB, Redis
- **Web data**: Web scraping and browsing
- **APIs**: OpenWeatherMap, NewsAPI, and others

These integrations can be configured and added to your agents using the same `tools` parameter shown in the examples above.

