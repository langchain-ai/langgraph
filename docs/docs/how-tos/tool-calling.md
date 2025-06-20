# Use tools

[Tools](https://python.langchain.com/docs/concepts/tools/) are a way to encapsulate a function and its input schema in a way that can be passed to a chat model that supports tool calling. This allows the model to request the execution of this function with specific inputs. This guide shows how you can create tools and use them in your graphs.

## Create tools

### Define simple tools

To create tools, you can use [@tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) decorator or vanilla Python functions.

=== "`@tool` decorator"
    ```python
    from langchain_core.tools import tool

    # highlight-next-line
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
    ```

=== "Python functions"

    This requires using LangGraph's prebuilt [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] or [agent](../../agents/agents), which automatically convert the functions to [LangChain tools](https://python.langchain.com/docs/concepts/tools/#tool-interface).
    
    ```python
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
    ```

### Customize tools

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

### Hide arguments from the model

Some tools require runtime-only arguments (e.g., user ID or session context) that should not be controllable by the model.

You can put these arguments in the [`state`](#read-state) or [`config`](#access-config) of the agent, and access
this information inside the tool:

```python
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState

@tool
def my_tool(
    # This will be populated by an LLM
    tool_arg: str,
    # access information that's dynamically updated inside the agent
    # highlight-next-line
    state: Annotated[MessagesState, InjectedState],
    # access static data that is passed at agent invocation
    # highlight-next-line
    config: RunnableConfig,
) -> str:
    """My tool."""
    do_something_with_state(state["messages"])
    do_something_with_config(config)
    ...
```

## Access config

You can provide static information to the graph at runtime, like a `user_id` or API credentials. This information can be accessed inside the tools through a special parameter **annotation** — `RunnableConfig`:

```python
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

@tool
def get_user_info(
    # highlight-next-line
    config: RunnableConfig,
) -> str:
    """Look up user info."""
    # highlight-next-line
    user_id = config["configurable"].get("user_id")
    return "User is John Smith" if user_id == "user_123" else "Unknown user"
```

??? example "Access config in tools"

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

## Short-term memory

LangGraph allows agents to access and update their [short-term memory](../../concepts/memory#short-term-memory) (state) inside the tools.

### Read state

To access the graph state inside the tools, you can use a special parameter **annotation** — [`InjectedState`][langgraph.prebuilt.InjectedState]: 

```python
from typing import Annotated
from langchain_core.tools import tool
# highlight-next-line
from langgraph.prebuilt import InjectedState

class CustomState(AgentState):
    # highlight-next-line
    user_id: str

@tool
def get_user_info(
    # highlight-next-line
    state: Annotated[CustomState, InjectedState]
) -> str:
    """Look up user info."""
    # highlight-next-line
    user_id = state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"
```

??? example "Access state in tools"

    ```python
    from typing import Annotated
    from langchain_core.tools import tool
    from langgraph.prebuilt import InjectedState, create_react_agent
    
    class CustomState(AgentState):
        # highlight-next-line
        user_id: str

    @tool
    def get_user_info(
        # highlight-next-line
        state: Annotated[CustomState, InjectedState]
    ) -> str:
        """Look up user info."""
        # highlight-next-line
        user_id = state["user_id"]
        return "User is John Smith" if user_id == "user_123" else "Unknown user"
    
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_user_info],
        # highlight-next-line
        state_schema=CustomState,
    )
    
    agent.invoke({
        "messages": "look up user information",
        # highlight-next-line
        "user_id": "user_123"
    })
    ```

### Update state

You can return state updates directly from the tools. This is useful for persisting intermediate results or making information accessible to subsequent tools or prompts.

```python
from langgraph.graph import MessagesState
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId

class CustomState(MessagesState):
    # highlight-next-line
    user_name: str

@tool
def update_user_info(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig
) -> Command:
    """Look up and update user info."""
    user_id = config["configurable"].get("user_id")
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    # highlight-next-line
    return Command(update={
        # highlight-next-line
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=tool_call_id
            )
        ]
    })
```

??? example "Update state from tools"

    This is an example of using the prebuilt agent with a tool that can update graph state.

    ```python
    from typing import Annotated
    from langchain_core.tools import tool, InjectedToolCallId
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import ToolMessage
    from langgraph.prebuilt import InjectedState, create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from langgraph.types import Command
    
    class CustomState(AgentState):
        # highlight-next-line
        user_name: str

    @tool
    def update_user_info(
        tool_call_id: Annotated[str, InjectedToolCallId],
        config: RunnableConfig
    ) -> Command:
        """Look up and update user info."""
        user_id = config["configurable"].get("user_id")
        name = "John Smith" if user_id == "user_123" else "Unknown user"
        # highlight-next-line
        return Command(update={
            # highlight-next-line
            "user_name": name,
            # update the message history
            "messages": [
                ToolMessage(
                    "Successfully looked up user information",
                    tool_call_id=tool_call_id
                )
            ]
        })
    
    def greet(
        # highlight-next-line
        state: Annotated[CustomState, InjectedState]
    ) -> str:
        """Use this to greet the user once you found their info."""
        user_name = state["user_name"]
        return f"Hello {user_name}!"
    
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_user_info, greet],
        # highlight-next-line
        state_schema=CustomState
    )
    
    agent.invoke(
        {"messages": [{"role": "user", "content": "greet the user"}]},
        # highlight-next-line
        config={"configurable": {"user_id": "user_123"}}
    )
    ```

!!! important

    If you want to use tools that return `Command` and update graph state, you can either use prebuilt [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] / [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] components, or implement your own tool-executing node that collects `Command` objects returned by the tools and returns a list of them, e.g.:
    
    ```python
    def call_tools(state):
        ...
        commands = [tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls]
        return commands
    ```

## Long-term memory

Use [long-term memory](../../concepts/memory#long-term-memory) to store user-specific or application-specific data across conversations. This is useful for applications like chatbots, where you want to remember user preferences or other information.

To use long-term memory, you need to:

1. [Configure a store](../persistence#add-long-term-memory) to persist data across invocations.
2. Use the [`get_store`][langgraph.config.get_store] function to access the store from within tools or prompts.

### Read

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
    
    1. The `InMemoryStore` is a store that stores data in memory. In a production setting, you would typically use a database or other persistent storage. Please review the [store documentation](../reference/store) for more options. If you're deploying with **LangGraph Platform**, the platform will provide a production-ready store for you.
    2. For this example, we write some sample data to the store using the `put` method. Please see the [BaseStore.put][langgraph.store.base.BaseStore.put] API reference for more details.
    3. The first argument is the namespace. This is used to group related data together. In this case, we are using the `users` namespace to group user data.
    4. A key within the namespace. This example uses a user ID for the key.
    5. The data that we want to store for the given user.
    6. The `get_store` function is used to access the store. You can call it from anywhere in your code, including tools and prompts. This function returns the store that was passed to the agent when it was created.
    7. The `get` method is used to retrieve data from the store. The first argument is the namespace, and the second argument is the key. This will return a `StoreValue` object, which contains the value and metadata about the value.
    8. The `store` is passed to the agent. This enables the agent to access the store when running tools. You can also use the `get_store` function to access the store from anywhere in your code.

### Update

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
    
    1. The `InMemoryStore` is a store that stores data in memory. In a production setting, you would typically use a database or other persistent storage. Please review the [store documentation](../reference/store) for more options. If you're deploying with **LangGraph Platform**, the platform will provide a production-ready store for you.
    2. The `UserInfo` class is a `TypedDict` that defines the structure of the user information. The LLM will use this to format the response according to the schema.
    3. The `save_user_info` function is a tool that allows an agent to update user information. This could be useful for a chat application where the user wants to update their profile information.
    4. The `get_store` function is used to access the store. You can call it from anywhere in your code, including tools and prompts. This function returns the store that was passed to the agent when it was created.
    5. The `put` method is used to store data in the store. The first argument is the namespace, and the second argument is the key. This will store the user information in the store.
    6. The `user_id` is passed in the config. This is used to identify the user whose information is being updated.

## Attach tools to a model

To attach tool schemas to a [chat model](https://python.langchain.com/docs/concepts/chat_models) you need to use `model.bind_tools()`:

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

model_with_tools.invoke("what's 42 x 7?")
```

```
AIMessage(
    content=[{'text': "I'll help you calculate that by using the multiply function.", 'type': 'text'}, {'id': 'toolu_01GhULkqytMTFDsNv6FsXy3Y', 'input': {'a': 42, 'b': 7}, 'name': 'multiply', 'type': 'tool_use'}]
    tool_calls=[{'name': 'multiply', 'args': {'a': 42, 'b': 7}, 'id': 'toolu_01GhULkqytMTFDsNv6FsXy3Y', 'type': 'tool_call'}]
)
```

## Use tools

LangChain tools conform to the [Runnable interface](https://python.langchain.com/docs/concepts/runnables/), which means that you can execute them using `.invoke()` / `.ainvoke()` methods:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# highlight-next-line
multiply.invoke({"a": 42, "b": 7})
```

```
294
```

If you want the tool to return a [ToolMessage](https://python.langchain.com/docs/concepts/messages/#toolmessage), invoke it with the tool call:

```python
tool_call = {
    "type": "tool_call",
    "id": "1",
    "args": {"a": 42, "b": 7}
}
multiply.invoke(tool_call)
```

```
ToolMessage(content='294', name='multiply', tool_call_id='1')
```

??? example "Use with a chat model"

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

    # highlight-next-line
    multiply.invoke(tool_call)
    ```

    ```
    ToolMessage(content='294', name='multiply', tool_call_id='toolu_0176DV4YKSD8FndkeuuLj36c')
    ```

## Use prebuilt agent

To create a tool-calling agent, you can use the prebuilt [create_react_agent][langgraph.prebuilt.chat_agent_executor.create_react_agent]

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
graph.invoke({"messages": [{"role": "user", "content": "what's 42 x 7?"}]})
```

See this [guide](../../agents/overview) to learn more.

## Use prebuilt `ToolNode`

[`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] is a prebuilt LangGraph [node](../../concepts/low_level#nodes) for executing tool calls.

**Why use `ToolNode`?**

* support for both sync and async tools
* concurrent execution of the tools
* error handling during tool execution. You can enable / disable this by setting `handle_tool_errors=True` (enabled by default). See [this section](#handle-errors) for more details on handling errors

ToolNode operates on [MessagesState](../../concepts/low_level#messagesstate):

* input: `MessagesState` where the last message is an `AIMessage` with `tool_calls` parameter
* output: `MessagesState` with [`ToolMessage`](https://python.langchain.com/docs/concepts/messages/#toolmessage) the result of tool calls

!!! tip

    `ToolNode` is designed to work well out-of-box with LangGraph's prebuilt [agent](../../agents/agents), but can also work with any `StateGraph` that uses `MessagesState.`

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

    This is an example of creating a tool-calling agent from scratch using `ToolNode`. You can also use LangGraph's prebuilt [agent](../../agents/agents).

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

### Handle errors

By default, the `ToolNode` will catch all exceptions raised during tool calls and will return those as tool messages. To control how the errors are handled, you can use `ToolNode`'s `handle_tool_errors` parameter:

=== "Enable error handling (default)"

    ```python
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode
    
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        if a == 42:
            raise ValueError("The ultimate error")
        return a * b
    
    tool_node = ToolNode([multiply])
    
    # Run with error handling (default)
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "multiply",
                "args": {"a": 42, "b": 7},
                "id": "tool_call_id",
                "type": "tool_call",
            }
        ],
    )
    
    tool_node.invoke({"messages": [message]})
    ```

    ```
    {'messages': [ToolMessage(content="Error: ValueError('The ultimate error')\n Please fix your mistakes.", name='multiply', tool_call_id='tool_call_id', status='error')]}
    ```

=== "Disable error handling"

    ```python
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        if a == 42:
            raise ValueError("The ultimate error")
        return a * b

    tool_node = ToolNode(
        [multiply],
        # highlight-next-line
        handle_tool_errors=False  # (1)!
    )
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "multiply",
                "args": {"a": 42, "b": 7},
                "id": "tool_call_id",
                "type": "tool_call",
            }
        ],
    )
    tool_node.invoke({"messages": [message]})
    ```

    1. This disables error handling (enabled by default). See all available strategies in the [API reference][langgraph.prebuilt.tool_node.ToolNode].

=== "Custom error handling"

    ```python
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode

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
    tool_node.invoke({"messages": [message]})
    ```

    1. This provides a custom message to send to the LLM in case of an exception. See all available strategies in the [API reference][langgraph.prebuilt.tool_node.ToolNode].

    ```
    {'messages': [ToolMessage(content="Can't use 42 as a first operand, you must switch operands!", name='multiply', tool_call_id='tool_call_id', status='error')]}
    ```

See [API reference][langgraph.prebuilt.tool_node.ToolNode] for more information on different tool error handling options.

## Handle large numbers of tools

As the number of available tools grows, you may want to limit the scope of the LLM's selection, to decrease token consumption and to help manage sources of error in LLM reasoning.

To address this, you can dynamically adjust the tools available to a model by retrieving relevant tools at runtime using semantic search.

See [`langgraph-bigtool`](https://github.com/langchain-ai/langgraph-bigtool) prebuilt library for a ready-to-use implementation and this [how-to guide](../many-tools) for more details.
