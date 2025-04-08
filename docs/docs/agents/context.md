# Context

You can provide context to the agent in two ways:

- via the config. You can think of this as static data that is passed once on agent invocation and is propagated through the agent execution
- via the agent graph [state](../concepts/low_level.md#state). You can think of this as any data that can be dynamically updated from inside the agent. For example, an agent can call a tool that looks up some data and writes that data to its state.

## Context passing via config

```python
agent = create_react_agent(...)

agent.invoke(
    {"messages": "hi!"},
    # highlight-next-line
    {"configurable": {"custom_data": "my custom data"}}
)
```

## Context passing via state

```python
from langgraph.prebuilt.chat_agent_executor import AgentState

class CustomState(AgentState):
    # highlight-next-line
    custom_data: str

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[...],
    # highlight-next-line
    state_schema=CustomState
)

agent.invoke({
    "messages": "hi!",
    # highlight-next-line
    "custom_data": "my custom data"
})
```

## Accessing context in prompt

You might want to include the context information in agent's system prompt. For example, you might want to pass user information. To do so, you can define a prompt as a function. This function takes the agent state and config and returns a list of messages to send to the chat model.

```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    ...
```

### Via state

This is especially useful for accessing any information that is dynamically updated inside the agent (for example, via [tools that update state](../how-tos/update-state-from-tools.ipynb)).

```python
class CustomState(AgentState):
    # highlight-next-line
    user_name: str

def prompt(state: CustomState):
    # highlight-next-line
    user_name = state["user_name"]
    system_msg = f"You are a helpful assistant. User's name is {user_name}"
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[...],
    # highlight-next-line
    state_schema=CustomState,
    # highlight-next-line
    prompt=prompt
)

agent.invoke({
    "messages": "hi!",
    # highlight-next-line
    "user_name": "John Smith"
})
```

### Via config

This is useful for accessing static data that is passed at agent invocation.

```python
def prompt(state: AgentState):
    # highlight-next-line
    user_name = config.get("configurable", {}).get("user_name")
    system_msg = f"You are a helpful assistant. User's name is {user_name}"
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    prompt=prompt
)

agent.invoke(
    ...,
    # highlight-next-line
    config={"configurable": {"user_name": "John Smith"}}
)
```

## Accessing context in tools

You can pass context to tools via additional tool function parameters. To ensure that an LLM doesn't try 
to populate those parameters in the tool calls, you need to add special type annotations: 
`RunnableConfig` for config and `Annotated[StateSchema, InjectedState]` for agent state. All parameters with these annotations
will be excluded the JSON schema passed to the LLM.

### Via state

```python
from typing import Annotated
from langgraph.prebuilt import InjectedState

class CustomState(AgentState):
    # highlight-next-line
    user_id: str

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

### Via config

```python
def get_user_info(
    config: RunnableConfig,
) -> str:
    """Look up user info."""
    # highlight-next-line
    user_id = config.get("configurable", {}).get("user_id")
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_user_info],
)

agent.invoke(
    {"messages": "look up user information"},
    # highlight-next-line
    config={"configurable": {"user_id": "user_id"}}
)
```