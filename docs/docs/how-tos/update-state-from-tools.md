# How to update graph state from tools

!!! info "Prerequisites"
    This guide assumes familiarity with the following:
    
    - [Command](../../concepts/low_level/#command.md)

A common use case is updating graph state from inside a tool. For example, in a customer support application you might want to look up customer account number or ID in the beginning of the conversation. To update the graph state from the tool, you can return `Command(update={"my_custom_key": "foo", "messages": [...]})` from the tool:

```python
@tool
def lookup_user_info(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig):
    """Use this to look up user information to better assist them with their questions."""
    user_info = get_user_info(config)
    return Command(
        update={
            # update the state keys
            "user_info": user_info,
            # update the message history
            "messages": [ToolMessage("Successfully looked up user information", tool_call_id=tool_call_id)]
        }
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

This guide shows how you can do this using LangGraph's prebuilt components ([`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] / [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode]).

!!! note

    Support for tools that return [`Command`][langgraph.types.Command] was added in LangGraph `v0.2.59`.

## Setup

First, let's install the required packages and set our API keys:


```
%%capture --no-stderr
%pip install -U langgraph langchain-openai
```


```python
import os
import getpass


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

Let's create a simple ReAct style agent that can look up user information and personalize the response based on the user info.

## Define tool

First, let's define the tool that we'll be using to look up user information. We'll use a naive implementation that simply looks user information up using a dictionary:


```python
USER_INFO = [
    {"user_id": "1", "name": "Bob Dylan", "location": "New York, NY"},
    {"user_id": "2", "name": "Taylor Swift", "location": "Beverly Hills, CA"},
]

USER_ID_TO_USER_INFO = {info["user_id"]: info for info in USER_INFO}
```


```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from typing_extensions import Any, Annotated


class State(AgentState):
    # updated by the tool
    user_info: dict[str, Any]


@tool
def lookup_user_info(
    tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig
):
    """Use this to look up user information to better assist them with their questions."""
    user_id = config.get("configurable", {}).get("user_id")
    if user_id is None:
        raise ValueError("Please provide user ID")

    if user_id not in USER_ID_TO_USER_INFO:
        raise ValueError(f"User '{user_id}' not found")

    user_info = USER_ID_TO_USER_INFO[user_id]
    return Command(
        update={
            # update the state keys
            "user_info": user_info,
            # update the message history
            "messages": [
                ToolMessage(
                    "Successfully looked up user information", tool_call_id=tool_call_id
                )
            ],
        }
    )
```

## Define prompt

Let's now add personalization: we'll respond differently to the user based on the state values AFTER the state has been updated from the tool. To achieve this, let's define a function that will dynamically construct the system prompt based on the graph state. It will be called ever time the LLM is called and the function output will be passed to the LLM:


```python
def prompt(state: State):
    user_info = state.get("user_info")
    if user_info is None:
        return state["messages"]

    system_msg = (
        f"User name is {user_info['name']}. User lives in {user_info['location']}"
    )
    return [{"role": "system", "content": system_msg}] + state["messages"]
```

## Define graph

Finally, let's combine this into a single graph using the prebuilt `create_react_agent`:


```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

agent = create_react_agent(
    model,
    # pass the tool that can update state
    [lookup_user_info],
    state_schema=State,
    # pass dynamic prompt function
    prompt=prompt,
)
```

## Use it!

Let's now try running our agent. We'll need to provide user ID in the config so that our tool knows what information to look up:


```python
for chunk in agent.stream(
    {"messages": [("user", "hi, what should i do this weekend?")]},
    # provide user ID in the config
    {"configurable": {"user_id": "1"}},
):
    print(chunk)
    print("\n")
```

We can see that the model correctly recommended some New York activities for Bob Dylan! Let's try getting recommendations for Taylor Swift:


```python
for chunk in agent.stream(
    {"messages": [("user", "hi, what should i do this weekend?")]},
    {"configurable": {"user_id": "2"}},
):
    print(chunk)
    print("\n")
```
