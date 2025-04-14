# Context

Information is propagagated inside an agent via a list of messages. However, agents need access to information beyond messages. We will refer to this information broadly as **context**. You can provide context to an agent in two ways:

- **via agent state**: pass any data that the agent can *update* during its execution. For example, an agent can call a tool that looks up some data and writes that data to its [state](../concepts/low_level.md#state).
- **via a config**: pass static information to the agent's tools and prompt (e.g., user information, authorization, DB connections, etc.)

=== "Via state"

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

=== "Via config"

    ```python
    agent = create_react_agent(...)

    agent.invoke(
        {"messages": "hi!"},
        # highlight-next-line
        {"configurable": {"custom_data": "my custom data"}}
    )
    ```

## Prompt

To include context in agent's system prompt (for example, user information), you can define a prompt as a function. This function takes the agent state and config and returns a list of messages to send to the chat model:

```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    ...
```

=== "Via state"

    This is especially useful for accessing any information that is [dynamically updated inside the agent](#update-context-from-tools).

    ```python
    class CustomState(AgentState):
        # highlight-next-line
        user_name: str

    def prompt(
        # highlight-next-line
        state: CustomState
    ):
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

=== "Via config"

    This is useful for accessing static data that is passed at agent invocation.

    ```python
    def prompt(
        state: AgentState,
        # highlight-next-line
        config: RunnableConfig,
    ):
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

## Tools

You can pass context to tools via additional tool function parameters. To ensure that an LLM doesn't try 
to populate those parameters in the tool calls, you need to add special type annotations: 
`RunnableConfig` for config and `Annotated[StateSchema, InjectedState]` for agent state. All parameters with these annotations will be excluded the JSON schema passed to the LLM.

=== "Via state"

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

=== "Via config"

    ```python
    def get_user_info(
        # highlight-next-line
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
        config={"configurable": {"user_id": "user_123"}}
    )
    ```

## Update context from tools

You can update context ([state](../concepts/low_level.md#state)) of the agent from tools. This is useful if the agent needs to load some data during execution and wants to make it available for later use in the prompt or other tools.

```python
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

class CustomState(AgentState):
    # highlight-next-line
    user_name: str

def get_user_info(
    # highlight-next-line
    tool_call_id: Annotated[str, InjectedToolCallId],
    # highlight-next-line
    config: RunnableConfig
) -> Command:
    """Look up user info."""
    # highlight-next-line
    user_id = config.get("configurable", {}).get("user_id")
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        # highlight-next-line
        "user_name": name,
        # update the message history
        # highlight-next-line
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                # highlight-next-line
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
    {"messages": "greet the user"},
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}}
)
```

See [how to update state from tools](../how-tos/update-state-from-tools.ipynb) for more information.