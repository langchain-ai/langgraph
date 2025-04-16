---
title: Context
---

# Context

Agents often need more than a list of messages to work effectively — they need **context**.

Context includes *any* data outside the message list that can shape agent behavior or tool execution. This can be:

- Information passed at runtime, like a `user_id` or API credentials.
- Internal state updated during a multi-step reasoning process.
- Persistent memory or facts from previous interactions.

LangGraph provides two primary mechanisms for supplying context:

| Type       | Description                                   | Mutable? | Lifetime                | Accessed in    |
|------------|-----------------------------------------------|----------|-------------------------|----------------|
| **Config** | data passed at the start of a run             | ❌        | per run                 | tools, prompts |
| **State**  | dynamic data that can change during execution | ✅        | per run or conversation | tools, prompts |

These values can be used to:

- Adjust the system prompt the model sees
- Feed tools with necessary inputs
- Track facts during an ongoing conversation

## Providing Runtime Context

Use this when you need to inject data into an agent at runtime.

### Using Config

- For immutable values like user metadata, access tokens, environment settings
- Passed once when running agent
- Available in tools and prompt functions via `config`

Pass information via the `config` argument. The information should be using the "configurable" key, which is reserved key for this purpose.

```python
agent.invoke(
    {"messages": "hi!"},
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}}
)
```

### Using State 

- The state represents the agent's "working" memory. By default the state is accessible across the entire agent run.
- By enabling the checkpointer, the state is persisted across runs and can be retained across an entire conversation.
- Mutable "short-term memory" the agent can update across steps
- Define a custom state schema to track additional data

```python
class CustomState(AgentState):
    # highlight-next-line
    user_name: str

agent.invoke({
    "messages": "hi!",
    "user_name": "Jane"
})
```

## Customizing Prompts with Context

Prompts define how the agent behaves. To incorporate runtime context, you can dynamically generate prompts
by creating a function that takes the agent state and config as arguments.

Use this for:

- Personalization
- Role/goal customization
- Conditional behavior (e.g., user is admin)

```python title="Dynamic Prompt"
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    # Generate a prompt based on the agent's state or config
    ...
```

=== "Using config"

    This is useful for accessing static data that is passed at agent invocation.

    ```python title="Prompt from Config"
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

=== "Using state"

    This is especially useful for accessing any information that is [dynamically updated inside the agent](#update-context-from-tools).

    ```python title="Prompt from AgentState"
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

## Tools

You can pass context to tools via additional tool function parameters. To ensure that an LLM doesn't try 
to populate those parameters in the tool calls, you need to add special type annotations: 
`RunnableConfig` for config and `Annotated[StateSchema, InjectedState]` for agent state. All parameters with these annotations will be excluded the JSON schema passed to the LLM.

=== "Using config"

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

=== "Using State"

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


## Update context from tools

Tools can modify the agent's state during execution. This is useful for persisting intermediate results or making information accessible to subsequent tools or prompts.

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