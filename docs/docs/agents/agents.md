---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# LangGraph quickstart

This guide shows you how to set up and use LangGraph's **prebuilt**, **reusable** components, which are designed to help you construct agentic systems quickly and reliably.

## Prerequisites

Before you start this tutorial, ensure you have the following:

- An [Anthropic](https://console.anthropic.com/settings/keys) API key 

## 1. Install dependencies

If you haven't already, install LangGraph and LangChain:

```
pip install -U langgraph "langchain[anthropic]"
```

!!! info 

    LangChain is installed so the agent can call the [model](https://python.langchain.com/docs/integrations/chat/).

## 2. Create an agent

To create an agent, use [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent]:

```python
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:  # (1)!
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",  # (2)!
    tools=[get_weather],  # (3)!
    prompt="You are a helpful assistant"  # (4)!
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

1. Define a tool for the agent to use. Tools can be defined as vanilla Python functions. For more advanced tool usage and customization, check the [tools](./tools.md) page.
2. Provide a language model for the agent to use. To learn more about configuring language models for the agents, check the [models](./models.md) page.
3. Provide a list of tools for the model to use.
4. Provide a system prompt (instructions) to the language model used by the agent.

## 3. Configure an LLM

To configure an LLM with specific parameters, such as temperature, use [init_chat_model](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html):

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# highlight-next-line
model = init_chat_model(
    "anthropic:claude-3-7-sonnet-latest",
    # highlight-next-line
    temperature=0
)

agent = create_react_agent(
    # highlight-next-line
    model=model,
    tools=[get_weather],
)
```

For more information on how to configure LLMs, see [Models](./models.md).

## 4. Add a custom prompt

Prompts instruct the LLM how to behave. Add one of the following types of prompts:

* **Static**: A string is interpreted as a **system message**.
* **Dynamic**: A list of messages generated at **runtime**, based on input or configuration.

=== "Static prompt"

    Define a fixed prompt string or list of messages:

    ```python
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        # A static prompt that never changes
        # highlight-next-line
        prompt="Never answer questions about the weather."
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    ```

=== "Dynamic prompt"

    Define a function that returns a message list based on the agent's state and configuration:

    ```python
    from langchain_core.messages import AnyMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from langgraph.prebuilt import create_react_agent

    # highlight-next-line
    def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  # (1)!
        user_name = config["configurable"].get("user_name")
        system_msg = f"You are a helpful assistant. Address the user as {user_name}."
        return [{"role": "system", "content": system_msg}] + state["messages"]

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        # highlight-next-line
        prompt=prompt
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        config={"configurable": {"user_name": "John Smith"}}
    )
    ```

    1. Dynamic prompts allow including non-message [context](./context.md) when constructing an input to the LLM, such as:

        - Information passed at runtime, like a `user_id` or API credentials (using `config`).
        - Internal agent state updated during a multi-step reasoning process (using `state`).

        Dynamic prompts can be defined as functions that take `state` and `config` and return a list of messages to send to the LLM.

For more information, see [Context](./context.md).

## 5. Add memory

To allow multi-turn conversations with an agent, you need to enable [persistence](../concepts/persistence.md) by providing a `checkpointer` when creating an agent. At runtime, you need to provide a config containing `thread_id` — a unique identifier for the conversation (session):

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# highlight-next-line
checkpointer = InMemorySaver()

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    checkpointer=checkpointer  # (1)!
)

# Run the agent
# highlight-next-line
config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    # highlight-next-line
    config  # (2)!
)
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    # highlight-next-line
    config
)
```

1. `checkpointer` allows the agent to store its state at every step in the tool calling loop. This enables [short-term memory](./memory.md#short-term-memory) and [human-in-the-loop](./human-in-the-loop.md) capabilities.
2. Pass configuration with `thread_id` to be able to resume the same conversation on future agent invocations.

When you enable the checkpointer, it stores agent state at every step in the provided checkpointer database (or in memory, if using `InMemorySaver`).

Note that in the above example, when the agent is invoked the second time with the same `thread_id`, the original message history from the first conversation is automatically included, together with the new user input.

For more information, see [Memory](./memory.md).

## 6. Configure structured output

To produce structured responses conforming to a schema, use the `response_format` parameter. The schema can be defined with a `Pydantic` model or `TypedDict`. The result will be accessible via the `structured_response` field.

```python
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

class WeatherResponse(BaseModel):
    conditions: str

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    response_format=WeatherResponse  # (1)!
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

# highlight-next-line
response["structured_response"]
```

1. When `response_format` is provided, a separate step is added at the end of the agent loop: agent message history is passed to an LLM with structured output to generate a structured response.

    To provide a system prompt to this LLM, use a tuple `(prompt, schema)`, e.g., `response_format=(prompt, WeatherResponse)`.

!!! Note "LLM post-processing"

    Structured output requires an additional call to the LLM to format the response according to the schema.

## Next steps

- [Deploy your agent locally](../tutorials/langgraph-platform/local-server.md)
- [Learn more about prebuilt agents](../agents/overview.md)
- [LangGraph Platform quickstart](../cloud/quick_start.md)
