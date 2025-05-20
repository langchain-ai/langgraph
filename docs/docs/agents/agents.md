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

- An [Anthropic](https://console.anthropic.com/settings/admin-keys) API key 

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


## 6. Add human-in-the-loop

Add `human-in-the-loop` controls using the prebuilt `InterruptToolNode`. This allows you to define a function that will be called before the agent takes any action.

```python
# highlight-next-line
from langgraph.prebuilt.interrupt import HumanInterruptConfig, InterruptToolNode
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

def get_available_items() -> list[str]:  # (1)!
   """Return a list of items the user can buy."""
   return [
      "4k LED TV",
      "1080p Monitor",
      "Wireless Headphones",
      "Gaming Keyboard",
   ]

def purchase(item: str) -> str:  # (2)!
   """Buy the given item."""
   return f"✅ Purchased '{item}' successfully."

# highlight-next-line
checkpointer = InMemorySaver()  # (3)!

# Configure a human-interrupt hook for `purchase`
interrupt_hook = InterruptToolNode(  # (4)!
   purchase=HumanInterruptConfig(
      allow_accept=True,
      allow_edit=False,
      allow_ignore=False,
      allow_respond=True,
   )
)

agent = create_react_agent(
   model="anthropic:claude-3-5-haiku-latest",
   tools=[get_available_items, purchase],
   prompt="You are a helpful shopping assistant.",
   post_model_hook=interrupt_hook, # (5)!
   checkpointer=checkpointer, # (6)!
)

agent.invoke( # (7)!
   {
      "messages": [
         {"role": "user", "content": "Buy me something. Surprise me"}
      ],
   },
   {"configurable": {"thread_id": "1"}},
)

# Resume execution using the Command primitive
from langgraph.types import Command # (8)!

agent.invoke(
   Command(
      # highlight-next-line
      resume={ # (9)!
         "type": "accept",
      }
   ),
   {"configurable": {"thread_id": "1"}},
)
```

1. Define a tool for listing available items. This tool is safe to call without human review.
2. Define a tool for purchasing an item. This tool requires human review before proceeding.
3. Human-in-the-loop requires a checkpointer. Use `InMemorySaver` for development and testing. In production, use a persistent checkpointer.
4. Specify what kind of human-in-the-loop interaction you want to allow. In this case, the user can accept the purchase, but not edit or ignore it. The user can also `respond` to the agent with a message, directing it to do something else.
5. The `post_model_hook` will be called after the LLM generates a response but before any actions like `purchase` are executed. This allows you to intercept the response and decide whether to proceed with the action or do something else.
6. The `checkpointer` is required to store the state of the agent and allow for asynchronous human-in-the-loop interactions. The agent can be paused for an indefinite time while waiting for human input.
7. Invoke the agent with a message and a configuration containing `thread_id`. The agent will use the checkpointer to store its state and allow for human-in-the-loop interactions. The agent will run until it reaches the `purchase` step, at which point it will pause and output an `__interrupt__` key in the response. The interrupt will look like this:
`Interrupt(value=[{'action_request': {'action': 'purchase', 'args': {'item': 'Wireless Headphones'}}, 'config': {'allow_accept': True, 'allow_edit': False, 'allow_ignore': False, 'allow_respond': True}, 'description': 'Please review tool call for `purchase` before execution.'}], ...)`
8. The `Command` primitive is used to resume the agent's execution. It allows you to specify what action to take next.
9. The `resume` value will be passed to the location where the agent was interrupted.


For more information, see [Human-in-the-loop](./human-in-the-loop.md).

## 7. Configure structured output

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
