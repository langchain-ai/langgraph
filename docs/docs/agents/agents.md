# Agents

## Basic setup

The most important things to configure when you create an agent are:

- `model`: a [chat model](https://python.langchain.com/docs/concepts/chat_models/) that the agent will use. Can be one of the following:
    - a string "<model_provider>:<model_name>". See [docs](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) for more information about supported models.
    - a `ChatModel` instance, e.g., [`ChatOpenAI`](https://python.langchain.com/docs/integrations/chat/openai/), [`ChatAnthropic`](https://python.langchain.com/docs/integrations/chat/anthropic/), etc.
- `tools`: a list of [tools]() for the agent to use.

```python
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    "anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# Run the agent
agent.invoke({"messages": "what is the weather in sf"})
```

## Context

You can provide the context to the agent in two ways:

- via the agent graph [state](../concepts/low_level.md#state). You can think of this as any data that can be dynamically updated from inside the agent. For example, an agent can call a tool that looks up some data and writes that data to its state.
- via the config. You can think of this as static data that is passed once on agent invocation and is propagated through the agent execution

```python
from langgraph.prebuilt.chat_agent_executor import AgentState

class CustomState(AgentState):
    custom_data: str

agent = create_react_agent(
    "anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    state_schema=CustomState
)
```

## Dynamic instructions

You might want to construct the system prompt dynamically. For example, you might want to include some information from the agent context (e.g., a state value or a config value). To do so, you can define a prompt as a function:

```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

def prompt(state: AgentState, config: RunnableConfig):
    user_name = config.get("configurable", {}).get("user_name")
    system_msg = f"You are a helpful assistant. User's name is {user_name}"
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    "anthropic:claude-3-7-sonnet-latest",
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

See more on how to manage context in your agent [here](./context.md).

## Structured output

By default the final agent response is simply an AI message with text content. However, you might want to return agent's response in a structured output that conforms to a given schema. To do so, you can provide the desired output schema via `response_format` parameter. The schema can be a Pydantic model or a `TypedDict` object:

```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    conditions: str

agent = create_react_agent(
    "anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    response_format=WeatherResponse
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "what is the weather in sf"
            }
        ]
    },
)

response["structured_response"]
```

!!! Note
    To return structured output, `create_react_agent` makes an additional LLM call at the end of the tool-calling loop.