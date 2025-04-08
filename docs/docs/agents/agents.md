# Agents

## Basic setup

The most important things to configure when you create an agent are:

- `model`: a [chat model](https://python.langchain.com/docs/concepts/chat_models/) that the agent will use. Can be one of the following:
    - a string `<model_provider>:<model_name>` (e.g., `openai:gpt-4o`). See [docs](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) for more information about supported models.
    - a `ChatModel` instance, e.g., [`ChatOpenAI`](https://python.langchain.com/docs/integrations/chat/openai/), [`ChatAnthropic`](https://python.langchain.com/docs/integrations/chat/anthropic/), etc.
- `tools`: a list of [tools](https://python.langchain.com/docs/concepts/tools/) for the agent to use.
- `prompt`: string or system message with instructions for the agent.

```python
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# Run the agent
agent.invoke({"messages": "what is the weather in sf"})
```

## Dynamic instructions

You might want to construct the system prompt dynamically. For example, you might want to include some information from the agent context (e.g., user information). To do so, you can define a prompt as a function:

```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

def prompt(state: AgentState, config: RunnableConfig):
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
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    response_format=WeatherResponse
)

response = agent.invoke({"messages": "what is the weather in sf"})

# highlight-next-line
response["structured_response"]
```

!!! Note
    To return structured output, `create_react_agent` makes an additional LLM call at the end of the tool-calling loop.

## Customizing models

If you want to customize the model parameters, you can pass a `ChatModel` instance as `model`:

```python
from langchain_anthropic import ChatAnthropic
# highlight-next-line
model = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0.7)

agent = create_react_agent(
    # highlight-next-line
    model=model,
    tools=[get_weather],
)
```