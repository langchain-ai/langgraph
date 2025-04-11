# Agents

An *agent* is composed of a **Large Language Model (LLM)**, a set of **tools** that the *agent* can use to perform actions, and a
set of instructions in the form of a **prompt**.

## Basic Configuration

Use the [`create_react_agent`](https://python.langchain.com/docs/api_reference/langgraph.prebuilt.chat_agent_executor/#create-react-agent) function to create an agent:

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

## Configure the LLM

Use [init_chat_model](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) to initialize an LLM with a specific configuration. For example, you can set the temperature, max tokens, etc.

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# highlight-next-line
model = init_chat_model("anthropic:claude-3-7-sonnet-latest", temperature=0)

agent = create_react_agent(
    # highlight-next-line
    model=model,
    tools=[get_weather],
)
```

## Prompt

A prompt consists of a series of messages that instruct the LLM on how to behave. 

You can provide a prompt as a string or a list of messages. The prompt can be static or dynamic, depending on your use case.

```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

def prompt(state: AgentState, config: RunnableConfig):
    user_name = config.get("configurable", {}).get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    prompt=prompt
)

agent.invoke(
    {"messages": "what is the weather in sf"},
    # highlight-next-line
    config={"configurable": {"user_name": "John Smith"}}
)
```

See the [context](./context.md) page for more information.

## Structured output

To return the agent's response in a structured output that conforms to a given schema, you can provide the schema via the `response_format` parameter. The schema can be a Pydantic model or a `TypedDict` object. Structured output will be returned in a separate `structured_response` field:

```python
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
    To return structured output, the agent makes an additional call to the LLM to format the final response.
