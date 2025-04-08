# Memory

There are two types of memory that are relevant when working with the agents:

- short-term memory — memory of all of the interactions (e.g., messages) in the same conversation or session (**thread**). This memory is used to enable multi-turn conversations with an agent.
- long-term memory — memory available across different conversations or sessions (**cross-thread**). Useful for storing facts about users, past agent actions, custom agent instructions and more.

## Short-term memory

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# highlight-next-line
checkpointer = InMemorySaver()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    checkpointer=checkpointer
)

# Run the agent
# highlight-next-line
config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke(
    {"messages": "what is the weather in sf"},
    # highlight-next-line
    config
)
ny_response = agent.invoke(
    {"messages": "what about new york?"},
    # highlight-next-line
    config
)
```

## Long-term memory

### Accessing

```python
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store

# highlight-next-line
store = InMemoryStore()
# highlight-next-line
store.put(("users",), "user_123", {"name": "John Smith"})

def get_user_info(config: RunnableConfig) -> str:
    """Look up user info."""
    # highlight-next-line
    store = get_store()
    user_id = config.get("configurable", {}).get("user_id")
    # highlight-next-line
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_user_info],
    # highlight-next-line
    store=store
)

# Run the agent
agent.invoke(
    {"messages": "look up user information"},
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}}
)
```

### Updating

```python
from typing import TypedDict

store = InMemoryStore()

class UserInfo(TypedDict):
    name: str

def save_user_info(user_info: UserInfo, config: RunnableConfig) -> str:
    """Save user info."""
    # highlight-next-line
    store = get_store()
    user_id = config.get("configurable", {}).get("user_id")
    # highlight-next-line
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[save_user_info],
    # highlight-next-line
    store=store
)

# Run the agent
agent.invoke(
    {"messages": "My name is John Smith"},
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}}
)

store.get(("users",), "user_123").value
```