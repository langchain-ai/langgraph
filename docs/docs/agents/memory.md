# Memory

There are two types of memory that are relevant when working with agents:

- short-term memory — allows an agent to remember the current conversation across multiple turns.
- long-term memory — allows an agent to remember information across different conversations.

![Memory](./assets/memory.png)

!!! Terminology
    - short-term memory in LangGraph is referred to as **thread**-level persistence
    - long-term memory in LangGraph is referred to **cross-thread** persistence

## Short-term memory

To allow multi-turn conversations with an agent, you need to enable LangGraph's [persistence](../concepts/persistence.md) by providing a `checkpointer` when creating an agent:

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

### Summarize conversation history

![Summary](./assets/summary.png)

Message history can grow quickly and exceed LLM context window size in an agent with many conversation turns or numerous tool calls. To shorten the message history in `create_react_agent`, you need to provide [`pre_model_hook`][langgraph.prebuilt.chat_agent_executor.create_react_agent] parameter. For example, you can shorten it by creating a running summary of the conversation:

```python
from langchain_anthropic import ChatAnthropic
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any

model = ChatAnthropic(model="claude-3-7-sonnet-latest")

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]


checkpointer = InMemorySaver()
agent = create_react_agent(
    model=model,
    tools=tools,
    # highlight-next-line
    pre_model_hook=summarization_node,
    # highlight-next-line
    state_schema=State,
    checkpointer=checkpointer,
)
```

## Long-term memory

To access and update long-term memory from inside an agent, you can provide a [store](../how-tos/cross-thread-persistence.ipynb) when creating an agent:

### Read

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
    # Same as that provided to `create_react_agent`
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

### Update

```python
from typing import TypedDict

store = InMemoryStore()

class UserInfo(TypedDict):
    name: str

def save_user_info(user_info: UserInfo, config: RunnableConfig) -> str:
    """Save user info."""
    # Same as that provided to `create_react_agent`
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

### Memory tools

**LangMem** is a prebuilt library that offers tools for managing long-term memories in your agent. See [LangMem docs](https://langchain-ai.github.io/langmem/) for more examples and guides.