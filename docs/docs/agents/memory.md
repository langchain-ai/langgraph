# Memory

There are two types of memory that are relevant when working with agents:

- short-term memory — allows an agent to remember the current conversation across multiple turns.
- long-term memory — allows an agent to remember information across different conversations.

!!! Terminology
    - short-term memory in LangGraph is referred to as **thread**-level persistence
    - long-term memory in LangGraph is referred to **cross-thread** persistence

## Short-term memory

To enable multi-turn conversations with an agent, you can provide a [checkpointer](../concepts/persistence.md) when creating an agent:

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

Message history can grow quickly and exceed LLM context window size in an agent with many conversation turns or numerous tool calls. To manage message history in `create_react_agent`, you need to define a `pre_model_hook` function or runnable that takes agent state an returns a state update. Below is an example that implements message summarization (using LangMem's prebuilt `SummarizationNode`):

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

### LangMem

[LangMem](https://langchain-ai.github.io/langmem/) is a prebuilt library that offers utilities for both long-term and short-term memory management.

Below is an example of how you can use LangMem's prebuilt tool for creating, updating and deleting long-term semantic memories (`create_manage_memory_tool`) together with `create_react_agent`:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store
# Lets agent create, update, and delete memories
# highlight-next-line
from langmem import create_manage_memory_tool


def prompt(state):
    """Prepare the messages for the LLM."""
    # Same as that provided to `create_react_agent`
    # highlight-next-line
    store = get_store()
    memories = store.search(
        # Search within the same namespace as the one
        # we've configured for the agent
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


store = InMemoryStore(
    index={ # Store extracted memories
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
checkpointer = InMemorySaver()

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[
        # The agent can call "manage_memory" to
        # create, update, and delete memories by ID
        # Namespaces add scope to memories. To
        # scope memories per-user, do ("memories", "{user_id}"):
        # highlight-next-line
        create_manage_memory_tool(namespace=("memories",)),
    ],
    prompt=prompt,
    # Our memories will be stored in this provided BaseStore instance
    store=store,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}
agent.invoke(
    {"messages": "I prefer dark display mode. Remember that."},
    config=config,
)

# New thread = new conversation!
# highlight-next-line
new_config = {"configurable": {"thread_id": "2"}}
agent.invoke(
    {"messages": "Hey there. Do you remember me? What are my preferences?"},
    # highlight-next-line
    config=new_config,
)
```

See [LangMem docs](https://langchain-ai.github.io/langmem/) for more examples and guides.