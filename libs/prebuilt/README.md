# LangGraph Prebuilt

This library defines high-level APIs for creating and executing LangGraph agents and tools.

> [!IMPORTANT]
> This library is meant to be bundled with `langgraph`, don't install it directly

## Agents

`langgraph-prebuilt` provides an [implementation](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) of a tool-calling [ReAct-style](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-implementation) agent - `create_react_agent`:

```bash
pip install langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

# Define the tools for the agent to use
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]
model = ChatAnthropic(model="claude-3-7-sonnet-latest")

app = create_react_agent(model, tools)
# run the agent
app.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
)
```

## Tools

### ToolNode

`langgraph-prebuilt` provides an [implementation](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode) of a node that executes tool calls - `ToolNode`:

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tool_node = ToolNode([search])
tool_calls = [{"name": "search", "args": {"query": "what is the weather in sf"}, "id": "1"}]
ai_message = AIMessage(content="", tool_calls=tool_calls)
# execute tool call
tool_node.invoke({"messages": [ai_message]})
```

### ValidationNode

`langgraph-prebuilt` provides an [implementation](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_validator.ValidationNode) of a node that validates tool calls against a pydantic schema - `ValidationNode`:

```python
from pydantic import BaseModel, field_validator
from langgraph.prebuilt import ValidationNode
from langchain_core.messages import AIMessage


class SelectNumber(BaseModel):
    a: int

    @field_validator("a")
    def a_must_be_meaningful(cls, v):
        if v != 37:
            raise ValueError("Only 37 is allowed")
        return v

validation_node = ValidationNode([SelectNumber])
validation_node.invoke({
    "messages": [AIMessage("", tool_calls=[{"name": "SelectNumber", "args": {"a": 42}, "id": "1"}])]
})
```

## Agent Inbox

The library contains schemas for using the [Agent Inbox](https://github.com/langchain-ai/agent-inbox) with LangGraph agents. Learn more about how to use Agent Inbox [here](https://github.com/langchain-ai/agent-inbox#interrupts).

```python
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanResponse

def my_graph_function():
    # Extract the last tool call from the `messages` field in the state
    tool_call = state["messages"][-1].tool_calls[0]
    # Create an interrupt
    request: HumanInterrupt = {
        "action_request": {
            "action": tool_call['name'],
            "args": tool_call['args']
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False
        },
        "description": _generate_email_markdown(state) # Generate a detailed markdown description.
    }
    # Send the interrupt request inside a list, and extract the first response
    response = interrupt([request])[0]
    if response['type'] == "response":
        # Do something with the response
    ...
```