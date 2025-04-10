# Human-in-the-loop

To review, edit and approve tool calls in an agent you can use LangGraph's built-in [human-in-the-loop](../concepts/human_in_the_loop.md) features, specifically the [`interrupt()`][langgraph.types.interrupt] primitive.

## Review tool calls

To add human-in-the-loop to your tools you need to:

1. Call `interrupt()` inside your tool. Then invoke the agent as usual.
2. After you invoke the agent and it is interrupted, provide a resume value via `Command(resume=...)` (e.g., approval, edits, feedback to the LLM, etc.)

See more information in the [concept guide](../concepts/human_in_the_loop.md).

```python
def add(a: int, b: int):
    """Add two numbers"""
    # highlight-next-line
    response = interrupt(
        f"Trying to call `add` with args {{'a': {a}, 'b': {b}}}. "
        "Please approve or suggest edits."
    )
    if response["type"] == "accept":
        return a + b
    elif response["type"] == "edit":
        a = response["args"]["a"]
        b = response["args"]["b"]
        return a + b
    else:
        raise ValueError(f"Unknown response type: {response['type']}")

def multiply(a: int, b: int):
    """Multiply two numbers"""
    return a * b

# highlight-next-line
checkpointer = InMemorySaver()
agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[add, multiply],
    # highlight-next-line
    checkpointer=checkpointer,
    # highlight-next-line
    version="v2"
)
config = {"configurable": {"thread_id": "1"}}
for chunk in agent.stream(
    {"messages": "what's 2 + 3 and 5 x 7? make both calculations in parallel"},
    # highlight-next-line
    config
):
    print(chunk)
    print("\n")

for chunk in agent.stream(
    # highlight-next-line
    Command(resume={"type": "accept"}),
    # Command(resume={"type": "edit", "args": {"a": 3, "b": 2}}),
    config
):
    print(chunk)
    print("\n")
```

## Using with Agent Inbox

You can create a wrapper to add interrupts to *any* tools. Below is an example implementation that also makes interrupts compatible with [Agent Inbox UI](https://github.com/langchain-ai/agent-inbox) and [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui):

```python
from typing import Callable
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt.interrupt import HumanInterruptConfig
from langgraph.prebuilt import create_react_agent

def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    @create_tool(tool.name, description=tool.description, args_schema=tool.args_schema)
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "Please review the tool call"
        }
        # NOTE: we're passing data to interrupt() in this format
        # to also support Agent Inbox UI
        # highlight-next-line
        response = interrupt([request])[0]
        # approve the tool call
        if response["type"] == "accept":
            tool_response = tool.invoke(tool_input, config)
        # update tool call args
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            tool_response = tool.invoke(tool_input, config)
        # respond to the LLM with user feedback
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

        return tool_response

    return call_tool_with_interrupt

# highlight-next-line
checkpointer = InMemorySaver()
agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[
        # highlight-next-line
        add_human_in_the_loop(add),
        multiply
    ],
    # highlight-next-line
    checkpointer=checkpointer,
    # highlight-next-line
    version="v2"
)

config = {"configurable": {"thread_id": "1"}}
for chunk in agent.stream(
    {"messages": "what's 2 + 3 and 5 x 7? make both calculations in parallel"},
    # highlight-next-line
    config
):
    print(chunk)
    print("\n")

for chunk in agent.stream(
    # highlight-next-line
    Command(resume=[{"type": "accept"}]),
    # Command(resume=[{"type": "edit", "args": {"args": {"a": 3, "b": 2}}}]),
    config
):
    print(chunk)
    print("\n")
```