# Human-in-the-loop

To review, edit and approve tool calls in an agent you can use LangGraph's built-in [human-in-the-loop](../concepts/human_in_the_loop.md) features, specifically the [`interrupt()`][langgraph.types.interrupt] primitive.

## Review tool calls

To add human-in-the-loop to your tools you need to:

1. Call `interrupt()` inside your tool. Then invoke the agent as usual.
2. After you invoke the agent and it is interrupted, provide a resume value via `Command(resume=...)` (e.g., approval, edits, feedback to the LLM, etc.)

See more information in the [concept guide](../concepts/human_in_the_loop.md).

```python
from typing import Callable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import create_react_agent

# An example of a sensitive tool that requires human review / approval
def book_hotel(hotel_name: str):
    """Book a hotel"""
    # highlight-next-line
    response = interrupt(  # (1)!
        f"Trying to call `book_hotel` with args {{'hotel_name': {hotel_name}}}. "
        "Please approve or suggest edits."
    )
    if response["type"] == "accept":
        pass
    elif response["type"] == "edit":
        hotel_name = response["args"]["hotel_name"]
    else:
        raise ValueError(f"Unknown response type: {response['type']}")
    return f"Successfully booked a stay at {hotel_name}."

# highlight-next-line
checkpointer = InMemorySaver()
agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[book_hotel],
    # highlight-next-line
    checkpointer=checkpointer,
)
config = {"configurable": {"thread_id": "1"}}
for chunk in agent.stream(
    {"messages": "book a stay at McKittrick hotel"},
    # highlight-next-line
    config
):
    print(chunk)
    print("\n")

for chunk in agent.stream(
    # highlight-next-line
    Command(resume={"type": "accept"}),  # (2)!
    # Command(resume={"type": "edit", "args": {"hotel_name": "McKittrick Hotel"}}),
    config
):
    print(chunk)
    print("\n")
```

1. The [`interrupt` function][langgraph.types.interrupt] pauses the agent graph at a specific node. In this case, we call `interrupt()` at the beginning of the tool function, which pauses the graph at the node that executes the tool. The information inside `interrupt()` (e.g., tool calls) can be presented to a human, and the graph can be resumed with the user input (tool call approval, edit or feedback).
2. The [`interrupt` function][langgraph.types.interrupt] is used in conjunction with the [`Command`](../reference/types.md#langgraph.types.Command) object to resume the graph with a value provided by the human.

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

    @create_tool(
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
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
```

You can use the `add_human_in_the_loop` wrapper to add `interrupt()` to any tool without having to add it *inside* the tool:

```python
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

# highlight-next-line
checkpointer = InMemorySaver()
agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[
        # highlight-next-line
        add_human_in_the_loop(book_hotel),
    ],
    # highlight-next-line
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}
for chunk in agent.stream(
    {"messages": "book a stay at McKittrick hotel"},
    # highlight-next-line
    config
):
    print(chunk)
    print("\n")

for chunk in agent.stream(
    # highlight-next-line
    Command(resume=[{"type": "accept"}]),
    # Command(resume=[{"type": "edit", "args": {"args": {"hotel_name": "McKittrick Hotel"}}}]),
    config
):
    print(chunk)
    print("\n")
```