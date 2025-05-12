---
search:
  boost: 2
tags:
  - human-in-the-loop
  - hil
  - agent
hide:
  - tags
---

# Human-in-the-loop

To review, edit and approve tool calls in an agent you can use LangGraph's built-in [Human-In-the-Loop (HIL)](../concepts/human_in_the_loop.md) features, specifically the [`interrupt()`][langgraph.types.interrupt] primitive.

LangGraph allows you to pause execution **indefinitely** — for minutes, hours, or even days—until human input is received.

This is possible because the agent state is **checkpointed into a database**, which allows the system to persist execution context and later resume the workflow, continuing from where it left off.

For a deeper dive into the **human-in-the-loop** concept, see the [concept guide](../concepts/human_in_the_loop.md).

<figure markdown="1">
![image](../concepts/img/human_in_the_loop/tool-call-review.png){: style="max-height:400px"}
<figcaption>
A human can review and edit the output from the agent before proceeding. This is particularly critical in applications where the tool calls requested may be sensitive or require human oversight.
</figcaption>
</figure>


## Review tool calls

To add a human approval step to a tool:

1. Use `interrupt()` in the tool to pause execution.
2. Resume with a `Command(resume=...)` to continue based on human input.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
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
checkpointer = InMemorySaver() # (2)!

agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[book_hotel],
    # highlight-next-line
    checkpointer=checkpointer, # (3)!
)
```

1. The [`interrupt` function][langgraph.types.interrupt] pauses the agent graph at a specific node. In this case, we call `interrupt()` at the beginning of the tool function, which pauses the graph at the node that executes the tool. The information inside `interrupt()` (e.g., tool calls) can be presented to a human, and the graph can be resumed with the user input (tool call approval, edit or feedback).
2. The `InMemorySaver` is used to store the agent state at every step in the tool calling loop. This enables [short-term memory](./memory.md#short-term-memory) and [human-in-the-loop](./human-in-the-loop.md) capabilities. In this example, we use `InMemorySaver` to store the agent state in memory. In a production application, the agent state will be stored in a database.
3. Initialize the agent with the `checkpointer`.

Run the agent with the `stream()` method, passing the `config` object to specify the thread ID. This allows the agent to resume the same conversation on future invocations.

```python
config = {
   "configurable": {
      # highlight-next-line
      "thread_id": "1"
   }
}

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "book a stay at McKittrick hotel"}]},
    # highlight-next-line
    config
):
    print(chunk)
    print("\n")
```

> You should see that the agent runs until it reaches the `interrupt()` call, at which point it pauses and waits for human input.

Resume the agent with a `Command(resume=...)` to continue based on human input.

```python
from langgraph.types import Command

for chunk in agent.stream(
    # highlight-next-line
    Command(resume={"type": "accept"}),  # (1)!
    # Command(resume={"type": "edit", "args": {"hotel_name": "McKittrick Hotel"}}),
    config
):
    print(chunk)
    print("\n")
```

1. The [`interrupt` function][langgraph.types.interrupt] is used in conjunction with the [`Command`](../reference/types.md#langgraph.types.Command) object to resume the graph with a value provided by the human.

## Using with Agent Inbox

You can create a wrapper to add interrupts to *any* tool. 

The example below provides a reference implementation compatible with [Agent Inbox UI](https://github.com/langchain-ai/agent-inbox) and [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui).

```python title="Wrapper that adds human-in-the-loop to any tool"
from typing import Callable
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt 
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt

def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review.""" 
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    @create_tool(  # (1)!
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "Please review the tool call"
        }
        # highlight-next-line
        response = interrupt([request])[0]  # (2)!
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

1. This wrapper creates a new tool that calls `interrupt()` **before** executing the wrapped tool.
2. `interrupt()` is using special input and output format that's expected by [Agent Inbox UI](https://github.com/langchain-ai/agent-inbox):
    - a list of [`HumanInterrupt`][langgraph.prebuilt.interrupt.HumanInterrupt] objects is sent to `AgentInbox` render interrupt information to the end user
    - resume value is provided by `AgentInbox` as a list (i.e., `Command(resume=[...])`)

You can use the `add_human_in_the_loop` wrapper to add `interrupt()` to any tool without having to add it *inside* the tool:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# highlight-next-line
checkpointer = InMemorySaver()

def book_hotel(hotel_name: str):
   """Book a hotel"""
   return f"Successfully booked a stay at {hotel_name}."


agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[
        # highlight-next-line
        add_human_in_the_loop(book_hotel), # (1)!
    ],
    # highlight-next-line
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}

# Run the agent
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "book a stay at McKittrick hotel"}]},
    # highlight-next-line
    config
):
    print(chunk)
    print("\n")
```

1. The `add_human_in_the_loop` wrapper is used to add `interrupt()` to the tool. This allows the agent to pause execution and wait for human input before proceeding with the tool call.

> You should see that the agent runs until it reaches the `interrupt()` call, 
>  at which point it pauses and waits for human input.

Resume the agent with a `Command(resume=...)`  to continue based on human input.

```python
from langgraph.types import Command 

for chunk in agent.stream(
    # highlight-next-line
    Command(resume=[{"type": "accept"}]),
    # Command(resume=[{"type": "edit", "args": {"args": {"hotel_name": "McKittrick Hotel"}}}]),
    config
):
    print(chunk)
    print("\n")
```

## Additional resources

* [Human-in-the-loop in LangGraph](../concepts/human_in_the_loop.md)
