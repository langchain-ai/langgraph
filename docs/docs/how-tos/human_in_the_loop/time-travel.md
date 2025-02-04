# How to view and update past graph state

!!! tip "Prerequisites"

    This guide assumes familiarity with the following concepts:

    * [Time Travel](../../../concepts/time-travel.md)
    * [Breakpoints](../../../concepts/breakpoints.md)
    * [LangGraph Glossary](../../../concepts/low_level.md)


Once you start [checkpointing](../../persistence.md) your graphs, you can easily **get** or **update** the state of the agent at any point in time. This permits a few things:

1. You can surface a state during an interrupt to a user to let them accept an action.
2. You can **rewind** the graph to reproduce or avoid issues.
3. You can **modify** the state to embed your agent into a larger system, or to let the user better control its actions.

The key methods used for this functionality are:

- [get_state](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.get_state): fetch the values from the target config
- [update_state](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.update_state): apply the given values to the target state

**Note:** this requires passing in a checkpointer.

Below is a quick example.

## Setup

First we need to install the packages required


```
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai
```

Next, we need to set API keys for OpenAI (the LLM we will use)


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Build the agent

We can now build the agent. We will build a relatively simple ReAct-style agent that does tool calling. We will use Anthropic's models and fake tools (just for demo purposes).


```python
# Set up the tool
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver


@tool
def play_song_on_spotify(song: str):
    """Play a song on Spotify"""
    # Call the spotify API ...
    return f"Successfully played {song} on Spotify!"


@tool
def play_song_on_apple(song: str):
    """Play a song on Apple Music"""
    # Call the apple music API ...
    return f"Successfully played {song} on Apple Music!"


tools = [play_song_on_apple, play_song_on_spotify]
tool_node = ToolNode(tools)

# Set up the model

model = ChatOpenAI(model="gpt-4o-mini")
model = model.bind_tools(tools, parallel_tool_calls=False)


# Define nodes and conditional edges


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Set up memory
memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable

# We add in `interrupt_before=["action"]`
# This will add a breakpoint before the `action` node is called
app = workflow.compile(checkpointer=memory)
```

## Interacting with the Agent

We can now interact with the agent. Let's ask it to play Taylor Swift's most popular song:



```python
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}
input_message = HumanMessage(content="Can you play Taylor Swift's most popular song?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

## Checking history

Let's browse the history of this thread, from start to finish.


```python
app.get_state(config).values["messages"]
```







```python
all_states = []
for state in app.get_state_history(config):
    print(state)
    all_states.append(state)
    print("--")
```

## Replay a state

We can go back to any of these states and restart the agent from there! Let's go back to right before the tool call gets executed.


```python
to_replay = all_states[2]
```


```python
to_replay.values
```







```python
to_replay.next
```






To replay from this place we just need to pass its config back to the agent. Notice that it just resumes from right where it left all - making a tool call.


```python
for event in app.stream(None, to_replay.config):
    for v in event.values():
        print(v)
```

## Branch off a past state

Using LangGraph's checkpointing, you can do more than just replay past states. You can branch off previous locations to let the agent explore alternate trajectories or to let a user "version control" changes in a workflow.

Let's show how to do this to edit the state at a particular point in time. Let's update the state to instead of playing the song on Apple to play it on Spotify:


```python
# Let's now get the last message in the state
# This is the one with the tool calls that we want to update
last_message = to_replay.values["messages"][-1]


# Let's now update the tool we are calling
last_message.tool_calls[0]["name"] = "play_song_on_spotify"

branch_config = app.update_state(
    to_replay.config,
    {"messages": [last_message]},
)
```

We can then invoke with this new `branch_config` to resume running from here with changed state. We can see from the log that the tool was called with different input.


```python
for event in app.stream(None, branch_config):
    for v in event.values():
        print(v)
```

Alternatively, we could update the state to not even call a tool!


```python
from langchain_core.messages import AIMessage

# Let's now get the last message in the state
# This is the one with the tool calls that we want to update
last_message = to_replay.values["messages"][-1]

# Let's now get the ID for the last message, and create a new message with that ID.
new_message = AIMessage(
    content="It's quiet hours so I can't play any music right now!", id=last_message.id
)

branch_config = app.update_state(
    to_replay.config,
    {"messages": [new_message]},
)
```


```python
branch_state = app.get_state(branch_config)
```


```python
branch_state.values
```







```python
branch_state.next
```






You can see the snapshot was updated and now correctly reflects that there is no next step.
