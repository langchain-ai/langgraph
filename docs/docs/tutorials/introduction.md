# 🚀 LangGraph Quickstart

In this tutorial, we will build a support chatbot in LangGraph that can:

✅ **Answer common questions** by searching the web  
✅ **Maintain conversation state** across calls  
✅ **Route complex queries** to a human for review  
✅ **Use custom state** to control its behavior  
✅ **Rewind and explore** alternative conversation paths  

We'll start with a **basic chatbot** and progressively add more sophisticated capabilities, introducing key LangGraph concepts along the way. Let’s dive in! 🌟

## Setup

First, install the required packages and configure your environment:


```
%%capture --no-stderr
%pip install -U langgraph langsmith langchain_anthropic
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Part 1: Build a Basic Chatbot

We'll first create a simple chatbot using LangGraph. This chatbot will respond directly to user messages. Though simple, it will illustrate the core concepts of building with LangGraph. By the end of this section, you will have a built rudimentary chatbot.

Start by creating a `StateGraph`. A `StateGraph` object defines the structure of our chatbot as a "state machine". We'll add `nodes` to represent the llm and functions our chatbot can call and `edges` to specify how the bot should transition between these functions.


```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
```

Our graph can now handle two key tasks:

1. Each `node` can receive the current `State` as input and output an update to the state.
2. Updates to `messages` will be appended to the existing list rather than overwriting it, thanks to the prebuilt [`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages) function used with the `Annotated` syntax.

------

!!! tip "Concept"

    When defining a graph, the first step is to define its `State`. The `State` includes the graph's schema and [reducer functions](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) that handle state updates. In our example, `State` is a `TypedDict` with one key: `messages`. The [`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages) reducer function is used to append new messages to the list instead of overwriting it. Keys without a reducer annotation will overwrite previous values. Learn more about state, reducers, and related concepts in [this guide](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages).

---------


Next, add a "`chatbot`" node. Nodes represent units of work. They are typically regular python functions.


```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
```

**Notice** how the `chatbot` node function takes the current `State` as input and returns a dictionary containing an updated `messages` list under the key "messages". This is the basic pattern for all LangGraph node functions.

The `add_messages` function in our `State` will append the llm's response messages to whatever messages are already in the state.

Next, add an `entry` point. This tells our graph **where to start its work** each time we run it.


```python
graph_builder.add_edge(START, "chatbot")
```

Similarly, set a `finish` point. This instructs the graph **"any time this node is run, you can exit."**


```python
graph_builder.add_edge("chatbot", END)
```

Finally, we'll want to be able to run our graph. To do so, call "`compile()`" on the graph builder. This creates a "`CompiledGraph`" we can use invoke on our state.


```python
graph = graph_builder.compile()
```

You can visualize the graph using the `get_graph` method and one of the "draw" methods, like `draw_ascii` or `draw_png`. The `draw` methods each require additional dependencies.


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![](data:image/png;base64,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)

Now let's run the chatbot! 

**Tip:** You can exit the chat loop at any time by typing "quit", "exit", or "q".


```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

**Congratulations!** You've built your first chatbot using LangGraph. This bot can engage in basic conversation by taking user input and generating responses using an LLM. You can inspect a [LangSmith Trace](https://smith.langchain.com/public/7527e308-9502-4894-b347-f34385740d5a/r) for the call above at the provided link.

However, you may have noticed that the bot's knowledge is limited to what's in its training data. In the next part, we'll add a web search tool to expand the bot's knowledge and make it more capable.

Below is the full code for this section for your reference:

<details>
<summary>Full Code</summary>
    <pre>
        
```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()
```

</pre>
</details>

## Part 2: 🛠️ Enhancing the Chatbot with Tools

To handle queries our chatbot can't answer "from memory", we'll integrate a web search tool. Our bot can use this tool to find relevant information and provide better responses.

#### Requirements

Before we start, make sure you have the necessary packages installed and API keys set up:

First, install the requirements to use the [Tavily Search Engine](https://python.langchain.com/docs/integrations/tools/tavily_search/), and set your [TAVILY_API_KEY](https://tavily.com/).


```
%%capture --no-stderr
%pip install -U tavily-python langchain_community
```


```python
_set_env("TAVILY_API_KEY")
```

Next, define the tool:


```python
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)
tools = [tool]
tool.invoke("What's a 'node' in LangGraph?")
```






The results are page summaries our chat bot can use to answer questions.


Next, we'll start defining our graph. The following is all **the same as in Part 1**, except we have added `bind_tools` on our LLM. This lets the LLM know the correct JSON format to use if it wants to use our search engine.


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# Modification: tell the LLM which tools it can call
# highlight-next-line
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
```

Next we need to create a function to actually run the tools if they are called. We'll do this by adding the tools to a new node.

Below, we implement a `BasicToolNode` that checks the most recent message in the state and calls tools if the message contains `tool_calls`. It relies on the LLM's `tool_calling` support, which is available in Anthropic, OpenAI, Google Gemini, and a number of other LLM providers.

We will later replace this with LangGraph's prebuilt [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode) to speed things up, but building it ourselves first is instructive.


```python
import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
```

With the tool node added, we can define the `conditional_edges`. 

Recall that **edges** route the control flow from one node to the next. **Conditional edges** usually contain "if" statements to route to different nodes depending on the current graph state. These functions receive the current graph `state` and return a string or list of strings indicating which node(s) to call next.

Below, call define a router function called `route_tools`, that checks for tool_calls in the chatbot's output. Provide this function to the graph by calling `add_conditional_edges`, which tells the graph that whenever the `chatbot` node completes to check this function to see where to go next. 

The condition will route to `tools` if tool calls are present and `END` if not.

Later, we will replace this with the prebuilt [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition) to be more concise, but implementing it ourselves first makes things more clear. 


```python
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

**Notice** that conditional edges start from a single node. This tells the graph "any time the '`chatbot`' node runs, either go to 'tools' if it calls a tool, or end the loop if it responds directly. 

Like the prebuilt `tools_condition`, our function returns the `END` string if no tool calls are made. When the graph transitions to `END`, it has no more tasks to complete and ceases execution. Because the condition can return `END`, we don't need to explicitly set a `finish_point` this time. Our graph already has a way to finish!

Let's visualize the graph we've built. The following function has some additional dependencies to run that are unimportant for this tutorial.


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![](data:image/jpg;base64,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)

Now we can ask the bot questions outside its training data.


```python
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

**Congrats!** You've created a conversational agent in langgraph that can use a search engine to retrieve updated information when needed. Now it can handle a wider range of user queries. To inspect all the steps your agent just took, check out this [LangSmith trace](https://smith.langchain.com/public/4fbd7636-25af-4638-9587-5a02fdbb0172/r).

Our chatbot still can't remember past interactions on its own, limiting its ability to have coherent, multi-turn conversations. In the next part, we'll add **memory** to address this.


The full code for the graph we've created in this section is reproduced below, replacing our `BasicToolNode` for the prebuilt [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode), and our `route_tools` condition with the prebuilt [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition)

<details>
<summary>Full Code</summary>
    <pre>

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()
```

</pre>
</details>

## Part 3: Adding Memory to the Chatbot

Our chatbot can now use tools to answer user questions, but it doesn't remember the context of previous interactions. This limits its ability to have coherent, multi-turn conversations.

LangGraph solves this problem through **persistent checkpointing**. If you provide a `checkpointer` when compiling the graph and a `thread_id` when calling your graph, LangGraph automatically saves the state after each step. When you invoke the graph again using the same `thread_id`, the graph loads its saved state, allowing the chatbot to pick up where it left off. 

We will see later that **checkpointing** is _much_ more powerful than simple chat memory - it lets you save and resume complex state at any time for error recovery, human-in-the-loop workflows, time travel interactions, and more. But before we get too ahead of ourselves, let's add checkpointing to enable multi-turn conversations.

To get started, create a `MemorySaver` checkpointer.


```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

**Notice** we're using an in-memory checkpointer. This is convenient for our tutorial (it saves it all in-memory). In a production application, you would likely change this to use `SqliteSaver` or `PostgresSaver` and connect to your own DB.

Next define the graph. Now that you've already built your own `BasicToolNode`, we'll replace it with LangGraph's prebuilt `ToolNode` and `tools_condition`, since these do some nice things like parallel API execution. Apart from that, the following is all copied from Part 2.


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
```

Finally, compile the graph with the provided checkpointer.


```python
graph = graph_builder.compile(checkpointer=memory)
```

Notice the connectivity of the graph hasn't changed since Part 2. All we are doing is checkpointing the `State` as the graph works through each node.


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![](data:image/jpg;base64,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)

Now you can interact with your bot! First, pick a thread to use as the key for this conversation.


```python
config = {"configurable": {"thread_id": "1"}}
```

Next, call your chat bot.


```python
user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

**Note:** The config was provided as the **second positional argument** when calling our graph. It importantly is _not_ nested within the graph inputs (`{'messages': []}`).

Let's ask a followup: see if it remembers your name.


```python
user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

**Notice** that we aren't using an external list for memory: it's all handled by the checkpointer! You can inspect the full execution in this [LangSmith trace](https://smith.langchain.com/public/29ba22b5-6d40-4fbe-8d27-b369e3329c84/r) to see what's going on.

Don't believe me? Try this using a different config.


```python
# The only difference is we change the `thread_id` here to "2" instead of "1"
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    # highlight-next-line
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

**Notice** that the **only** change we've made is to modify the `thread_id` in the config. See this call's [LangSmith trace](https://smith.langchain.com/public/51a62351-2f0a-4058-91cc-9996c5561428/r) for comparison. 

By now, we have made a few checkpoints across two different threads. But what goes into a checkpoint? To inspect a graph's `state` for a given config at any time, call `get_state(config)`.


```python
snapshot = graph.get_state(config)
snapshot
```







```python
snapshot.next  # (since the graph ended this turn, `next` is empty. If you fetch a state from within a graph invocation, next tells which node will execute next)
```






The snapshot above contains the current state values, corresponding config, and the `next` node to process. In our case, the graph has reached an `END` state, so `next` is empty.

**Congratulations!** Your chatbot can now maintain conversation state across sessions thanks to LangGraph's checkpointing system. This opens up exciting possibilities for more natural, contextual interactions. LangGraph's checkpointing even handles **arbitrarily complex graph states**, which is much more expressive and powerful than simple chat memory.

In the next part, we'll introduce human oversight to our bot to handle situations where it may need guidance or verification before proceeding.
  
Check out the code snippet below to review our graph from this section.

<details>
<summary>Full Code</summary>
    <pre>

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)
```
</pre>
</pre>
</details>

## Part 4: Human-in-the-loop

Agents can be unreliable and may need human input to successfully accomplish tasks. Similarly, for some actions, you may want to require human approval before running to ensure that everything is running as intended.

LangGraph's [persistence](../../concepts/persistence.md) layer supports human-in-the-loop workflows, allowing execution to pause and resume based on user feedback. The primary interface to this functionality is the [interrupt](../../concepts/human_in_the_loop/#interrupt.md) function. Calling `interrupt` inside a node will pause execution. Execution can be resumed, together with new input from a human, by passing in a [Command](../../concepts/human_in_the_loop/#the-command-primitive.md). `interrupt` is ergonomically similar to Python's built-in `input()`, [with some caveats](../../concepts/human_in_the_loop/#interrupt.md). We demonstrate an example below.

First, start with our existing code from Part 3. We will make one change, which is to add a simple `human_assistance` tool accessible to the chatbot. This tool uses `interrupt` to receive information from a human.


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# highlight-next-line
from langgraph.types import Command, interrupt


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# highlight-next-line
@tool
# highlight-next-line
def human_assistance(query: str) -> str:
    # highlight-next-line
    """Request assistance from a human."""
    # highlight-next-line
    human_response = interrupt({"query": query})
    # highlight-next-line
    return human_response["data"]


tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
```

------

!!! tip

    Check out the [Human-in-the-loop section](../../how-tos/#human-in-the-loop.md) of the How-to Guides for more examples of Human-in-the-loop workflows, including how to [review and edit tool calls](../../how-tos/human_in_the_loop/review-tool-calls/.md) before they are executed.

---------

We compile the graph with a checkpointer, as before:


```python
memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)
```

Visualizing the graph, we recover the same layout as before. We have just added a tool!


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![](data:image/png;base64,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)

Let's now prompt the chatbot with a question that will engage the new `human_assistance` tool:


```python
user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

The chatbot generated a tool call, but then execution has been interrupted! Note that if we inspect the graph state, we see that it stopped at the tools node:


```python
snapshot = graph.get_state(config)
snapshot.next
```






Let's take a closer look at the `human_assistance` tool:

```python
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]
```

Similar to Python's built-in `input()` function, calling `interrupt` inside the tool will pause execution. Progress is persisted based on our choice of [checkpointer](../../concepts/persistence/#checkpointer-libraries.md)-- so if we are persisting with Postgres, we can resume at any time as long as the database is alive. Here we are persisting with the in-memory checkpointer, so we can resume any time as long as our Python kernel is running.

To resume execution, we pass a [Command](../../concepts/human_in_the_loop/#the-command-primitive.md) object containing data expected by the tool. The format of this data can be customized based on our needs. Here, we just need a dict with a key `"data"`:


```python
human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

Our input has been received and processed as a tool message. Review this call's [LangSmith trace](https://smith.langchain.com/public/9f0f87e3-56a7-4dde-9c76-b71675624e91/r) to see the exact work that was done in the above call. Notice that the state is loaded in the first step so that our chatbot can continue where it left off.

**Congrats!** You've used an `interrupt` to add human-in-the-loop execution to your chatbot, allowing for human oversight and intervention when needed. This opens up the potential UIs you can create with your AI systems. Since we have already added a **checkpointer**, as long as the underlying persistence layer is running, the graph can be paused **indefinitely** and resumed at any time as if nothing had happened.

Human-in-the-loop workflows enable a variety of new workflows and user experiences. Check out [this section](../../how-tos/#human-in-the-loop.md) of the How-to Guides for more examples of Human-in-the-loop workflows, including how to [review and edit tool calls](../../how-tos/human_in_the_loop/review-tool-calls/.md) before they are executed.


<details>
<summary>Full Code</summary>
    <pre>

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```
</pre>
</details>

## Part 5: Customizing State

So far, we've relied on a simple state with one entry-- a list of messages. You can go far with this simple state, but if you want to define complex behavior without relying on the message list, you can add additional fields to the state. Here we will demonstrate a new scenario, in which the chatbot is using its search tool to find specific information, and forwarding them to a human for review. Let's have the chatbot research the birthday of an entity. We will add `name` and `birthday` keys to the state:


```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # highlight-next-line
    name: str
    # highlight-next-line
    birthday: str
```

Adding this information to the state makes it easily accessible by other graph nodes (e.g., a downstream node that stores or processes the information), as well as the graph's persistence layer.

Here, we will populate the state keys inside of our `human_assistance` tool. This allows a human to review the information before it is stored in the state. We will again use `Command`, this time to issue a state update from inside our tool. Read more about use cases for `Command` [here](../../concepts/low_level/#using-inside-tools.md).


```python
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

from langgraph.types import Command, interrupt


@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)
```

Otherwise, the rest of our graph is the same:


```python
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

Let's prompt our application to look up the "birthday" of the LangGraph library. We will direct the chatbot to reach out to the `human_assistance` tool once it has the required information. Note that setting `name` and `birthday` in the arguments for the tool, we force the chatbot to generate proposals for these fields.


```python
user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
)
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

We've hit the `interrupt` in the `human_assistance` tool again. In this case, the chatbot failed to identify the correct date, so we can supply it:


```python
human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
)

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

Note that these fields are now reflected in the state:


```python
snapshot = graph.get_state(config)

{k: v for k, v in snapshot.values.items() if k in ("name", "birthday")}
```






This makes them easily accessible to downstream nodes (e.g., a node that further processes or stores the information).

### Manually updating state

LangGraph gives a high degree of control over the application state. For instance, at any point (including when interrupted), we can manually override a key using `graph.update_state`:


```python
graph.update_state(config, {"name": "LangGraph (library)"})
```






If we call `graph.get_state`, we can see the new value is reflected:


```python
snapshot = graph.get_state(config)

{k: v for k, v in snapshot.values.items() if k in ("name", "birthday")}
```






Manual state updates will even [generate a trace](https://smith.langchain.com/public/7ebb7827-378d-49fe-9f6c-5df0e90086c8/r) in LangSmith. If desired, they can also be used to control human-in-the-loop workflows, as described in [this guide](../../how-tos/human_in_the_loop/edit-graph-state/.md). Use of the `interrupt` function is generally recommended instead, as it allows data to be transmitted in a human-in-the-loop interaction independently of state updates.

**Congratulations!** You've added custom keys to the state to facilitate a more complex workflow, and learned how to generate state updates from inside tools.

We're almost done with the tutorial, but there is one more concept we'd like to review before finishing that connects `checkpointing` and `state updates`. 

This section's code is reproduced below for your reference.


<details>
<summary>Full Code</summary>
    <pre>

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt



class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


@tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)


tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```
</pre>
</details>

## Part 6: Time Travel

In a typical chat bot workflow, the user interacts with the bot 1 or more times to accomplish a task. In the previous sections, we saw how to add memory and a human-in-the-loop to be able to checkpoint our graph state and control future responses.

But what if you want to let your user start from a previous response and "branch off" to explore a separate outcome? Or what if you want users to be able to "rewind" your assistant's work to fix some mistakes or try a different strategy (common in applications like autonomous software engineers)?

You can create both of these experiences and more using LangGraph's built-in "time travel" functionality. 

In this section, you will "rewind" your graph by fetching a checkpoint using the graph's `get_state_history` method. You can then resume execution at this previous point in time.

For this, let's use the simple chatbot with tools from [Part 3](#part-3-adding-memory-to-the-chatbot.md):


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

Let's have our graph take a couple steps. Every step will be checkpointed in its state history:


```python
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```


```python
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it!"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

Now that we've had the agent take a couple steps, we can `replay` the full state history to see everything that occurred.


```python
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state
```

**Notice** that checkpoints are saved for every step of the graph. This __spans invocations__ so you can rewind across a full thread's history. We've picked out `to_replay` as a state to resume from. This is the state after the `chatbot` node in the second graph invocation above.

Resuming from this point should call the **action** node next.


```python
print(to_replay.next)
print(to_replay.config)
```

**Notice** that the checkpoint's config (`to_replay.config`) contains a `checkpoint_id` **timestamp**. Providing this `checkpoint_id` value tells LangGraph's checkpointer to **load** the state from that moment in time. Let's try it below:


```python
# The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

Notice that the graph resumed execution from the `**action**` node. You can tell this is the case since the first value printed above is the response from our search engine tool.

**Congratulations!** You've now used time-travel checkpoint traversal in LangGraph. Being able to rewind and explore alternative paths opens up a world of possibilities for debugging, experimentation, and interactive applications.

## Next Steps

Take your journey further by exploring deployment and advanced features:

### Server Quickstart

- **[LangGraph Server Quickstart](../langgraph-platform/local-server.md)**: Launch a LangGraph server locally and interact with it using the REST API and LangGraph Studio Web UI.

### LangGraph Cloud

- **[LangGraph Cloud QuickStart](../../cloud/quick_start.md)**: Deploy your LangGraph app using LangGraph Cloud.

### LangGraph Framework

- **[LangGraph Concepts](../../concepts.md)**: Learn the foundational concepts of LangGraph.  
- **[LangGraph How-to Guides](../../how-tos.md)**: Guides for common tasks with LangGraph.

### LangGraph Platform

Expand your knowledge with these resources:

- **[LangGraph Platform Concepts](../../concepts#langgraph-platform.md)**: Understand the foundational concepts of the LangGraph Platform.  
- **[LangGraph Platform How-to Guides](../../how-tos#langgraph-platform.md)**: Guides for common tasks with LangGraph Platform. 
