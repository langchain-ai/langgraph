# ⚡ Playbook construction ⚡

## Overview

Inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). It can be used without LangChain.

### Key Features

- **Cycles and Branching**: Implement loops and conditionals in your apps.
- **Persistence**: Automatically save state after each step in the graph. Pause and resume the graph execution at any point to support error recovery, human-in-the-loop workflows, time travel and more.
- **Human-in-the-Loop**: Interrupt graph execution to approve or edit next action planned by the agent.
- **Streaming Support**: Stream outputs as they are produced by each node (including token streaming).
- **Does not LangChain**


## Installation

## Example

One of the central concepts is state. Each graph execution creates a state that is passed between nodes in the graph as they execute, and each node updates this internal state with its return value after it executes. The way that the graph updates its internal state is defined by either the type of graph chosen or a custom function.

Let's take a look at a simple example of an agent that can search the web using [Tavily Search API](https://tavily.com/).

```python

# Define the tools for the agent to use
tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools)

model = ChatOpenAI(temperature=0).bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
```

```
'The current weather in San Francisco is as follows:\n- Temperature: 60.1°F (15.6°C)\n- Condition: Partly cloudy\n- Wind: 5.6 mph (9.0 kph) from SSW\n- Humidity: 83%\n- Visibility: 9.0 miles (16.0 km)\n- UV Index: 4.0\n\nFor more details, you can visit [Weather API](https://www.weatherapi.com/).'
```

Now when we pass the same `"thread_id"`, the conversation context is retained via the saved state (i.e. stored list of messages)

```python
final_state = app.invoke(
    {"messages": [HumanMessage(content="what about ny")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
```

```
'The current weather in New York is as follows:\n- Temperature: 20.3°C (68.5°F)\n- Condition: Overcast\n- Wind: 2.2 mph from the north\n- Humidity: 65%\n- Cloud Cover: 100%\n- UV Index: 5.0\n\nFor more details, you can visit [Weather API](https://www.weatherapi.com/).'
```

### Step-by-step Breakdown:

1. <details>
    <summary>Initialize the model and tools.</summary>

    - convert the LangChain tools into the format for OpenAI tool calling using the `.bind_tools()` method.
    - we define the tools we want to use -- a web search tool in our case. It is really easy to create your own tools - see documentation here on how to do that [here](https://python.langchain.com/docs/modules/agents/tools/custom_tools).
   </details>
2. <details>
    <summary>Initialize graph with state.</summary>

    - we initialize graph (`StateGraph`) by passing state schema (in our case `MessagesState`)
    - `MessagesState` is a prebuilt state schema that has one attribute -- a list of LangChain `Message` objects, as well as logic for merging the updates from each node into the state
   </details>
3. <details>
    <summary>Define graph nodes.</summary>

    There are two main nodes we need:
      - The `agent` node: responsible for deciding what (if any) actions to take.
      - The `tools` node that invokes tools: if the agent decides to take an action, this node will then execute that action.
   </details>
4. <details>
    <summary>Define entry point and graph edges.</summary>

      First, we need to set the entry point for graph execution - `agent` node.

      Then we define one normal and one conditional edge. Conditional edge means that the destination depends on the contents of the graph's state (`MessageState`). In our case, the destination is not known until the agent (LLM) decides.

      - Conditional edge: after the agent is called, we should either:
        - a. Run tools if the agent said to take an action, OR
        - b. Finish (respond to the user) if the agent did not ask to run tools
      - Normal edge: after the tools are invoked, the graph should always return to the agent to decide what to do next
   </details>
5. <details>
    <summary>Compile the graph.</summary>

    - When we compile the graph, we turn it into a LangChain [Runnable](https://python.langchain.com/v0.2/docs/concepts/#runnable-interface), which automatically enables calling `.invoke()`, `.stream()` and `.batch()` with your inputs
    - We can also optionally pass checkpointer object for persisting state between graph runs, and enabling memory, human-in-the-loop workflows, time travel and more. In our case we use `MemorySaver` - a simple in-memory checkpointer
    </details>
6. <details>
   <summary>Execute the graph.</summary>

    1. LangGraph adds the input message to the internal state, then passes the state to the entrypoint node, `"agent"`.
    2. The `"agent"` node executes, invoking the chat model.
    3. The chat model returns an `AIMessage`. LangGraph adds this to the state.
    4. Graph cycles the following steps until there are no more `tool_calls` on `AIMessage`:
      - If `AIMessage` has `tool_calls`, `"tools"` node executes
      - The `"agent"` node executes again and returns `AIMessage`
    5. Execution progresses to the special `END` value and outputs the final state.
    And as a result, we get a list of all our chat messages as output.
   </details>

