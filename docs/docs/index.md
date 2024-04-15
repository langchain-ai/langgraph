# 🦜🕸️LangGraph

[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)

⚡ Build language agents as graphs ⚡


## Overview

Suppose you're building a customer support assistant. You want your assistant to:

1. Try to answer user questions using a knowledge base
2. Escalate to a human if it's not confident in its answer
3. Relay the human's resolution back to the user
4. Remember the full conversation context across multiple user messages

With raw LLMs, the code to control the agentic loop, conversation state, route between the chatbot and human, and checkpoint the full application state can get complex.

LangGraph makes it simple. First install:

```shell
pip install -U langgraph
```

Then define your assistant:

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_anthropic


# Define the chatbot state 
class ChatbotState(TypedDict):
    conversation_history: Annotated[ConversationHistory, operator.add]
    pending_human_request: Optional[HumanRequest]

# Create nodes for the chatbot and human
def chatbot(state: ChatbotState):
    # TODO

def human(state: ChatbotState):
    # TODO

# Create the graph
graph = StateGraph(ChatbotState)
graph.add_node("chatbot", chatbot)
graph.add_node("human", human)

# Define routing logic between chatbot and human
def should_escalate(state):
    if state['pending_human_request']:
        return "human"
    else:
        return "chatbot"

graph.add_conditional_edges("chatbot", should_escalate, {
    "human": "human", 
    "chatbot": "chatbot"
})

graph.add_edge("human", "chatbot")

memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)
# Run the graph
result = app.invoke(new_user_message)
```

The graph handles all the hard parts:

- `conversation_history` in the state contains the assistant's "memory"
- Conditional edges enable dynamic routing between the chatbot and human based on the chatbot's confidence
- Persistence makes it easy to route to a human so they can respond and resume at any time

With LangGraph, you can build complex, stateful agents without getting bogged down in manual state and interrupt management. Just define your nodes, edges, and state schema - and let the graph take care of the rest.

## Concepts

- [Graphs](concepts.md#graphs)
- [State](concepts.md#state): The data structure passed between nodes, allowing you to persist context 
- [Nodes](concepts.#nodes): The building blocks of your graph - LLMs, tools, or custom logic 
- [Edges](concepts.md#edges): The connections that define the flow of data between your nodes
- [Conditional Edges](concepts.md#conditional_edges): Special edges that let you dynamically route between nodes based on state
- [Persistence](concepts.md#persistence): Save and resume your graph's state for long-running applications

## How-To Guides

Check out the [How-To Guides](how-tos/index.md) for instructions on handling common tasks with LangGraph.

- Manage State
- Tool Integration  
- Human-in-the-Loop
- Async Execution
- Streaming Responses
- Subgraphs & Branching
- Persistence, Visualization, Time Travel 
- Benchmarking

## Tutorials

Consult the [Tutorials](tutorials/index.md) to learn more about implementing advanced 

- **Agent Executors**: Chat and Langchain agents
- **Planning Agents**: Plan-and-Execute, ReWOO, LLMCompiler  
- **Reflection & Critique**: Improving quality via reflection
- **Multi-Agent Systems**: Collaboration, supervision, teams
- **Research & QA**: Web research, retrieval-augmented QA  
- **Applications**: Chatbots, code assist, web tasks
- **Evaluation & Analysis**: Simulation, self-discovery, swarms

## Why LangGraph?

LangGraph extends the core strengths of LangChain Runnables (shared interface for streaming, async, and batch calls) to make it easy to:

- Seamless state management across multiple turns of conversation or tool usage
- The ability to flexibly route between nodes based on dynamic criteria 
- Smooth switching between LLMs and human intervention  
- Persistence for long-running, multi-session applications

If you're building a straightforward DAG,, LangChain expression language is a great fit. But for more complex, stateful applications with nonlinear flows, LangGraph is the perfect tool for the job.