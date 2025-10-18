# uv Example

This example demonstrates using LangGraph CLI with uv.

## Setup

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Initialize a new uv project:
```bash
uv init my-langgraph-app
cd my-langgraph-app
```

3. Add dependencies:
```bash
uv add langchain-openai langchain-community
uv add --dev pytest
```

4. Create your LangGraph configuration:
```json
{
  "dependencies": [
    "langchain_openai",
    "langchain_community",
    "./graphs"
  ],
  "graphs": {
    "agent": "./graphs/agent.py:graph"
  }
}
```

5. Create your graph:
```python
# graphs/agent.py
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    messages: list

def agent_node(state: State):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_edge("agent", END)
graph = graph.compile()
```

## Running

Now you can run the development server with:

```bash
uv run langgraph dev
```

The CLI will automatically:
- Detect the uv.lock file or pyproject.toml with [tool.uv] section
- Activate the uv virtual environment
- Load your dependencies from the virtual environment
- Start the development server

## Alternative: Auto-install

You can also use the `--install-deps` flag to automatically install dependencies:

```bash
langgraph dev --install-deps
```
