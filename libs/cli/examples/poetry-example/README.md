# Poetry Example

This example demonstrates using LangGraph CLI with Poetry.

## Setup

1. Install poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Initialize a new Poetry project:
```bash
poetry init
```

3. Add dependencies:
```bash
poetry add langchain-openai langchain-community
poetry add --group dev pytest
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
poetry run langgraph dev
```

The CLI will automatically:
- Detect the pyproject.toml with [tool.poetry] section
- Activate the poetry virtual environment
- Load your dependencies from the virtual environment
- Start the development server

## Alternative: Auto-install

You can also use the `--install-deps` flag to automatically install dependencies:

```bash
langgraph dev --install-deps
```
