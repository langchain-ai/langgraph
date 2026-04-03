# Serve a LangGraph agent as an A2A endpoint

This guide shows how to attach a declarative **A2A Agent Card** to a LangGraph
graph and serve it as an [Agent-to-Agent (A2A)](https://github.com/google/A2A)
compatible endpoint.

## Prerequisites

```bash
pip install langgraph[a2a]
```

## 1. Build your agent

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools=[search, calculator])
```

## 2. Declare the Agent Card

```python
from langgraph.a2a import AgentCard, AgentSkill

card = AgentCard(
    name="Research Assistant",
    description="Searches the web and performs calculations.",
    url="https://research.mycompany.com",
    org="My Company",
    skills=[
        AgentSkill(
            id="web-search",
            name="Web Search",
            description="Search the web for current information.",
            tags=["search", "research"],
            examples=["What happened in AI this week?"],
        ),
        AgentSkill(
            id="calculator",
            name="Calculator",
            description="Perform arithmetic and unit conversions.",
            tags=["math"],
            examples=["Convert 42 miles to kilometers"],
        ),
    ],
)
```

## 3. Attach the card

```python
agent = agent.with_agent_card(card)
```

The card is now accessible at `agent.agent_card` and can be serialized:

```python
import json
print(json.dumps(agent.agent_card.to_dict(), indent=2))
```

## 4. Serve the A2A endpoint

```python
agent.serve_a2a(host="0.0.0.0", port=8080)
```

This starts a Starlette server exposing:

| Route | Method | Description |
|---|---|---|
| `/.well-known/agent.json` | GET | Agent Card discovery |
| `/` | POST | A2A `tasks/send` endpoint |

## 5. Verify with curl

**Discover the agent:**

```bash
curl http://localhost:8080/.well-known/agent.json | jq .
```

**Send a task:**

```bash
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "id": "task-001",
    "message": {
      "role": "user",
      "parts": {"text": "What happened in AI this week?"}
    }
  }'
```

## API reference

- `AgentCard` — declarative metadata (name, description, url, skills, auth)
- `AgentSkill` — individual skill with id, name, description, tags, examples
- `graph.with_agent_card(card)` — attach a card (fluent, returns self)
- `graph.serve_a2a(host, port, config)` — start the A2A server
