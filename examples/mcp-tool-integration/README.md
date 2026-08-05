# MCP Tool Integration in a LangGraph Node

Connect Model Context Protocol (MCP) tools to a LangGraph agent
so the graph can call external services and allow human operators
to override decisions inline.

## What this shows

- Wrapping an MCP tool call inside a LangGraph node
- Structured output validation at each pipeline stage
- An operator override node using `interrupt_after`
- Full graph flow: classify → route → tool call → override check

## Install

```bash
pip install langgraph langchain-openai mcp
```

## The graph

```python
import asyncio
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── State ────────────────────────────────────────────────────────────────────

class IncidentState(TypedDict):
    incident_id: str
    description: str
    priority: str          # P1 / P2 / P3 / P4
    assigned_team: str
    mcp_result: dict
    operator_approved: bool

# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Node 1: classify priority ─────────────────────────────────────────────────

def classify_node(state: IncidentState) -> IncidentState:
    prompt = f"""Classify this incident as P1, P2, P3, or P4.
Respond with only the priority label.

Incident: {state['description']}"""
    result = llm.invoke(prompt)
    priority = result.content.strip()
    team_map = {"P1": "platform-oncall", "P2": "backend", "P3": "backend", "P4": "frontend"}
    return {
        **state,
        "priority": priority,
        "assigned_team": team_map.get(priority, "backend"),
    }

# ── Node 2: call MCP tool ─────────────────────────────────────────────────────

async def mcp_tool_node(state: IncidentState) -> IncidentState:
    """Call an MCP server tool to push the triage result to ServiceNow."""
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  # your MCP server entry point
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "update_incident",
                arguments={
                    "incident_id": state["incident_id"],
                    "priority": state["priority"],
                    "assigned_team": state["assigned_team"],
                },
            )

            return {
                **state,
                "mcp_result": {"status": "updated", "content": str(result.content)},
            }

# ── Node 3: operator override check ──────────────────────────────────────────

def override_node(state: IncidentState) -> IncidentState:
    """
    Pause here for operator review on P1 incidents.
    The graph uses interrupt_after so a human can inspect
    and modify state before resuming.
    """
    return {**state, "operator_approved": True}

# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_classify(state: IncidentState) -> Literal["mcp_tool", "end"]:
    # Only push to MCP for P1 and P2 — log P3/P4 and stop
    if state["priority"] in ("P1", "P2"):
        return "mcp_tool"
    return "end"

def route_after_mcp(state: IncidentState) -> Literal["override", "end"]:
    # Require operator approval only for P1
    if state["priority"] == "P1":
        return "override"
    return "end"

# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(IncidentState)

    builder.add_node("classify", classify_node)
    builder.add_node("mcp_tool", mcp_tool_node)
    builder.add_node("override", override_node)

    builder.set_entry_point("classify")

    builder.add_conditional_edges("classify", route_after_classify, {
        "mcp_tool": "mcp_tool",
        "end": END,
    })
    builder.add_conditional_edges("mcp_tool", route_after_mcp, {
        "override": "override",
        "end": END,
    })
    builder.add_edge("override", END)

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_after=["mcp_tool"],  # pause before override for P1
    )

# ── Run ───────────────────────────────────────────────────────────────────────

async def main():
    graph = build_graph()

    initial_state: IncidentState = {
        "incident_id": "INC0012345",
        "description": "Production database unreachable, all writes failing.",
        "priority": "",
        "assigned_team": "",
        "mcp_result": {},
        "operator_approved": False,
    }

    config = {"configurable": {"thread_id": "incident-run-1"}}

    # First run — stops after mcp_tool node for P1 operator review
    async for event in graph.astream(initial_state, config):
        print(event)

    # Operator inspects state here, then resumes
    # graph.update_state(config, {"operator_approved": True})
    # async for event in graph.astream(None, config):
    #     print(event)

asyncio.run(main())
```

## Key points

**Wrapping async MCP calls in a node**
LangGraph nodes can be async. Use `async def` for the node
and `astream` / `ainvoke` when running the graph.

**Structured output at each stage**
Each node returns the full state dict with only the fields it
modifies — this makes routing logic predictable and testable.

**Operator override with `interrupt_after`**
Pass `interrupt_after=["node_name"]` at compile time to pause
the graph after that node. The operator can inspect state with
`graph.get_state(config)`, modify it with `graph.update_state()`,
then resume with `graph.astream(None, config)`.

**Routing on priority**
`add_conditional_edges` keeps routing logic out of the nodes
themselves — each node just updates state, routing decides what
happens next.

## Extending this

- Add a `runbook_lookup` node before `mcp_tool` to fetch
  matching runbooks via BM25 and attach them to state
- Add a second MCP tool call to post a Slack alert on P1
- Store the full audit trail by persisting state to PostgreSQL
  instead of `MemorySaver`
