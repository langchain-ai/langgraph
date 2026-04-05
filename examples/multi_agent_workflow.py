"""
Multi-Agent Workflow with LangGraph
Author: Rehan Malik

Demonstrates supervisor-worker pattern for orchestrating
specialized AI agents with tool calling.
"""

from typing import TypedDict
from dataclasses import dataclass, field


@dataclass
class Tool:
    name: str
    description: str
    handler: callable = None

    def execute(self, **kwargs) -> dict:
        if self.handler:
            return self.handler(**kwargs)
        return {"status": "simulated", "tool": self.name, "args": kwargs}


class WorkflowState(TypedDict):
    task: str
    plan: list
    current_step: int
    results: list
    final_output: str
    status: str


def supervisor(state: WorkflowState) -> WorkflowState:
    """Supervisor agent - decomposes task and assigns to workers."""
    task = state["task"]
    state["plan"] = [
        {"step": "research", "agent": "researcher", "input": task},
        {"step": "analyze", "agent": "analyst", "input": "research results"},
        {"step": "synthesize", "agent": "writer", "input": "analysis"},
    ]
    state["current_step"] = 0
    state["status"] = "planning_complete"
    return state


def researcher(state: WorkflowState) -> WorkflowState:
    """Research agent - gathers relevant information."""
    step = state["plan"][state["current_step"]]
    result = {
        "agent": "researcher",
        "findings": f"Research on: {step['input']}",
        "sources": 5,
        "confidence": 0.88
    }
    state["results"].append(result)
    state["current_step"] += 1
    return state


def analyst(state: WorkflowState) -> WorkflowState:
    """Analyst agent - processes and analyzes data."""
    prev_result = state["results"][-1] if state["results"] else {}
    result = {
        "agent": "analyst",
        "analysis": f"Analysis of {prev_result.get('findings', 'N/A')}",
        "key_insights": 3,
        "confidence": 0.91
    }
    state["results"].append(result)
    state["current_step"] += 1
    return state


def writer(state: WorkflowState) -> WorkflowState:
    """Writer agent - creates final output."""
    all_findings = [r.get("findings", r.get("analysis", "")) for r in state["results"]]
    state["final_output"] = f"Synthesized report from {len(state['results'])} agent outputs"
    state["status"] = "complete"
    return state


def route_next(state: WorkflowState) -> str:
    """Route to next agent based on plan."""
    if state["current_step"] >= len(state["plan"]):
        return "complete"
    return state["plan"][state["current_step"]]["agent"]


if __name__ == "__main__":
    state = WorkflowState(
        task="Analyze the impact of RAG on enterprise search quality",
        plan=[], current_step=0, results=[],
        final_output="", status="pending"
    )

    state = supervisor(state)
    print(f"Plan: {len(state['plan'])} steps")

    state = researcher(state)
    state = analyst(state)
    state = writer(state)

    print(f"Status: {state['status']}")
    print(f"Output: {state['final_output']}")
    print(f"Agent results: {len(state['results'])}")
