import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from typing_extensions import TypedDict

from langgraph.advanced_graph import (
    AdvancedStateGraph,
    Context,
    any_of,
    channel_condition,
    timer_condition,
)
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, Send

pytestmark = pytest.mark.anyio


class MainAgentState(TypedDict):
    input: str
    output: list[str]
    done: str | None


class SubAgentState(TypedDict):
    input: str
    output: str


@dataclass(frozen=True)
class Decision:
    type: Literal["end", "sub_agent", "tool"]
    sub_agent: str | None = None
    tool: str | None = None
    complete: str | None = None


class MockPlanner:
    def __init__(self) -> None:
        self.responses: list[list[Decision]] = []
        self._idx = 0

    async def ainvoke(self, _: MainAgentState) -> list[Decision]:
        if self._idx >= len(self.responses):
            return []
        response = self.responses[self._idx]
        self._idx += 1
        return response


def build_sub_agent() -> Any:
    # Sub-agent uses the regular/simple StateGraph API.
    sub_agent = StateGraph(SubAgentState)

    async def research_node(state: SubAgentState) -> dict[str, str]:
        # Make timing deterministic for the prototype flow assertions.
        if state["input"] == "research lunch options":
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0.09)
        return {"output": f"research sub agent completed for: {state['input']}"}

    sub_agent.add_node("research_node", research_node)
    sub_agent.add_edge(START, "research_node")
    sub_agent.add_edge("research_node", END)
    return sub_agent.compile()


async def test_async_sub_graph() -> None:
    planner = MockPlanner()
    sub_agent = build_sub_agent()

    async def llm_node(state: MainAgentState) -> Command:
        # Planner decides whether to call a tool, spawn a sub-agent, or finish.
        decisions = await planner.ainvoke(state)
        sends: list[Send] = []
        for decision in decisions:
            if decision.type == "end":
                # NOTE: this can be simplified further in the future with a dedicated
                # complete primitive, instead of routing to a finish node manually.
                return Command(
                    goto=Send(
                        order_food_node,
                        decision.complete or "order flow completed",
                    )
                )
            if decision.type == "sub_agent" and decision.sub_agent:
                sends.append(Send("sub_agent_node", decision.sub_agent))
            if decision.type == "tool" and decision.tool:
                sends.append(Send("tool_node", decision.tool))
        # Keep the main loop responsive: wait for one inbound message and continue.
        sends.append(Send("wait_node", state))
        return Command(goto=sends)

    async def wait_node(ctx: Context, state: MainAgentState) -> Command:
        # Lightweight interrupt: only this node blocks for the next relevant signal.
        event = await ctx.wait_for(
            any_of(
                channel_condition("tool_completion_channel"),
                channel_condition("subagent_completion_channel"),
                channel_condition("user_input_channel"),
                timer_condition(minutes=1),
            )
        )
        if event["condition"] == "channel":
            channel = event["channel"]
            payload = event["value"]
            if channel == "tool_completion_channel":
                state["output"].append(f"tool: {payload}")
            elif channel == "subagent_completion_channel":
                state["output"].append(f"sub_agent: {payload}")
            elif channel == "user_input_channel":
                state["output"].append(f"user_input: {payload}")
        else:
            state["output"].append("timer: no updates yet")
        # Loop back to planner with updated output.
        return Command(goto=Send("llm_node", state))

    async def tool_node(ctx: Context, tool_input: str) -> None:
        await asyncio.sleep(0.03)
        # Fire-and-forget style completion: publish result to inbox and exit.
        # (i.e., just complete without explicitly going to a next node)
        ctx.publish_to_channel(
            "tool_completion_channel",
            f"tool completed for: {tool_input}",
        )

    async def sub_agent_node(ctx: Context, sub_agent_input: str) -> None:
        # Sub-agent remains a regular StateGraph, compiled independently.
        sub_agent_output = await sub_agent.ainvoke(
            {"input": sub_agent_input, "output": ""}
        )
        # Same pattern as tool node: publish result and complete current node.
        ctx.publish_to_channel(
            "subagent_completion_channel",
            sub_agent_output["output"],
        )

    async def order_food_node(complete_message: str) -> dict[str, str]:
        return {"done": complete_message}

    advanced_flow = AdvancedStateGraph(MainAgentState)
    # Default behavior is an unbounded async channel (maxsize=None).
    advanced_flow.add_async_channel("tool_completion_channel", str)
    advanced_flow.add_async_channel("subagent_completion_channel", str)
    advanced_flow.add_async_channel("user_input_channel", str)
    advanced_flow.add_entry_node(llm_node)
    advanced_flow.node(wait_node)
    advanced_flow.node(tool_node)
    advanced_flow.node(sub_agent_node)
    advanced_flow.add_finish_node(order_food_node)
    main_agent = advanced_flow.compile()

    planner.responses = [
        [
            # First planner pass triggers one sub-agent + one tool.
            Decision(type="sub_agent", sub_agent="research lunch options"),
            Decision(type="tool", tool="slack_tool"),
        ],
        # Second planner pass triggers another sub-agent.
        [Decision(type="sub_agent", sub_agent="find vegetarian fallback")],
        [],
        [],
        # Final pass decides to end.
        [Decision(type="end", complete="order submitted")],
    ]

    handler = await main_agent.astart(
        {"input": "help me get something for lunch", "output": [], "done": None}
    )

    # External input can be injected while graph execution is in progress.
    await asyncio.sleep(0.01)
    await handler.apublish_to_channel("user_input_channel", "No spicy food please")
    result = await handler.aresult()

    assert result == {
        "input": "help me get something for lunch",
        "output": [
            "user_input: No spicy food please",
            "tool: tool completed for: slack_tool",
            "sub_agent: research sub agent completed for: research lunch options",
            "sub_agent: research sub agent completed for: find vegetarian fallback",
        ],
        "done": "order submitted",
    }
