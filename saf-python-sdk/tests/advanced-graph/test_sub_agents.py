import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from typing_extensions import TypedDict

from saf_python_sdk.advanced_graph import (
    AdvancedStateGraph,
    Context,
    any_of,
    channel_condition,
    timer_condition,
)
from saf_python_sdk.types import Command, Send

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


class MockLLM:
    def __init__(self) -> None:
        self.responses: list[list[Decision]] = []
        self._idx = 0

    async def ainvoke(self, _: MainAgentState) -> list[Decision]:
        if self._idx >= len(self.responses):
            return []
        response = self.responses[self._idx]
        self._idx += 1
        return response


class FakeSubAgent:
    async def ainvoke(self, state: SubAgentState) -> SubAgentState:
        await asyncio.sleep(5)
        return {"input": state["input"], "output": f"research sub agent completed for: {state['input']}"}


def build_sub_agent() -> Any:
    return FakeSubAgent()


def build_main_agent(planner: MockLLM, sub_agent: Any) -> Any:
    async def llm_node(state: MainAgentState) -> Command:
        decisions = await planner.ainvoke(state)
        sends: list[Send] = []
        for decision in decisions:
            if decision.type == "end":
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
        sends.append(Send("wait_node", None))
        return Command(goto=sends)

    async def wait_node(ctx: Context, state: MainAgentState) -> Command:
        event = await ctx.wait_for(
            any_of(
                channel_condition("tool_completion_channel"),
                channel_condition("subagent_completion_channel"),
                channel_condition("user_input_channel"),
                timer_condition(seconds=1),
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
            return Command(update=state, goto=Send("llm_node", None))
        else:
            state["output"].append("timer: no updates yet")
            return Command(update=state, goto=Send("wait_node", None))

    async def tool_node(ctx: Context, tool_input: str) -> None:
        await asyncio.sleep(0.1)
        ctx.publish_to_channel(
            "tool_completion_channel",
            f"tool completed for: {tool_input}",
        )

    async def sub_agent_node(ctx: Context, sub_agent_input: str) -> None:
        sub_agent_output = await sub_agent.ainvoke(
            {"input": sub_agent_input, "output": ""}
        )
        ctx.publish_to_channel(
            "subagent_completion_channel",
            sub_agent_output["output"],
        )

    async def order_food_node(input: str, state: MainAgentState) -> dict[str, Any]:
        complete_message = input
        return {
            "done": complete_message,
            "output": [*state["output"], f"order_food: {complete_message}"],
        }

    advanced_flow = AdvancedStateGraph(MainAgentState)
    advanced_flow.add_async_channel("tool_completion_channel", str)
    advanced_flow.add_async_channel("subagent_completion_channel", str)
    advanced_flow.add_async_channel("user_input_channel", str)
    advanced_flow.add_entry_node(llm_node)
    advanced_flow.add_node(wait_node)
    advanced_flow.add_node(tool_node)
    advanced_flow.add_node(sub_agent_node)
    advanced_flow.add_finish_node(order_food_node)

    return advanced_flow.compile()


async def test_async_sub_graph() -> None:
    llm = MockLLM()
    sub_agent = build_sub_agent()
    main_agent = build_main_agent(llm, sub_agent)

    llm.responses = [
        [
            Decision(type="sub_agent", sub_agent="research lunch options"),
            Decision(type="tool", tool="slack_tool"),
        ],
        [],
        [],
        [Decision(type="sub_agent", sub_agent="find vegetarian fallback")],
        [Decision(type="end", complete="order submitted")],
    ]

    handler = await main_agent.astart(
        {"input": "help me get something for lunch", "output": [], "done": None}
    )

    await asyncio.sleep(0.01)
    await handler.apublish_to_channel("user_input_channel", "No spicy food please")
    result = await handler.aresult()

    assert result["input"] == "help me get something for lunch"
    assert result["done"] == "order submitted"

    output = result["output"]
    assert output.count("timer: no updates yet") >= 3
    assert "user_input: No spicy food please" in output
    assert "tool: tool completed for: slack_tool" in output
    assert (
        "sub_agent: research sub agent completed for: research lunch options" in output
    )
    assert (
        "sub_agent: research sub agent completed for: find vegetarian fallback"
        in output
    )
    assert output[-1] == "order_food: order submitted"

    first_sub_idx = output.index(
        "sub_agent: research sub agent completed for: research lunch options"
    )
    second_sub_idx = output.index(
        "sub_agent: research sub agent completed for: find vegetarian fallback"
    )
    order_food_idx = output.index("order_food: order submitted")
    assert first_sub_idx < second_sub_idx < order_food_idx
    assert llm._idx == len(llm.responses)

