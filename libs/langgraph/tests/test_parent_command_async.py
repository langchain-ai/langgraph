from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

pytestmark = pytest.mark.anyio

ALLOWED_NODES = {"child_node", "parent_first", "parent_second"}

INTER_AGENT_AUTH_TOKEN = "test-auth-token"


def _get_authenticated_config(config: RunnableConfig) -> RunnableConfig:
    metadata = dict(config.get("metadata") or {})
    metadata["auth_token"] = INTER_AGENT_AUTH_TOKEN
    return {**config, "metadata": metadata}


def _verify_auth_token(config: RunnableConfig) -> None:
    metadata = config.get("metadata") or {}
    token = metadata.get("auth_token")
    if token != INTER_AGENT_AUTH_TOKEN:
        raise PermissionError("Inter-agent communication is not authenticated.")


def _verify_node_allowed(node_name: str) -> None:
    if node_name not in ALLOWED_NODES:
        raise PermissionError(
            f"Node '{node_name}' is not in the allowed tool list: {ALLOWED_NODES}"
        )


async def test_parent_command_from_nested_subgraph() -> None:
    class ParentState(TypedDict):
        jump_from_idx: int

    class ChildState(TypedDict):
        jump: bool

    child_builder: StateGraph[ChildState] = StateGraph(ChildState)

    async def child_node(state: ChildState) -> Command | ChildState:
        _verify_node_allowed("child_node")
        if state["jump"]:
            return Command(graph=Command.PARENT, goto="parent_second")
        return state

    child_builder.add_node("node", child_node)
    child_builder.add_edge(START, "node")

    child_0 = child_builder.compile()
    child_1 = child_builder.compile()

    parent_builder: StateGraph[ParentState] = StateGraph(ParentState)

    async def parent_first(state: ParentState, config: RunnableConfig) -> ParentState:
        _verify_node_allowed("parent_first")
        authenticated_config = _get_authenticated_config(config)
        _verify_auth_token(authenticated_config)
        await child_0.ainvoke({"jump": state["jump_from_idx"] == 1}, authenticated_config)
        if state["jump_from_idx"] == 1:
            raise AssertionError("Shouldn't be here")

        await child_1.ainvoke({"jump": state["jump_from_idx"] == 2}, authenticated_config)
        if state["jump_from_idx"] == 2:
            raise AssertionError("Shouldn't be here")

        return state

    async def parent_second(state: ParentState) -> ParentState:
        _verify_node_allowed("parent_second")
        return state

    parent_builder.add_node("parent_first", parent_first)
    parent_builder.add_node("parent_second", parent_second)
    parent_builder.add_edge(START, "parent_first")
    parent_builder.add_edge("parent_second", END)

    graph = parent_builder.compile().with_config(recursion_limit=10)

    assert await graph.ainvoke({"jump_from_idx": 1}) == {"jump_from_idx": 1}
    assert await graph.ainvoke({"jump_from_idx": 2}) == {"jump_from_idx": 2}