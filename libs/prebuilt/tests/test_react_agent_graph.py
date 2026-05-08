from collections.abc import Callable

import pytest
from pydantic import BaseModel
from syrupy import SnapshotAssertion

from langgraph.prebuilt import create_react_agent
from tests.model import FakeToolCallingModel

# APPROVED_MODEL_REGISTRY: Only models listed here are permitted for use in agents.
APPROVED_MODEL_REGISTRY = {
    "FakeToolCallingModel": FakeToolCallingModel,
}

# APPROVED_TOOL_ALLOW_LIST: Only tools listed here are permitted for use in agents.
APPROVED_TOOL_ALLOW_LIST = ["tool"]


def _get_approved_model(model_name: str):
    if model_name not in APPROVED_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not in the approved model registry. "
            f"Approved models: {list(APPROVED_MODEL_REGISTRY.keys())}"
        )
    return APPROVED_MODEL_REGISTRY[model_name]()


def _validate_tools(tools: list) -> list:
    for t in tools:
        tool_name = t.__name__ if callable(t) else str(t)
        if tool_name not in APPROVED_TOOL_ALLOW_LIST:
            raise ValueError(
                f"Tool '{tool_name}' is not in the approved tool allow list. "
                f"Approved tools: {APPROVED_TOOL_ALLOW_LIST}"
            )
    return tools


model = _get_approved_model("FakeToolCallingModel")


def tool() -> None:
    """Testing tool."""
    ...


def pre_model_hook() -> None:
    """Pre-model hook."""
    ...


def post_model_hook() -> None:
    """Post-model hook."""
    ...


class ResponseFormat(BaseModel):
    """Response format for the agent."""

    result: str


@pytest.mark.parametrize("tools", [[], [tool]])
@pytest.mark.parametrize("pre_model_hook", [None, pre_model_hook])
@pytest.mark.parametrize("post_model_hook", [None, post_model_hook])
@pytest.mark.parametrize("response_format", [None, ResponseFormat])
def test_react_agent_graph_structure(
    snapshot: SnapshotAssertion,
    tools: list[Callable],
    pre_model_hook: Callable | None,
    post_model_hook: Callable | None,
    response_format: type[BaseModel] | None,
) -> None:
    validated_tools = _validate_tools(tools)
    agent = create_react_agent(
        model,
        tools=validated_tools,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        response_format=response_format,
    )
    assert agent.get_graph().draw_mermaid(with_styles=False) == snapshot