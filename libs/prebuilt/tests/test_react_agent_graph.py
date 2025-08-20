from typing import Callable, Union

import pytest
from pydantic import BaseModel
from syrupy import SnapshotAssertion

from langgraph.prebuilt import create_agent
from tests.model import FakeToolCallingModel

model = FakeToolCallingModel()


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
def test_react_agent_graph_structure(
    snapshot: SnapshotAssertion,
    tools: list[Callable],
    pre_model_hook: Union[Callable, None],
    post_model_hook: Union[Callable, None],
) -> None:
    agent = create_agent(
        model,
        tools=tools,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
    )
    try:
        assert agent.get_graph().draw_mermaid(with_styles=False) == snapshot
    except Exception as e:
        raise ValueError(
            "The graph structure has changed. Please update the snapshot."
            "Configuration used:\n"
            f"tools: {tools}, "
            f"pre_model_hook: {pre_model_hook}, "
            f"post_model_hook: {post_model_hook}, "
        ) from e
