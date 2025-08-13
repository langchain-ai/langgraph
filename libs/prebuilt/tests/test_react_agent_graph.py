from typing import Callable, Union

import pytest
from pydantic import BaseModel
from syrupy import SnapshotAssertion

from langgraph.prebuilt import create_react_agent
from tests.model import FakeToolCallingModel

model = FakeToolCallingModel()


def tool() -> None:
    """Testing tool."""
    ...


def tool2() -> None:
    """Another testing tool."""
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
    pre_model_hook: Union[Callable, None],
    post_model_hook: Union[Callable, None],
    response_format: Union[type[BaseModel], None],
) -> None:
    agent = create_react_agent(
        model,
        tools=tools,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        response_format=response_format,
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
            f"response_format: {response_format}"
        ) from e


@pytest.mark.parametrize("tools", [[], [tool, tool2]], ids=["no_tools", "two_tools"])
@pytest.mark.parametrize(
    "pre_model_hook", [None, pre_model_hook], ids=["no_pre_hook", "with_pre_hook"]
)
@pytest.mark.parametrize(
    "post_model_hook", [None, post_model_hook], ids=["no_post_hook", "with_post_hook"]
)
@pytest.mark.parametrize(
    "response_format",
    [None, ResponseFormat],
    ids=["no_response_format", "with_response_format"],
)
def test_react_agent_graph_structure_with_individual_nodes(
    snapshot: SnapshotAssertion,
    tools: list[Callable],
    pre_model_hook: Union[Callable, None],
    post_model_hook: Union[Callable, None],
    response_format: Union[type[BaseModel], None],
) -> None:
    agent = create_react_agent(
        model,
        tools=tools,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        response_format=response_format,
        use_individual_tool_nodes=True,
    )
    assert agent.get_graph().draw_mermaid(with_styles=False) == snapshot
