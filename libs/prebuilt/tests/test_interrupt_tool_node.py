import pytest
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.interrupt import HumanInterruptConfig, InterruptToolNode
from langgraph.types import Command
from tests.model import FakeToolCallingModel


def hello_tool(name: str) -> str:
    """Return a greeting for the provided person."""
    return f"Hello, {name}!"


post_model_hook = InterruptToolNode(
    hello_tool=HumanInterruptConfig(
        allow_accept=True,
        allow_edit=True,
        allow_ignore=True,
        allow_respond=True,
    )
)

default_model = FakeToolCallingModel(
    tool_calls=[
        [
            {
                "name": "hello_tool",
                "args": {"name": "lady gaga"},
                "id": "some-random-id",
            }
        ]
    ]
)


def test_interrupt_surfaced(
    request: pytest.FixtureRequest,
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    agent = create_react_agent(
        default_model,
        [hello_tool],
        checkpointer=sync_checkpointer,
        post_model_hook=post_model_hook,
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    result = agent.invoke({"messages": [("user", "Say hi to lady gaga!")]}, config)

    interrupt_data = result["__interrupt__"]
    assert interrupt_data[0].value == [
        {
            "action_request": {"action": "hello_tool", "args": {"name": "lady gaga"}},
            "config": {
                "allow_accept": True,
                "allow_edit": True,
                "allow_ignore": True,
                "allow_respond": True,
            },
            "description": "Please review tool call for `hello_tool` before execution.",
        }
    ]

    response = agent.invoke(Command(resume={"type": "accept"}), config=config)
    tool_message: ToolMessage = response["messages"][-2]
    assert tool_message.content == "Hello, lady gaga!"
    assert tool_message.name == "hello_tool"


@pytest.mark.parametrize(
    "resume, expected_content",
    [
        ({"type": "accept"}, "Hello, lady gaga!"),
        (
            {"type": "ignore"},
            "User ignored the tool call for `hello_tool` with id some-random-id",
        ),
        (
            {
                "type": "edit",
                "args": {"action": "hello_tool", "args": {"name": "bruno mars"}},
            },
            "Hello, bruno mars!",
        ),
    ],
)
def test_interrupt_resume_variants(
    request: pytest.FixtureRequest,
    sync_checkpointer: BaseCheckpointSaver,
    resume: dict,
    expected_content: str,
) -> None:
    agent = create_react_agent(
        default_model,
        [hello_tool],
        checkpointer=sync_checkpointer,
        post_model_hook=post_model_hook,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    agent.invoke({"messages": [("user", "Say hi to lady gaga!")]}, config)

    response = agent.invoke(Command(resume=resume), config=config)
    tool_message: ToolMessage = response["messages"][-2]
    assert tool_message.name == "hello_tool"
    assert tool_message.content == expected_content

    if resume["type"] == "edit":
        ai_msg = response["messages"][-1]
        assert ai_msg.tool_calls == [
            {
                "name": "hello_tool",
                "args": {"name": "lady gaga"},
                "id": "some-random-id",
                "type": "tool_call",
            }
        ]


def test_resume_with_response(
    request: pytest.FixtureRequest,
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "hello_tool",
                    "args": {"name": "lady gaga"},
                    "id": "some-random-id",
                }
            ],
            [
                {
                    "name": "hello_tool",
                    "args": {"name": "bruno mars"},
                    "id": "some-random-id-2",
                }
            ],
        ]
    )

    agent = create_react_agent(
        model,
        [hello_tool],
        checkpointer=sync_checkpointer,
        post_model_hook=post_model_hook,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    agent.invoke({"messages": [("user", "Say hi to lady gaga!")]}, config)

    # Provide user response
    agent.invoke(
        Command(
            resume={
                "type": "response",
                "args": "actually, please say hello to bruno mars",
            }
        ),
        config=config,
    )

    # Accept the updated call
    response = agent.invoke(Command(resume={"type": "accept"}), config=config)

    assert len(response["messages"]) == 6
    tool_message: ToolMessage = response["messages"][-2]
    assert tool_message.name == "hello_tool"
    assert tool_message.content == "Hello, bruno mars!"


def test_resume_with_type_not_allowed(sync_checkpointer: BaseCheckpointSaver) -> None:
    agent = create_react_agent(
        default_model,
        [hello_tool],
        checkpointer=sync_checkpointer,
        post_model_hook=post_model_hook,
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    agent.invoke({"messages": [("user", "Say hi to lady gaga!")]}, config)

    with pytest.raises(ValueError) as exc_info:
        agent.invoke(Command(resume={"type": "not-allowed"}), config=config)

    assert (
        str(exc_info.value)
        == "Unexpected human response: {'type': 'not-allowed'}. Expected one with `'type'` in ['accept', 'edit', 'response', 'ignore'] based on hello_tool's interrupt configuration."
    )
