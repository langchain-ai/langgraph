import json
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, Type, Union

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as dec_tool
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import ToolNode, ValidationNode, create_react_agent
from langgraph.prebuilt.tool_node import InjectedState
from tests.conftest import (
    ALL_CHECKPOINTERS_ASYNC,
    ALL_CHECKPOINTERS_SYNC,
    awith_checkpointer,
)
from tests.messages import _AnyIdHumanMessage

pytestmark = pytest.mark.anyio


class FakeToolCallingModel(BaseChatModel):
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_string = "-".join([m.content for m in messages])
        message = AIMessage(content=messages_string, id="0")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if len(tools) > 0:
            raise ValueError("Not supported yet!")
        return self


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_no_modifier(request: pytest.FixtureRequest, checkpointer_name: str) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        "checkpointer_" + checkpointer_name
    )
    model = FakeToolCallingModel()

    agent = create_react_agent(model, [], checkpointer=checkpointer)
    inputs = [HumanMessage("hi?")]
    thread = {"configurable": {"thread_id": "123"}}
    response = agent.invoke({"messages": inputs}, thread, debug=True)
    expected_response = {"messages": inputs + [AIMessage(content="hi?", id="0")]}
    assert response == expected_response

    if checkpointer:
        saved = checkpointer.get_tuple(thread)
        assert saved is not None
        assert saved.checkpoint["channel_values"] == {
            "messages": [
                _AnyIdHumanMessage(content="hi?"),
                AIMessage(content="hi?", id="0"),
            ],
            "agent": "agent",
        }
        assert saved.metadata == {
            "parents": {},
            "source": "loop",
            "writes": {"agent": {"messages": [AIMessage(content="hi?", id="0")]}},
            "step": 1,
        }
        assert saved.pending_writes == []


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_no_modifier_async(checkpointer_name: str) -> None:
    async with awith_checkpointer(checkpointer_name) as checkpointer:
        model = FakeToolCallingModel()

        agent = create_react_agent(model, [], checkpointer=checkpointer)
        inputs = [HumanMessage("hi?")]
        thread = {"configurable": {"thread_id": "123"}}
        response = await agent.ainvoke({"messages": inputs}, thread, debug=True)
        expected_response = {"messages": inputs + [AIMessage(content="hi?", id="0")]}
        assert response == expected_response

        if checkpointer:
            saved = await checkpointer.aget_tuple(thread)
            assert saved is not None
            assert saved.checkpoint["channel_values"] == {
                "messages": [
                    _AnyIdHumanMessage(content="hi?"),
                    AIMessage(content="hi?", id="0"),
                ],
                "agent": "agent",
            }
            assert saved.metadata == {
                "parents": {},
                "source": "loop",
                "writes": {"agent": {"messages": [AIMessage(content="hi?", id="0")]}},
                "step": 1,
            }
            assert saved.pending_writes == []


def test_passing_two_modifiers():
    model = FakeToolCallingModel()
    with pytest.raises(ValueError):
        create_react_agent(model, [], messages_modifier="Foo", state_modifier="Bar")


def test_system_message_modifier():
    model = FakeToolCallingModel()
    messages_modifier = SystemMessage(content="Foo")
    agent_1 = create_react_agent(model, [], messages_modifier=messages_modifier)
    agent_2 = create_react_agent(model, [], state_modifier=messages_modifier)
    for agent in [agent_1, agent_2]:
        inputs = [HumanMessage("hi?")]
        response = agent.invoke({"messages": inputs})
        expected_response = {
            "messages": inputs + [AIMessage(content="Foo-hi?", id="0")]
        }
        assert response == expected_response


def test_system_message_string_modifier():
    model = FakeToolCallingModel()
    messages_modifier = "Foo"
    agent_1 = create_react_agent(model, [], messages_modifier=messages_modifier)
    agent_2 = create_react_agent(model, [], state_modifier=messages_modifier)
    for agent in [agent_1, agent_2]:
        inputs = [HumanMessage("hi?")]
        response = agent.invoke({"messages": inputs})
        expected_response = {
            "messages": inputs + [AIMessage(content="Foo-hi?", id="0")]
        }
        assert response == expected_response


def test_callable_messages_modifier():
    model = FakeToolCallingModel()

    def messages_modifier(messages):
        modified_message = f"Bar {messages[-1].content}"
        return [HumanMessage(content=modified_message)]

    agent = create_react_agent(model, [], messages_modifier=messages_modifier)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Bar hi?", id="0")]}
    assert response == expected_response


def test_callable_state_modifier():
    model = FakeToolCallingModel()

    def state_modifier(state):
        modified_message = f"Bar {state['messages'][-1].content}"
        return [HumanMessage(content=modified_message)]

    agent = create_react_agent(model, [], state_modifier=state_modifier)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Bar hi?", id="0")]}
    assert response == expected_response


def test_runnable_messages_modifier():
    model = FakeToolCallingModel()

    messages_modifier = RunnableLambda(
        lambda messages: [HumanMessage(content=f"Baz {messages[-1].content}")]
    )

    agent = create_react_agent(model, [], messages_modifier=messages_modifier)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Baz hi?", id="0")]}
    assert response == expected_response


def test_runnable_state_modifier():
    model = FakeToolCallingModel()

    state_modifier = RunnableLambda(
        lambda state: [HumanMessage(content=f"Baz {state['messages'][-1].content}")]
    )

    agent = create_react_agent(model, [], state_modifier=state_modifier)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Baz hi?", id="0")]}
    assert response == expected_response


async def test_tool_node():
    def tool1(some_val: int, some_other_val: str) -> str:
        """Tool 1 docstring."""
        if some_val == 0:
            raise ValueError("Test error")
        return f"{some_val} - {some_other_val}"

    async def tool2(some_val: int, some_other_val: str) -> str:
        """Tool 2 docstring."""
        if some_val == 0:
            raise ValueError("Test error")
        return f"tool2: {some_val} - {some_other_val}"

    async def tool3(some_val: int, some_other_val: str) -> str:
        """Tool 3 docstring."""
        return [
            {"key_1": some_val, "key_2": "foo"},
            {"key_1": some_other_val, "key_2": "baz"},
        ]

    result = ToolNode([tool1]).invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool1",
                            "args": {"some_val": 1, "some_other_val": "foo"},
                            "id": "some 0",
                        }
                    ],
                )
            ]
        }
    )

    tool_message: ToolMessage = result["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.content == "1 - foo"
    assert tool_message.tool_call_id == "some 0"

    result_error = ToolNode([tool1]).invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool1",
                            "args": {"some_val": 0, "some_other_val": "foo"},
                            "id": "some 0",
                        }
                    ],
                )
            ]
        }
    )

    tool_message: ToolMessage = result_error["messages"][-1]
    assert tool_message.type == "tool"
    assert (
        tool_message.content
        == f"Error: {repr(ValueError('Test error'))}\n Please fix your mistakes."
    )
    assert tool_message.tool_call_id == "some 0"

    result2 = await ToolNode([tool2]).ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool2",
                            "args": {"some_val": 2, "some_other_val": "bar"},
                            "id": "some 1",
                        }
                    ],
                )
            ]
        }
    )
    tool_message: ToolMessage = result2["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.content == "tool2: 2 - bar"

    with pytest.raises(ValueError):
        await ToolNode([tool2], handle_tool_errors=False).ainvoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool2",
                                "args": {"some_val": 0, "some_other_val": "bar"},
                                "id": "some 1",
                            }
                        ],
                    )
                ]
            }
        )

    # incorrect tool name
    result_incorrect_name = ToolNode([tool1, tool2]).invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool3",
                            "args": {"some_val": 1, "some_other_val": "foo"},
                            "id": "some 0",
                        }
                    ],
                )
            ]
        }
    )
    tool_message: ToolMessage = result_incorrect_name["messages"][-1]
    assert tool_message.type == "tool"
    assert (
        tool_message.content
        == "Error: tool3 is not a valid tool, try one of [tool1, tool2]."
    )
    assert tool_message.tool_call_id == "some 0"

    # list of dicts tool content
    result3 = await ToolNode([tool3]).ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool3",
                            "args": {"some_val": 2, "some_other_val": "bar"},
                            "id": "some 0",
                        }
                    ],
                )
            ]
        }
    )
    tool_message: ToolMessage = result3["messages"][-1]
    assert tool_message.type == "tool"
    assert (
        tool_message.content
        == '[{"key_1": 2, "key_2": "foo"}, {"key_1": "bar", "key_2": "baz"}]'
    )
    assert tool_message.tool_call_id == "some 0"


def my_function(some_val: int, some_other_val: str) -> str:
    return f"{some_val} - {some_other_val}"


class MyModel(BaseModel):
    some_val: int
    some_other_val: str


class MyModelV1(BaseModelV1):
    some_val: int
    some_other_val: str


@dec_tool
def my_tool(some_val: int, some_other_val: str) -> str:
    """Cool."""
    return f"{some_val} - {some_other_val}"


@pytest.mark.parametrize(
    "tool_schema",
    [
        my_function,
        MyModel,
        MyModelV1,
        my_tool,
    ],
)
@pytest.mark.parametrize("use_message_key", [True, False])
async def test_validation_node(tool_schema: Any, use_message_key: bool):
    validation_node = ValidationNode([tool_schema])
    tool_name = getattr(tool_schema, "name", getattr(tool_schema, "__name__", None))
    inputs = [
        AIMessage(
            "hi?",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": {"some_val": 1, "some_other_val": "foo"},
                    "id": "some 0",
                },
                {
                    "name": tool_name,
                    # Wrong type for some_val
                    "args": {"some_val": "bar", "some_other_val": "foo"},
                    "id": "some 1",
                },
            ],
        ),
    ]
    if use_message_key:
        inputs = {"messages": inputs}
    result = await validation_node.ainvoke(inputs)
    if use_message_key:
        result = result["messages"]

    def check_results(messages: list):
        assert len(messages) == 2
        assert all(m.type == "tool" for m in messages)
        assert not messages[0].additional_kwargs.get("is_error")
        assert messages[1].additional_kwargs.get("is_error")

    check_results(result)
    result_sync = validation_node.invoke(inputs)
    if use_message_key:
        result_sync = result_sync["messages"]
    check_results(result_sync)


def test_tool_node_inject_state() -> None:
    def tool1(some_val: int, state: Annotated[dict, InjectedState]) -> str:
        """Tool 1 docstring."""
        return state["foo"]

    def tool2(some_val: int, state: Annotated[dict, InjectedState()]) -> str:
        """Tool 1 docstring."""
        return state["foo"]

    def tool3(
        some_val: int,
        foo: Annotated[str, InjectedState("foo")],
        msgs: Annotated[List[AnyMessage], InjectedState("messages")],
    ) -> str:
        """Tool 1 docstring."""
        return foo

    def tool4(
        some_val: int, msgs: Annotated[List[AnyMessage], InjectedState("messages")]
    ) -> str:
        """Tool 1 docstring."""
        return msgs[0].content

    node = ToolNode([tool1, tool2, tool3, tool4], handle_tool_errors=False)
    for tool_name in ("tool1", "tool2", "tool3"):
        tool_call = {
            "name": tool_name,
            "args": {"some_val": 1},
            "id": "some 0",
            "type": "tool_call",
        }
        msg = AIMessage("hi?", tool_calls=[tool_call])
        result = node.invoke({"messages": [msg], "foo": "bar"})
        tool_message = result["messages"][-1]
        assert tool_message.content == "bar"

        if tool_name == "tool3":
            with pytest.raises(KeyError):
                node.invoke({"messages": [msg], "notfoo": "bar"})

            with pytest.raises(ValueError):
                node.invoke([msg])
        else:
            tool_message = node.invoke({"messages": [msg], "notfoo": "bar"})[
                "messages"
            ][-1]
            assert "KeyError" in tool_message.content
            tool_message = node.invoke([msg])[-1]
            assert "KeyError" in tool_message.content

    tool_call = {
        "name": "tool4",
        "args": {"some_val": 1},
        "id": "some 0",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])
    result = node.invoke({"messages": [msg]})
    tool_message = result["messages"][-1]
    assert tool_message.content == "hi?"

    result = node.invoke([msg])
    tool_message = result[-1]
    assert tool_message.content == "hi?"


def test_tool_node_ensure_utf8() -> None:
    @dec_tool
    def get_day_list(days: list[str]) -> list[str]:
        """choose days"""
        return days

    data = ["星期一", "水曜日", "목요일", "Friday"]
    tools = [get_day_list]
    tool_calls = [ToolCall(name=get_day_list.name, args={"days": data}, id="test_id")]
    outputs: list[ToolMessage] = ToolNode(tools).invoke(
        [AIMessage(content="", tool_calls=tool_calls)]
    )
    assert outputs[0].content == json.dumps(data, ensure_ascii=False)
