import dataclasses
import inspect
import json
from functools import partial
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

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
from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools import tool as dec_tool
from pydantic import BaseModel, Field, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from typing_extensions import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langgraph.prebuilt import (
    ToolNode,
    ValidationNode,
    create_react_agent,
    tools_condition,
)
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    StructuredResponse,
    _validate_chat_history,
)
from langgraph.prebuilt.tool_node import (
    TOOL_CALL_ERROR_TEMPLATE,
    InjectedState,
    InjectedStore,
    _get_state_args,
    _infer_handled_types,
)
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Interrupt, interrupt
from langgraph.utils.config import get_stream_writer
from tests.conftest import (
    ALL_CHECKPOINTERS_ASYNC,
    ALL_CHECKPOINTERS_SYNC,
    IS_LANGCHAIN_CORE_030_OR_GREATER,
    awith_checkpointer,
)
from tests.messages import _AnyIdHumanMessage, _AnyIdToolMessage

pytestmark = pytest.mark.anyio

REACT_TOOL_CALL_VERSIONS = ["v1", "v2"]


class FakeToolCallingModel(BaseChatModel):
    tool_calls: Optional[list[list[ToolCall]]] = None
    structured_response: Optional[StructuredResponse] = None
    index: int = 0
    tool_style: Literal["openai", "anthropic"] = "openai"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_string = "-".join([m.content for m in messages])
        tool_calls = (
            self.tool_calls[self.index % len(self.tool_calls)]
            if self.tool_calls
            else []
        )
        message = AIMessage(
            content=messages_string, id=str(self.index), tool_calls=tool_calls.copy()
        )
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    def with_structured_output(
        self, schema: Type[BaseModel]
    ) -> Runnable[LanguageModelInput, StructuredResponse]:
        if self.structured_response is None:
            raise ValueError("Structured response is not set")

        return RunnableLambda(lambda x: self.structured_response)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if len(tools) == 0:
            raise ValueError("Must provide at least one tool")

        tool_dicts = []
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise TypeError(
                    "Only BaseTool is supported by FakeToolCallingModel.bind_tools"
                )

            # NOTE: this is a simplified tool spec for testing purposes only
            if self.tool_style == "openai":
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                        },
                    }
                )
            elif self.tool_style == "anthropic":
                tool_dicts.append(
                    {
                        "name": tool.name,
                    }
                )

        return self.bind(tools=tool_dicts)


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_no_prompt(
    request: pytest.FixtureRequest, checkpointer_name: str, version: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        "checkpointer_" + checkpointer_name
    )
    model = FakeToolCallingModel()

    agent = create_react_agent(
        model,
        [],
        checkpointer=checkpointer,
        version=version,
    )
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
            "thread_id": "123",
        }
        assert saved.pending_writes == []


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_no_prompt_async(checkpointer_name: str) -> None:
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
                "thread_id": "123",
            }
            assert saved.pending_writes == []


def test_passing_two_modifiers():
    model = FakeToolCallingModel()
    with pytest.raises(ValueError):
        create_react_agent(model, [], messages_modifier="Foo", state_modifier="Bar")

    with pytest.raises(ValueError):
        create_react_agent(model, [], messages_modifier="Foo", prompt="Bar")


def test_system_message_prompt():
    prompt = SystemMessage(content="Foo")
    for agent in (
        create_react_agent(FakeToolCallingModel(), [], prompt=prompt),
        create_react_agent(FakeToolCallingModel(), [], messages_modifier=prompt),
        create_react_agent(FakeToolCallingModel(), [], state_modifier=prompt),
    ):
        inputs = [HumanMessage("hi?")]
        response = agent.invoke({"messages": inputs})
        expected_response = {
            "messages": inputs + [AIMessage(content="Foo-hi?", id="0", tool_calls=[])]
        }
        assert response == expected_response


def test_string_prompt():
    prompt = "Foo"
    for agent in (
        create_react_agent(FakeToolCallingModel(), [], prompt=prompt),
        create_react_agent(FakeToolCallingModel(), [], messages_modifier=prompt),
        create_react_agent(FakeToolCallingModel(), [], state_modifier=prompt),
    ):
        inputs = [HumanMessage("hi?")]
        response = agent.invoke({"messages": inputs})
        expected_response = {
            "messages": inputs + [AIMessage(content="Foo-hi?", id="0", tool_calls=[])]
        }
        assert response == expected_response


def test_callable_prompt():
    def messages_modifier(messages):
        modified_message = f"Bar {messages[-1].content}"
        return [HumanMessage(content=modified_message)]

    def prompt(state):
        modified_message = f"Bar {state['messages'][-1].content}"
        return [HumanMessage(content=modified_message)]

    for agent in (
        create_react_agent(FakeToolCallingModel(), [], prompt=prompt),
        create_react_agent(FakeToolCallingModel(), [], state_modifier=prompt),
        create_react_agent(
            FakeToolCallingModel(), [], messages_modifier=messages_modifier
        ),
    ):
        inputs = [HumanMessage("hi?")]
        response = agent.invoke({"messages": inputs})
        expected_response = {
            "messages": inputs + [AIMessage(content="Bar hi?", id="0")]
        }
        assert response == expected_response


async def test_callable_prompt_async():
    async def prompt(state):
        modified_message = f"Bar {state['messages'][-1].content}"
        return [HumanMessage(content=modified_message)]

    for agent in (
        create_react_agent(FakeToolCallingModel(), [], prompt=prompt),
        create_react_agent(FakeToolCallingModel(), [], state_modifier=prompt),
    ):
        inputs = [HumanMessage("hi?")]
        response = await agent.ainvoke({"messages": inputs})
        expected_response = {
            "messages": inputs + [AIMessage(content="Bar hi?", id="0")]
        }
        assert response == expected_response


def test_runnable_prompt():
    messages_modifier = RunnableLambda(
        lambda messages: [HumanMessage(content=f"Baz {messages[-1].content}")]
    )
    prompt = RunnableLambda(
        lambda state: [HumanMessage(content=f"Baz {state['messages'][-1].content}")]
    )

    for agent in (
        create_react_agent(FakeToolCallingModel(), [], prompt=prompt),
        create_react_agent(FakeToolCallingModel(), [], state_modifier=prompt),
        create_react_agent(
            FakeToolCallingModel(), [], messages_modifier=messages_modifier
        ),
    ):
        inputs = [HumanMessage("hi?")]
        response = agent.invoke({"messages": inputs})
        expected_response = {
            "messages": inputs + [AIMessage(content="Baz hi?", id="0")]
        }
        assert response == expected_response


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_prompt_with_store(version: str):
    def add(a: int, b: int):
        """Adds a and b"""
        return a + b

    in_memory_store = InMemoryStore()
    in_memory_store.put(("memories", "1"), "user_name", {"data": "User name is Alice"})
    in_memory_store.put(("memories", "2"), "user_name", {"data": "User name is Bob"})

    def prompt(state, config, *, store):
        user_id = config["configurable"]["user_id"]
        system_str = store.get(("memories", user_id), "user_name").value["data"]
        return [SystemMessage(system_str)] + state["messages"]

    def prompt_no_store(state, config):
        return SystemMessage("foo") + state["messages"]

    model = FakeToolCallingModel()

    # test state modifier that uses store works
    agent = create_react_agent(
        model,
        [add],
        prompt=prompt,
        store=in_memory_store,
        version=version,
    )
    response = agent.invoke(
        {"messages": [("user", "hi")]}, {"configurable": {"user_id": "1"}}
    )
    assert response["messages"][-1].content == "User name is Alice-hi"

    # test state modifier that doesn't use store works
    agent = create_react_agent(
        model,
        [add],
        prompt=prompt_no_store,
        store=in_memory_store,
        version=version,
    )
    response = agent.invoke(
        {"messages": [("user", "hi")]}, {"configurable": {"user_id": "2"}}
    )
    assert response["messages"][-1].content == "foo-hi"


async def test_prompt_with_store_async():
    async def add(a: int, b: int):
        """Adds a and b"""
        return a + b

    in_memory_store = InMemoryStore()
    await in_memory_store.aput(
        ("memories", "1"), "user_name", {"data": "User name is Alice"}
    )
    await in_memory_store.aput(
        ("memories", "2"), "user_name", {"data": "User name is Bob"}
    )

    async def prompt(state, config, *, store):
        user_id = config["configurable"]["user_id"]
        system_str = (await store.aget(("memories", user_id), "user_name")).value[
            "data"
        ]
        return [SystemMessage(system_str)] + state["messages"]

    async def prompt_no_store(state, config):
        return SystemMessage("foo") + state["messages"]

    model = FakeToolCallingModel()

    # test state modifier that uses store works
    agent = create_react_agent(model, [add], prompt=prompt, store=in_memory_store)
    response = await agent.ainvoke(
        {"messages": [("user", "hi")]}, {"configurable": {"user_id": "1"}}
    )
    assert response["messages"][-1].content == "User name is Alice-hi"

    # test state modifier that doesn't use store works
    agent = create_react_agent(
        model, [add], prompt=prompt_no_store, store=in_memory_store
    )
    response = await agent.ainvoke(
        {"messages": [("user", "hi")]}, {"configurable": {"user_id": "2"}}
    )
    assert response["messages"][-1].content == "foo-hi"


@pytest.mark.parametrize("tool_style", ["openai", "anthropic"])
@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_model_with_tools(tool_style: str, version: str):
    model = FakeToolCallingModel(tool_style=tool_style)

    @dec_tool
    def tool1(some_val: int) -> str:
        """Tool 1 docstring."""
        return f"Tool 1: {some_val}"

    @dec_tool
    def tool2(some_val: int) -> str:
        """Tool 2 docstring."""
        return f"Tool 2: {some_val}"

    # check valid agent constructor
    agent = create_react_agent(
        model.bind_tools([tool1, tool2]),
        [tool1, tool2],
        version=version,
    )
    result = agent.nodes["tools"].invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool1",
                            "args": {"some_val": 2},
                            "id": "some 1",
                        },
                        {
                            "name": "tool2",
                            "args": {"some_val": 2},
                            "id": "some 2",
                        },
                    ],
                )
            ]
        }
    )
    tool_messages: ToolMessage = result["messages"][-2:]
    for tool_message in tool_messages:
        assert tool_message.type == "tool"
        assert tool_message.content in {"Tool 1: 2", "Tool 2: 2"}
        assert tool_message.tool_call_id in {"some 1", "some 2"}

    # test mismatching tool lengths
    with pytest.raises(ValueError):
        create_react_agent(model.bind_tools([tool1]), [tool1, tool2])

    # test missing bound tools
    with pytest.raises(ValueError):
        create_react_agent(model.bind_tools([tool1]), [tool2])


def test__validate_messages():
    # empty input
    _validate_chat_history([])

    # single human message
    _validate_chat_history(
        [
            HumanMessage(content="What's the weather?"),
        ]
    )

    # human + AI
    _validate_chat_history(
        [
            HumanMessage(content="What's the weather?"),
            AIMessage(content="The weather is sunny and 75°F."),
        ]
    )

    # Answered tool calls
    _validate_chat_history(
        [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="Let me check that for you.",
                tool_calls=[{"id": "call1", "name": "get_weather", "args": {}}],
            ),
            ToolMessage(content="Sunny, 75°F", tool_call_id="call1"),
            AIMessage(content="The weather is sunny and 75°F."),
        ]
    )

    # Unanswered tool calls
    with pytest.raises(ValueError):
        _validate_chat_history(
            [
                AIMessage(
                    content="I'll check that for you.",
                    tool_calls=[
                        {"id": "call1", "name": "get_weather", "args": {}},
                        {"id": "call2", "name": "get_time", "args": {}},
                    ],
                )
            ]
        )

    with pytest.raises(ValueError):
        _validate_chat_history(
            [
                HumanMessage(content="What's the weather and time?"),
                AIMessage(
                    content="I'll check that for you.",
                    tool_calls=[
                        {"id": "call1", "name": "get_weather", "args": {}},
                        {"id": "call2", "name": "get_time", "args": {}},
                    ],
                ),
                ToolMessage(content="Sunny, 75°F", tool_call_id="call1"),
                AIMessage(
                    content="The weather is sunny and 75°F. Let me check the time."
                ),
            ]
        )


def test__infer_handled_types() -> None:
    def handle(e):  # type: ignore
        return ""

    def handle2(e: Exception) -> str:
        return ""

    def handle3(e: Union[ValueError, ToolException]) -> str:
        return ""

    class Handler:
        def handle(self, e: ValueError) -> str:
            return ""

    handle4 = Handler().handle

    def handle5(e: Union[Union[TypeError, ValueError], ToolException]):
        return ""

    expected: tuple = (Exception,)
    actual = _infer_handled_types(handle)
    assert expected == actual

    expected = (Exception,)
    actual = _infer_handled_types(handle2)
    assert expected == actual

    expected = (ValueError, ToolException)
    actual = _infer_handled_types(handle3)
    assert expected == actual

    expected = (ValueError,)
    actual = _infer_handled_types(handle4)
    assert expected == actual

    expected = (TypeError, ValueError, ToolException)
    actual = _infer_handled_types(handle5)
    assert expected == actual

    with pytest.raises(ValueError):

        def handler(e: str):
            return ""

        _infer_handled_types(handler)

    with pytest.raises(ValueError):

        def handler(e: list[Exception]):
            return ""

        _infer_handled_types(handler)

    with pytest.raises(ValueError):

        def handler(e: Union[str, int]):
            return ""

        _infer_handled_types(handler)


@pytest.mark.skipif(
    not IS_LANGCHAIN_CORE_030_OR_GREATER,
    reason="Pydantic v1 is required for this test to pass in langchain-core < 0.3",
)
@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_react_agent_with_structured_response(version: str) -> None:
    class WeatherResponse(BaseModel):
        temperature: float = Field(description="The temperature in fahrenheit")

    tool_calls = [[{"args": {}, "id": "1", "name": "get_weather"}], []]

    def get_weather():
        """Get the weather"""
        return "The weather is sunny and 75°F."

    expected_structured_response = WeatherResponse(temperature=75)
    model = FakeToolCallingModel(
        tool_calls=tool_calls, structured_response=expected_structured_response
    )
    for response_format in (WeatherResponse, ("Meow", WeatherResponse)):
        agent = create_react_agent(
            model,
            [get_weather],
            response_format=response_format,
            version=version,
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})
        assert response["structured_response"] == expected_structured_response
        assert len(response["messages"]) == 4
        assert response["messages"][-2].content == "The weather is sunny and 75°F."


# tools for testing ToolNode
def tool1(some_val: int, some_other_val: str) -> str:
    """Tool 1 docstring."""
    if some_val == 0:
        raise ValueError("Test error")
    return f"{some_val} - {some_other_val}"


async def tool2(some_val: int, some_other_val: str) -> str:
    """Tool 2 docstring."""
    if some_val == 0:
        raise ToolException("Test error")
    return f"tool2: {some_val} - {some_other_val}"


async def tool3(some_val: int, some_other_val: str) -> str:
    """Tool 3 docstring."""
    return [
        {"key_1": some_val, "key_2": "foo"},
        {"key_1": some_other_val, "key_2": "baz"},
    ]


async def tool4(some_val: int, some_other_val: str) -> str:
    """Tool 4 docstring."""
    return [
        {"type": "image_url", "image_url": {"url": "abdc"}},
    ]


@dec_tool
def tool5(some_val: int):
    """Tool 5 docstring."""
    raise ToolException("Test error")


tool5.handle_tool_error = "foo"


async def test_tool_node():
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
                            "id": "some 2",
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
    assert tool_message.tool_call_id == "some 2"

    # list of content blocks tool content
    result4 = await ToolNode([tool4]).ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool4",
                            "args": {"some_val": 2, "some_other_val": "bar"},
                            "id": "some 3",
                        }
                    ],
                )
            ]
        }
    )
    tool_message: ToolMessage = result4["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.content == [{"type": "image_url", "image_url": {"url": "abdc"}}]
    assert tool_message.tool_call_id == "some 3"


async def test_tool_node_tool_call_input():
    # Single tool call
    tool_call_1 = {
        "name": "tool1",
        "args": {"some_val": 1, "some_other_val": "foo"},
        "id": "some 0",
        "type": "tool_call",
    }
    result = ToolNode([tool1]).invoke([tool_call_1])
    assert result["messages"] == [
        ToolMessage(content="1 - foo", tool_call_id="some 0", name="tool1"),
    ]

    # Multiple tool calls
    tool_call_2 = {
        "name": "tool1",
        "args": {"some_val": 2, "some_other_val": "bar"},
        "id": "some 1",
        "type": "tool_call",
    }
    result = ToolNode([tool1]).invoke([tool_call_1, tool_call_2])
    assert result["messages"] == [
        ToolMessage(content="1 - foo", tool_call_id="some 0", name="tool1"),
        ToolMessage(content="2 - bar", tool_call_id="some 1", name="tool1"),
    ]

    # Test with unknown tool
    tool_call_3 = tool_call_1.copy()
    tool_call_3["name"] = "tool2"
    result = ToolNode([tool1]).invoke([tool_call_1, tool_call_3])
    assert result["messages"] == [
        ToolMessage(content="1 - foo", tool_call_id="some 0", name="tool1"),
        ToolMessage(
            content="Error: tool2 is not a valid tool, try one of [tool1].",
            name="tool2",
            tool_call_id="some 0",
            status="error",
        ),
    ]


async def test_tool_node_error_handling():
    def handle_all(e: Union[ValueError, ToolException, ValidationError]):
        return TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))

    # test catching all exceptions, via:
    # - handle_tool_errors = True
    # - passing a tuple of all exceptions
    # - passing a callable with all exceptions in the signature
    for handle_tool_errors in (
        True,
        (ValueError, ToolException, ValidationError),
        handle_all,
    ):
        result_error = await ToolNode(
            [tool1, tool2, tool3], handle_tool_errors=handle_tool_errors
        ).ainvoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool1",
                                "args": {"some_val": 0, "some_other_val": "foo"},
                                "id": "some id",
                            },
                            {
                                "name": "tool2",
                                "args": {"some_val": 0, "some_other_val": "bar"},
                                "id": "some other id",
                            },
                            {
                                "name": "tool3",
                                "args": {"some_val": 0},
                                "id": "another id",
                            },
                        ],
                    )
                ]
            }
        )

        assert all(m.type == "tool" for m in result_error["messages"])
        assert all(m.status == "error" for m in result_error["messages"])
        assert (
            result_error["messages"][0].content
            == f"Error: {repr(ValueError('Test error'))}\n Please fix your mistakes."
        )
        assert (
            result_error["messages"][1].content
            == f"Error: {repr(ToolException('Test error'))}\n Please fix your mistakes."
        )
        assert (
            "ValidationError" in result_error["messages"][2].content
            or "validation error" in result_error["messages"][2].content
        )

        assert result_error["messages"][0].tool_call_id == "some id"
        assert result_error["messages"][1].tool_call_id == "some other id"
        assert result_error["messages"][2].tool_call_id == "another id"


async def test_tool_node_error_handling_callable():
    def handle_value_error(e: ValueError):
        return "Value error"

    def handle_tool_exception(e: ToolException):
        return "Tool exception"

    for handle_tool_errors in ("Value error", handle_value_error):
        result_error = await ToolNode(
            [tool1], handle_tool_errors=handle_tool_errors
        ).ainvoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool1",
                                "args": {"some_val": 0, "some_other_val": "foo"},
                                "id": "some id",
                            },
                        ],
                    )
                ]
            }
        )
        tool_message: ToolMessage = result_error["messages"][-1]
        assert tool_message.type == "tool"
        assert tool_message.status == "error"
        assert tool_message.content == "Value error"

    # test raising for an unhandled exception, via:
    # - passing a tuple of all exceptions
    # - passing a callable with all exceptions in the signature
    for handle_tool_errors in ((ValueError,), handle_value_error):
        with pytest.raises(ToolException) as exc_info:
            await ToolNode(
                [tool1, tool2], handle_tool_errors=handle_tool_errors
            ).ainvoke(
                {
                    "messages": [
                        AIMessage(
                            "hi?",
                            tool_calls=[
                                {
                                    "name": "tool1",
                                    "args": {"some_val": 0, "some_other_val": "foo"},
                                    "id": "some id",
                                },
                                {
                                    "name": "tool2",
                                    "args": {"some_val": 0, "some_other_val": "bar"},
                                    "id": "some other id",
                                },
                            ],
                        )
                    ]
                }
            )
        assert str(exc_info.value) == "Test error"

    for handle_tool_errors in ((ToolException,), handle_tool_exception):
        with pytest.raises(ValueError) as exc_info:
            await ToolNode(
                [tool1, tool2], handle_tool_errors=handle_tool_errors
            ).ainvoke(
                {
                    "messages": [
                        AIMessage(
                            "hi?",
                            tool_calls=[
                                {
                                    "name": "tool1",
                                    "args": {"some_val": 0, "some_other_val": "foo"},
                                    "id": "some id",
                                },
                                {
                                    "name": "tool2",
                                    "args": {"some_val": 0, "some_other_val": "bar"},
                                    "id": "some other id",
                                },
                            ],
                        )
                    ]
                }
            )
        assert str(exc_info.value) == "Test error"


async def test_tool_node_handle_tool_errors_false():
    with pytest.raises(ValueError) as exc_info:
        ToolNode([tool1], handle_tool_errors=False).invoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool1",
                                "args": {"some_val": 0, "some_other_val": "foo"},
                                "id": "some id",
                            }
                        ],
                    )
                ]
            }
        )

    assert str(exc_info.value) == "Test error"

    with pytest.raises(ToolException):
        await ToolNode([tool2], handle_tool_errors=False).ainvoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool2",
                                "args": {"some_val": 0, "some_other_val": "bar"},
                                "id": "some id",
                            }
                        ],
                    )
                ]
            }
        )

    assert str(exc_info.value) == "Test error"

    # test validation errors get raised if handle_tool_errors is False
    with pytest.raises((ValidationError, ValidationErrorV1)):
        ToolNode([tool1], handle_tool_errors=False).invoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool1",
                                "args": {"some_val": 0},
                                "id": "some id",
                            }
                        ],
                    )
                ]
            }
        )


def test_tool_node_individual_tool_error_handling():
    # test error handling on individual tools (and that it overrides overall error handling!)
    result_individual_tool_error_handler = ToolNode(
        [tool5], handle_tool_errors="bar"
    ).invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool5",
                            "args": {"some_val": 0},
                            "id": "some 0",
                        }
                    ],
                )
            ]
        }
    )

    tool_message: ToolMessage = result_individual_tool_error_handler["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.status == "error"
    assert tool_message.content == "foo"
    assert tool_message.tool_call_id == "some 0"


def test_tool_node_incorrect_tool_name():
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
    assert tool_message.status == "error"
    assert (
        tool_message.content
        == "Error: tool3 is not a valid tool, try one of [tool1, tool2]."
    )
    assert tool_message.tool_call_id == "some 0"


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_tool_node_node_interrupt(version: str):
    def tool_normal(some_val: int) -> str:
        """Tool docstring."""
        return "normal"

    def tool_interrupt(some_val: int) -> str:
        """Tool docstring."""
        raise NodeInterrupt("foo")

    def handle(e: NodeInterrupt):
        return "handled"

    for handle_tool_errors in (True, (NodeInterrupt,), "handled", handle, False):
        node = ToolNode([tool_interrupt], handle_tool_errors=handle_tool_errors)
        with pytest.raises(NodeInterrupt) as exc_info:
            node.invoke(
                {
                    "messages": [
                        AIMessage(
                            "hi?",
                            tool_calls=[
                                {
                                    "name": "tool_interrupt",
                                    "args": {"some_val": 0},
                                    "id": "some 0",
                                }
                            ],
                        )
                    ]
                }
            )
            assert exc_info.value == "foo"

    # test inside react agent
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="tool_interrupt", args={"some_val": 0}, id="1"),
                ToolCall(name="tool_normal", args={"some_val": 1}, id="2"),
            ],
            [],
        ]
    )
    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "1"}}
    agent = create_react_agent(
        model,
        [tool_interrupt, tool_normal],
        checkpointer=checkpointer,
        version=version,
    )
    result = agent.invoke({"messages": [HumanMessage("hi?")]}, config)
    expected_messages = [
        _AnyIdHumanMessage(content="hi?"),
        AIMessage(
            content="hi?",
            id="0",
            tool_calls=[
                {
                    "name": "tool_interrupt",
                    "args": {"some_val": 0},
                    "id": "1",
                    "type": "tool_call",
                },
                {
                    "name": "tool_normal",
                    "args": {"some_val": 1},
                    "id": "2",
                    "type": "tool_call",
                },
            ],
        ),
        _AnyIdToolMessage(content="normal", name="tool_normal", tool_call_id="2"),
    ]
    if version == "v1":
        # Interrupt blocks second tool result
        assert result["messages"] == expected_messages[:-1]
    elif version == "v2":
        assert result["messages"] == expected_messages

    state = agent.get_state(config)
    assert state.next == ("tools",)
    task = state.tasks[0]
    assert task.name == "tools"
    assert task.interrupts == (Interrupt(value="foo", when="during"),)


@pytest.mark.skipif(
    not IS_LANGCHAIN_CORE_030_OR_GREATER,
    reason="Langchain core 0.3.0 or greater is required",
)
@pytest.mark.parametrize("input_type", ["dict", "tool_calls"])
async def test_tool_node_command(input_type: str):
    from langchain_core.tools.base import InjectedToolCallId

    @dec_tool
    def transfer_to_bob(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Bob"""
        return Command(
            update={
                "messages": [
                    ToolMessage(content="Transferred to Bob", tool_call_id=tool_call_id)
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        )

    @dec_tool
    async def async_transfer_to_bob(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Bob"""
        return Command(
            update={
                "messages": [
                    ToolMessage(content="Transferred to Bob", tool_call_id=tool_call_id)
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        )

    class CustomToolSchema(BaseModel):
        tool_call_id: Annotated[str, InjectedToolCallId]

    class MyCustomTool(BaseTool):
        def _run(*args: Any, **kwargs: Any):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Transferred to Bob",
                            tool_call_id=kwargs["tool_call_id"],
                        )
                    ]
                },
                goto="bob",
                graph=Command.PARENT,
            )

        async def _arun(*args: Any, **kwargs: Any):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Transferred to Bob",
                            tool_call_id=kwargs["tool_call_id"],
                        )
                    ]
                },
                goto="bob",
                graph=Command.PARENT,
            )

    custom_tool = MyCustomTool(
        name="custom_transfer_to_bob",
        description="Transfer to bob",
        args_schema=CustomToolSchema,
    )
    async_custom_tool = MyCustomTool(
        name="async_custom_transfer_to_bob",
        description="Transfer to bob",
        args_schema=CustomToolSchema,
    )

    # test mixing regular tools and tools returning commands
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    tool_calls = [
        {"args": {"a": 1, "b": 2}, "id": "1", "name": "add", "type": "tool_call"},
        {"args": {}, "id": "2", "name": "transfer_to_bob", "type": "tool_call"},
    ]
    if input_type == "dict":
        input_ = {"messages": [AIMessage("", tool_calls=tool_calls)]}
    elif input_type == "tool_calls":
        input_ = tool_calls
    result = ToolNode([add, transfer_to_bob]).invoke(input_)

    assert result == [
        {
            "messages": [
                ToolMessage(
                    content="3",
                    tool_call_id="1",
                    name="add",
                )
            ]
        },
        Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="2",
                        name="transfer_to_bob",
                    )
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        ),
    ]

    # test tools returning commands

    # test sync tools
    for tool in [transfer_to_bob, custom_tool]:
        result = ToolNode([tool]).invoke(
            {
                "messages": [
                    AIMessage(
                        "", tool_calls=[{"args": {}, "id": "1", "name": tool.name}]
                    )
                ]
            }
        )
        assert result == [
            Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Transferred to Bob",
                            tool_call_id="1",
                            name=tool.name,
                        )
                    ]
                },
                goto="bob",
                graph=Command.PARENT,
            )
        ]

    # test async tools
    for tool in [async_transfer_to_bob, async_custom_tool]:
        result = await ToolNode([tool]).ainvoke(
            {
                "messages": [
                    AIMessage(
                        "", tool_calls=[{"args": {}, "id": "1", "name": tool.name}]
                    )
                ]
            }
        )
        assert result == [
            Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Transferred to Bob",
                            tool_call_id="1",
                            name=tool.name,
                        )
                    ]
                },
                goto="bob",
                graph=Command.PARENT,
            )
        ]

    # test multiple commands
    result = ToolNode([transfer_to_bob, custom_tool]).invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[
                        {"args": {}, "id": "1", "name": "transfer_to_bob"},
                        {"args": {}, "id": "2", "name": "custom_transfer_to_bob"},
                    ],
                )
            ]
        }
    )
    assert result == [
        Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="1",
                        name="transfer_to_bob",
                    )
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        ),
        Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="2",
                        name="custom_transfer_to_bob",
                    )
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        ),
    ]

    # test validation (mismatch between input type and command.update type)
    with pytest.raises(ValueError):

        @dec_tool
        def list_update_tool(tool_call_id: Annotated[str, InjectedToolCallId]):
            """My tool"""
            return Command(
                update=[ToolMessage(content="foo", tool_call_id=tool_call_id)]
            )

        ToolNode([list_update_tool]).invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[
                            {"args": {}, "id": "1", "name": "list_update_tool"}
                        ],
                    )
                ]
            }
        )

    # test validation (missing tool message in the update for current graph)
    with pytest.raises(ValueError):

        @dec_tool
        def no_update_tool():
            """My tool"""
            return Command(update={"messages": []})

        ToolNode([no_update_tool]).invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[{"args": {}, "id": "1", "name": "no_update_tool"}],
                    )
                ]
            }
        )

    # test validation (tool message with a wrong tool call ID)
    with pytest.raises(ValueError):

        @dec_tool
        def mismatching_tool_call_id_tool():
            """My tool"""
            return Command(
                update={"messages": [ToolMessage(content="foo", tool_call_id="2")]}
            )

        ToolNode([mismatching_tool_call_id_tool]).invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[
                            {
                                "args": {},
                                "id": "1",
                                "name": "mismatching_tool_call_id_tool",
                            }
                        ],
                    )
                ]
            }
        )

    # test validation (missing tool message in the update for parent graph is OK)
    @dec_tool
    def node_update_parent_tool():
        """No update"""
        return Command(update={"messages": []}, graph=Command.PARENT)

    assert ToolNode([node_update_parent_tool]).invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[
                        {"args": {}, "id": "1", "name": "node_update_parent_tool"}
                    ],
                )
            ]
        }
    ) == [Command(update={"messages": []}, graph=Command.PARENT)]


@pytest.mark.skipif(
    not IS_LANGCHAIN_CORE_030_OR_GREATER,
    reason="Langchain core 0.3.0 or greater is required",
)
async def test_tool_node_command_list_input():
    from langchain_core.tools.base import InjectedToolCallId

    @dec_tool
    def transfer_to_bob(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Bob"""
        return Command(
            update=[
                ToolMessage(content="Transferred to Bob", tool_call_id=tool_call_id)
            ],
            goto="bob",
            graph=Command.PARENT,
        )

    @dec_tool
    async def async_transfer_to_bob(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Bob"""
        return Command(
            update=[
                ToolMessage(content="Transferred to Bob", tool_call_id=tool_call_id)
            ],
            goto="bob",
            graph=Command.PARENT,
        )

    class CustomToolSchema(BaseModel):
        tool_call_id: Annotated[str, InjectedToolCallId]

    class MyCustomTool(BaseTool):
        def _run(*args: Any, **kwargs: Any):
            return Command(
                update=[
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id=kwargs["tool_call_id"],
                    )
                ],
                goto="bob",
                graph=Command.PARENT,
            )

        async def _arun(*args: Any, **kwargs: Any):
            return Command(
                update=[
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id=kwargs["tool_call_id"],
                    )
                ],
                goto="bob",
                graph=Command.PARENT,
            )

    custom_tool = MyCustomTool(
        name="custom_transfer_to_bob",
        description="Transfer to bob",
        args_schema=CustomToolSchema,
    )
    async_custom_tool = MyCustomTool(
        name="async_custom_transfer_to_bob",
        description="Transfer to bob",
        args_schema=CustomToolSchema,
    )

    # test mixing regular tools and tools returning commands
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    result = ToolNode([add, transfer_to_bob]).invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"args": {"a": 1, "b": 2}, "id": "1", "name": "add"},
                    {"args": {}, "id": "2", "name": "transfer_to_bob"},
                ],
            )
        ]
    )

    assert result == [
        [
            ToolMessage(
                content="3",
                tool_call_id="1",
                name="add",
            )
        ],
        Command(
            update=[
                ToolMessage(
                    content="Transferred to Bob",
                    tool_call_id="2",
                    name="transfer_to_bob",
                )
            ],
            goto="bob",
            graph=Command.PARENT,
        ),
    ]

    # test tools returning commands

    # test sync tools
    for tool in [transfer_to_bob, custom_tool]:
        result = ToolNode([tool]).invoke(
            [AIMessage("", tool_calls=[{"args": {}, "id": "1", "name": tool.name}])]
        )
        assert result == [
            Command(
                update=[
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="1",
                        name=tool.name,
                    )
                ],
                goto="bob",
                graph=Command.PARENT,
            )
        ]

    # test async tools
    for tool in [async_transfer_to_bob, async_custom_tool]:
        result = await ToolNode([tool]).ainvoke(
            [AIMessage("", tool_calls=[{"args": {}, "id": "1", "name": tool.name}])]
        )
        assert result == [
            Command(
                update=[
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="1",
                        name=tool.name,
                    )
                ],
                goto="bob",
                graph=Command.PARENT,
            )
        ]

    # test multiple commands
    result = ToolNode([transfer_to_bob, custom_tool]).invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"args": {}, "id": "1", "name": "transfer_to_bob"},
                    {"args": {}, "id": "2", "name": "custom_transfer_to_bob"},
                ],
            )
        ]
    )
    assert result == [
        Command(
            update=[
                ToolMessage(
                    content="Transferred to Bob",
                    tool_call_id="1",
                    name="transfer_to_bob",
                )
            ],
            goto="bob",
            graph=Command.PARENT,
        ),
        Command(
            update=[
                ToolMessage(
                    content="Transferred to Bob",
                    tool_call_id="2",
                    name="custom_transfer_to_bob",
                )
            ],
            goto="bob",
            graph=Command.PARENT,
        ),
    ]

    # test validation (mismatch between input type and command.update type)
    with pytest.raises(ValueError):

        @dec_tool
        def list_update_tool(tool_call_id: Annotated[str, InjectedToolCallId]):
            """My tool"""
            return Command(
                update={
                    "messages": [ToolMessage(content="foo", tool_call_id=tool_call_id)]
                }
            )

        ToolNode([list_update_tool]).invoke(
            [
                AIMessage(
                    "",
                    tool_calls=[{"args": {}, "id": "1", "name": "list_update_tool"}],
                )
            ]
        )

    # test validation (missing tool message in the update for current graph)
    with pytest.raises(ValueError):

        @dec_tool
        def no_update_tool():
            """My tool"""
            return Command(update=[])

        ToolNode([no_update_tool]).invoke(
            [
                AIMessage(
                    "",
                    tool_calls=[{"args": {}, "id": "1", "name": "no_update_tool"}],
                )
            ]
        )

    # test validation (tool message with a wrong tool call ID)
    with pytest.raises(ValueError):

        @dec_tool
        def mismatching_tool_call_id_tool():
            """My tool"""
            return Command(update=[ToolMessage(content="foo", tool_call_id="2")])

        ToolNode([mismatching_tool_call_id_tool]).invoke(
            [
                AIMessage(
                    "",
                    tool_calls=[
                        {"args": {}, "id": "1", "name": "mismatching_tool_call_id_tool"}
                    ],
                )
            ]
        )

    # test validation (missing tool message in the update for parent graph is OK)
    @dec_tool
    def node_update_parent_tool():
        """No update"""
        return Command(update=[], graph=Command.PARENT)

    assert ToolNode([node_update_parent_tool]).invoke(
        [
            AIMessage(
                "",
                tool_calls=[{"args": {}, "id": "1", "name": "node_update_parent_tool"}],
            )
        ]
    ) == [Command(update=[], graph=Command.PARENT)]


@pytest.mark.skipif(
    not IS_LANGCHAIN_CORE_030_OR_GREATER,
    reason="Langchain core 0.3.0 or greater is required",
)
@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_react_agent_update_state(version: str):
    from langchain_core.tools.base import InjectedToolCallId

    class State(AgentState):
        user_name: str

    @dec_tool
    def get_user_name(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Retrieve user name"""
        user_name = interrupt("Please provider user name:")
        return Command(
            update={
                "user_name": user_name,
                "messages": [
                    ToolMessage(
                        "Successfully retrieved user name", tool_call_id=tool_call_id
                    )
                ],
            }
        )

    def prompt(state: State):
        user_name = state.get("user_name")
        if user_name is None:
            return state["messages"]

        system_msg = f"User name is {user_name}"
        return [{"role": "system", "content": system_msg}] + state["messages"]

    checkpointer = MemorySaver()
    tool_calls = [[{"args": {}, "id": "1", "name": "get_user_name"}]]
    model = FakeToolCallingModel(tool_calls=tool_calls)
    agent = create_react_agent(
        model,
        [get_user_name],
        state_schema=State,
        prompt=prompt,
        checkpointer=checkpointer,
        version=version,
    )
    config = {"configurable": {"thread_id": "1"}}
    # run until interrpupted
    agent.invoke({"messages": [("user", "what's my name")]}, config)
    # supply the value for the interrupt
    response = agent.invoke(Command(resume="Archibald"), config)
    # confirm that the state was updated
    assert response["user_name"] == "Archibald"
    assert len(response["messages"]) == 4
    tool_message: ToolMessage = response["messages"][-2]
    assert tool_message.content == "Successfully retrieved user name"
    assert tool_message.tool_call_id == "1"
    assert tool_message.name == "get_user_name"


@pytest.mark.skipif(
    not IS_LANGCHAIN_CORE_030_OR_GREATER,
    reason="Langchain core 0.3.0 or greater is required",
)
@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_react_agent_parallel_tool_calls(version: str):
    human_assistance_execution_count = 0

    @dec_tool
    def human_assistance(query: str) -> str:
        """Request assistance from a human."""
        nonlocal human_assistance_execution_count
        human_response = interrupt({"query": query})
        human_assistance_execution_count += 1
        return human_response["data"]

    get_weather_execution_count = 0

    @dec_tool
    def get_weather(location: str) -> str:
        """Use this tool to get the weather."""
        nonlocal get_weather_execution_count
        get_weather_execution_count += 1
        return "It's sunny!"

    checkpointer = MemorySaver()
    tool_calls = [
        [
            {"args": {"location": "sf"}, "id": "1", "name": "get_weather"},
            {"args": {"query": "request help"}, "id": "2", "name": "human_assistance"},
        ],
        [],
    ]
    model = FakeToolCallingModel(tool_calls=tool_calls)
    agent = create_react_agent(
        model,
        [human_assistance, get_weather],
        checkpointer=checkpointer,
        version=version,
    )
    config = {"configurable": {"thread_id": "1"}}
    query = "Get user assistance and also check the weather"
    message_types = []
    for event in agent.stream(
        {"messages": [("user", query)]}, config, stream_mode="values"
    ):
        message_types.append([message.type for message in event["messages"]])

    if version == "v1":
        assert message_types == [
            ["human"],
            ["human", "ai"],
        ]
    elif version == "v2":
        assert message_types == [
            ["human"],
            ["human", "ai"],
            ["human", "ai", "tool"],
        ]

    # Resume
    message_types = []
    for event in agent.stream(
        Command(resume={"data": "Hello"}), config, stream_mode="values"
    ):
        message_types.append([message.type for message in event["messages"]])

    assert message_types == [
        ["human", "ai"],
        ["human", "ai", "tool", "tool"],
        ["human", "ai", "tool", "tool", "ai"],
    ]

    if version == "v1":
        assert human_assistance_execution_count == 1
        assert get_weather_execution_count == 2
    elif version == "v2":
        assert human_assistance_execution_count == 1
        assert get_weather_execution_count == 1


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


class _InjectStateSchema(TypedDict):
    messages: list
    foo: str


class _InjectedStatePydanticSchema(BaseModelV1):
    messages: list
    foo: str


class _InjectedStatePydanticV2Schema(BaseModel):
    messages: list
    foo: str


@dataclasses.dataclass
class _InjectedStateDataclassSchema:
    messages: list
    foo: str


T = TypeVar("T")


@pytest.mark.parametrize(
    "schema_",
    [
        _InjectStateSchema,
        _InjectedStatePydanticSchema,
        _InjectedStatePydanticV2Schema,
        _InjectedStateDataclassSchema,
    ],
)
def test_tool_node_inject_state(schema_: Type[T]) -> None:
    def tool1(some_val: int, state: Annotated[T, InjectedState]) -> str:
        """Tool 1 docstring."""
        if isinstance(state, dict):
            return state["foo"]
        else:
            return getattr(state, "foo")

    def tool2(some_val: int, state: Annotated[T, InjectedState()]) -> str:
        """Tool 2 docstring."""
        if isinstance(state, dict):
            return state["foo"]
        else:
            return getattr(state, "foo")

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

    node = ToolNode([tool1, tool2, tool3, tool4])
    for tool_name in ("tool1", "tool2", "tool3"):
        tool_call = {
            "name": tool_name,
            "args": {"some_val": 1},
            "id": "some 0",
            "type": "tool_call",
        }
        msg = AIMessage("hi?", tool_calls=[tool_call])
        result = node.invoke(schema_(**{"messages": [msg], "foo": "bar"}))
        tool_message = result["messages"][-1]
        assert tool_message.content == "bar", f"Failed for tool={tool_name}"

        if tool_name == "tool3":
            failure_input = None
            try:
                failure_input = schema_(**{"messages": [msg], "notfoo": "bar"})
            except Exception:
                pass
            if failure_input is not None:
                with pytest.raises(KeyError):
                    node.invoke(failure_input)

                with pytest.raises(ValueError):
                    node.invoke([msg])
        else:
            failure_input = None
            try:
                failure_input = schema_(**{"messages": [msg], "notfoo": "bar"})
            except Exception:
                # We'd get a validation error from pydantic state and wouldn't make it to the node
                # anyway
                pass
            if failure_input is not None:
                messages_ = node.invoke(failure_input)
                tool_message = messages_["messages"][-1]
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
    result = node.invoke(schema_(**{"messages": [msg], "foo": ""}))
    tool_message = result["messages"][-1]
    assert tool_message.content == "hi?"

    result = node.invoke([msg])
    tool_message = result[-1]
    assert tool_message.content == "hi?"


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_create_react_agent_inject_vars(version: str) -> None:
    class AgentStateExtraKey(AgentState):
        foo: int

    store = InMemoryStore()
    namespace = ("test",)
    store.put(namespace, "test_key", {"bar": 3})

    def tool1(
        some_val: int,
        state: Annotated[dict, InjectedState],
        store: Annotated[BaseStore, InjectedStore()],
    ) -> str:
        """Tool 1 docstring."""
        store_val = store.get(namespace, "test_key").value["bar"]
        return some_val + state["foo"] + store_val

    tool_call = {
        "name": "tool1",
        "args": {"some_val": 1},
        "id": "some 0",
        "type": "tool_call",
    }
    model = FakeToolCallingModel(tool_calls=[[tool_call], []])
    agent = create_react_agent(
        model,
        [tool1],
        state_schema=AgentStateExtraKey,
        store=store,
        version=version,
    )
    input_message = HumanMessage("hi")
    result = agent.invoke({"messages": [input_message], "foo": 2})
    assert result["messages"] == [
        input_message,
        AIMessage(content="hi", tool_calls=[tool_call], id="0"),
        _AnyIdToolMessage(content="6", name="tool1", tool_call_id="some 0"),
        AIMessage("hi-hi-6", id="1"),
    ]
    assert result["foo"] == 2


@pytest.mark.skipif(
    not IS_LANGCHAIN_CORE_030_OR_GREATER,
    reason="Langchain core 0.3.0 or greater is required",
)
def test_tool_node_inject_store() -> None:
    store = InMemoryStore()
    namespace = ("test",)

    def tool1(some_val: int, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """Tool 1 docstring."""
        store_val = store.get(namespace, "test_key").value["foo"]
        return f"Some val: {some_val}, store val: {store_val}"

    def tool2(some_val: int, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """Tool 2 docstring."""
        store_val = store.get(namespace, "test_key").value["foo"]
        return f"Some val: {some_val}, store val: {store_val}"

    def tool3(
        some_val: int,
        bar: Annotated[str, InjectedState("bar")],
        store: Annotated[BaseStore, InjectedStore()],
    ) -> str:
        """Tool 3 docstring."""
        store_val = store.get(namespace, "test_key").value["foo"]
        return f"Some val: {some_val}, store val: {store_val}, state val: {bar}"

    node = ToolNode([tool1, tool2, tool3], handle_tool_errors=True)
    store.put(namespace, "test_key", {"foo": "bar"})

    class State(MessagesState):
        bar: str

    builder = StateGraph(State)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    graph = builder.compile(store=store)

    for tool_name in ("tool1", "tool2"):
        tool_call = {
            "name": tool_name,
            "args": {"some_val": 1},
            "id": "some 0",
            "type": "tool_call",
        }
        msg = AIMessage("hi?", tool_calls=[tool_call])
        node_result = node.invoke({"messages": [msg]}, store=store)
        graph_result = graph.invoke({"messages": [msg]})
        for result in (node_result, graph_result):
            result["messages"][-1]
            tool_message = result["messages"][-1]
            assert (
                tool_message.content == "Some val: 1, store val: bar"
            ), f"Failed for tool={tool_name}"

    tool_call = {
        "name": "tool3",
        "args": {"some_val": 1},
        "id": "some 0",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])
    node_result = node.invoke({"messages": [msg], "bar": "baz"}, store=store)
    graph_result = graph.invoke({"messages": [msg], "bar": "baz"})
    for result in (node_result, graph_result):
        result["messages"][-1]
        tool_message = result["messages"][-1]
        assert (
            tool_message.content == "Some val: 1, store val: bar, state val: baz"
        ), f"Failed for tool={tool_name}"

    # test injected store without passing store to compiled graph
    failing_graph = builder.compile()
    with pytest.raises(ValueError):
        failing_graph.invoke({"messages": [msg], "bar": "baz"})


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


def test_tool_node_messages_key() -> None:
    @dec_tool
    def add(a: int, b: int):
        """Adds a and b."""
        return a + b

    model = FakeToolCallingModel(
        tool_calls=[[ToolCall(name=add.name, args={"a": 1, "b": 2}, id="test_id")]]
    )

    class State(TypedDict):
        subgraph_messages: Annotated[list[AnyMessage], add_messages]

    def call_model(state: State):
        response = model.invoke(state["subgraph_messages"])
        model.tool_calls = []
        return {"subgraph_messages": response}

    builder = StateGraph(State)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode([add], messages_key="subgraph_messages"))
    builder.add_conditional_edges(
        "agent", partial(tools_condition, messages_key="subgraph_messages")
    )
    builder.add_edge(START, "agent")
    builder.add_edge("tools", "agent")

    graph = builder.compile()
    result = graph.invoke({"subgraph_messages": [HumanMessage(content="hi")]})
    assert result["subgraph_messages"] == [
        _AnyIdHumanMessage(content="hi"),
        AIMessage(
            content="hi",
            id="0",
            tool_calls=[ToolCall(name=add.name, args={"a": 1, "b": 2}, id="test_id")],
        ),
        _AnyIdToolMessage(content="3", name=add.name, tool_call_id="test_id"),
        AIMessage(content="hi-hi-3", id="1"),
    ]


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
async def test_return_direct(version: str) -> None:
    @dec_tool(return_direct=True)
    def tool_return_direct(input: str) -> str:
        """A tool that returns directly."""
        return f"Direct result: {input}"

    @dec_tool
    def tool_normal(input: str) -> str:
        """A normal tool."""
        return f"Normal result: {input}"

    first_tool_call = [
        ToolCall(
            name="tool_return_direct",
            args={"input": "Test direct"},
            id="1",
        ),
    ]
    expected_ai = AIMessage(
        content="Test direct",
        id="0",
        tool_calls=first_tool_call,
    )
    model = FakeToolCallingModel(tool_calls=[first_tool_call, []])
    agent = create_react_agent(
        model,
        [tool_return_direct, tool_normal],
        version=version,
    )

    # Test direct return for tool_return_direct
    result = agent.invoke(
        {"messages": [HumanMessage(content="Test direct", id="hum0")]}
    )
    assert result["messages"] == [
        HumanMessage(content="Test direct", id="hum0"),
        expected_ai,
        ToolMessage(
            content="Direct result: Test direct",
            name="tool_return_direct",
            tool_call_id="1",
            id=result["messages"][2].id,
        ),
    ]
    second_tool_call = [
        ToolCall(
            name="tool_normal",
            args={"input": "Test normal"},
            id="2",
        ),
    ]
    model = FakeToolCallingModel(tool_calls=[second_tool_call, []])
    agent = create_react_agent(model, [tool_return_direct, tool_normal])
    result = agent.invoke(
        {"messages": [HumanMessage(content="Test normal", id="hum1")]}
    )
    assert result["messages"] == [
        HumanMessage(content="Test normal", id="hum1"),
        AIMessage(content="Test normal", id="0", tool_calls=second_tool_call),
        ToolMessage(
            content="Normal result: Test normal",
            name="tool_normal",
            tool_call_id="2",
            id=result["messages"][2].id,
        ),
        AIMessage(content="Test normal-Test normal-Normal result: Test normal", id="1"),
    ]

    both_tool_calls = [
        ToolCall(
            name="tool_return_direct",
            args={"input": "Test both direct"},
            id="3",
        ),
        ToolCall(
            name="tool_normal",
            args={"input": "Test both normal"},
            id="4",
        ),
    ]
    model = FakeToolCallingModel(tool_calls=[both_tool_calls, []])
    agent = create_react_agent(model, [tool_return_direct, tool_normal])
    result = agent.invoke({"messages": [HumanMessage(content="Test both", id="hum2")]})
    assert result["messages"] == [
        HumanMessage(content="Test both", id="hum2"),
        AIMessage(content="Test both", id="0", tool_calls=both_tool_calls),
        ToolMessage(
            content="Direct result: Test both direct",
            name="tool_return_direct",
            tool_call_id="3",
            id=result["messages"][2].id,
        ),
        ToolMessage(
            content="Normal result: Test both normal",
            name="tool_normal",
            tool_call_id="4",
            id=result["messages"][3].id,
        ),
    ]


def test__get_state_args() -> None:
    class Schema1(BaseModel):
        a: Annotated[str, InjectedState]

    class Schema2(Schema1):
        b: Annotated[int, InjectedState("bar")]

    @dec_tool(args_schema=Schema2)
    def foo(a: str, b: int) -> float:
        """return"""
        return 0.0

    assert _get_state_args(foo) == {"a": None, "b": "bar"}


def test_inspect_react() -> None:
    model = FakeToolCallingModel(tool_calls=[])
    agent = create_react_agent(model, [])
    inspect.getclosurevars(agent.nodes["agent"].bound.func)


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_react_with_subgraph_tools(version: str) -> None:
    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define the subgraphs
    def add(state):
        return {"result": state["a"] + state["b"]}

    add_subgraph = (
        StateGraph(State, output=Output).add_node(add).add_edge(START, "add").compile()
    )

    def multiply(state):
        return {"result": state["a"] * state["b"]}

    multiply_subgraph = (
        StateGraph(State, output=Output)
        .add_node(multiply)
        .add_edge(START, "multiply")
        .compile()
    )

    multiply_subgraph.invoke({"a": 2, "b": 3})

    # Add subgraphs as tools

    def addition(a: int, b: int):
        """Add two numbers"""
        return add_subgraph.invoke({"a": a, "b": b})["result"]

    def multiplication(a: int, b: int):
        """Multiply two numbers"""
        return multiply_subgraph.invoke({"a": a, "b": b})["result"]

    model = FakeToolCallingModel(
        tool_calls=[
            [
                {"args": {"a": 2, "b": 3}, "id": "1", "name": "addition"},
                {"args": {"a": 2, "b": 3}, "id": "2", "name": "multiplication"},
            ],
            [],
        ]
    )
    checkpointer = MemorySaver()
    tool_node = ToolNode([addition, multiplication], handle_tool_errors=False)
    agent = create_react_agent(
        model,
        tool_node,
        checkpointer=checkpointer,
        version=version,
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="What's 2 + 3 and 2 * 3?")]},
        config={"configurable": {"thread_id": "1"}},
    )
    assert result["messages"] == [
        _AnyIdHumanMessage(content="What's 2 + 3 and 2 * 3?"),
        AIMessage(
            content="What's 2 + 3 and 2 * 3?",
            id="0",
            tool_calls=[
                ToolCall(name="addition", args={"a": 2, "b": 3}, id="1"),
                ToolCall(name="multiplication", args={"a": 2, "b": 3}, id="2"),
            ],
        ),
        ToolMessage(
            content="5", name="addition", tool_call_id="1", id=result["messages"][2].id
        ),
        ToolMessage(
            content="6",
            name="multiplication",
            tool_call_id="2",
            id=result["messages"][3].id,
        ),
        AIMessage(
            content="What's 2 + 3 and 2 * 3?-What's 2 + 3 and 2 * 3?-5-6", id="1"
        ),
    ]


def test_tool_node_stream_writer() -> None:
    @dec_tool
    def streaming_tool(x: int) -> str:
        """Do something with writer."""
        my_writer = get_stream_writer()
        for value in ["foo", "bar", "baz"]:
            my_writer({"custom_tool_value": value})

        return x

    tool_node = ToolNode([streaming_tool])
    graph = (
        StateGraph(MessagesState)
        .add_node("tools", tool_node)
        .add_edge(START, "tools")
        .compile()
    )

    tool_call = {
        "name": "streaming_tool",
        "args": {"x": 1},
        "id": "1",
        "type": "tool_call",
    }
    inputs = {
        "messages": [AIMessage("", tool_calls=[tool_call])],
    }

    assert list(graph.stream(inputs, stream_mode="custom")) == [
        {"custom_tool_value": "foo"},
        {"custom_tool_value": "bar"},
        {"custom_tool_value": "baz"},
    ]
    assert list(graph.stream(inputs, stream_mode=["custom", "updates"])) == [
        ("custom", {"custom_tool_value": "foo"}),
        ("custom", {"custom_tool_value": "bar"}),
        ("custom", {"custom_tool_value": "baz"}),
        (
            "updates",
            {
                "tools": {
                    "messages": [
                        _AnyIdToolMessage(
                            content="1",
                            name="streaming_tool",
                            tool_call_id="1",
                        ),
                    ],
                },
            },
        ),
    ]
