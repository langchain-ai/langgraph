import dataclasses
import inspect
import json
from functools import partial
from typing import (
    Annotated,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import InjectedToolCallId, ToolException
from langchain_core.tools import tool as dec_tool
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langgraph.prebuilt import (
    ToolNode,
    create_react_agent,
    tools_condition,
)
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    AgentStatePydantic,
    StateSchemaType,
    _get_model,
    _should_bind_tools,
    _validate_chat_history,
)
from langgraph.prebuilt.tool_node import (
    InjectedState,
    InjectedStore,
    _get_state_args,
    _infer_handled_types,
)
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Interrupt, interrupt
from langgraph.utils.config import get_stream_writer
from tests.any_str import AnyStr
from tests.conftest import (
    ALL_CHECKPOINTERS_ASYNC,
    ALL_CHECKPOINTERS_SYNC,
    IS_LANGCHAIN_CORE_030_OR_GREATER,
    awith_checkpointer,
)
from tests.messages import _AnyIdHumanMessage, _AnyIdToolMessage
from tests.model import FakeToolCallingModel

pytestmark = pytest.mark.anyio

REACT_TOOL_CALL_VERSIONS = ["v1", "v2"]


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
        create_react_agent(model, [], state_modifier="Foo", prompt="Bar")


def test_system_message_prompt():
    prompt = SystemMessage(content="Foo")
    for agent in (
        create_react_agent(FakeToolCallingModel(), [], prompt=prompt),
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
        create_react_agent(FakeToolCallingModel(), [], state_modifier=prompt),
    ):
        inputs = [HumanMessage("hi?")]
        response = agent.invoke({"messages": inputs})
        expected_response = {
            "messages": inputs + [AIMessage(content="Foo-hi?", id="0", tool_calls=[])]
        }
        assert response == expected_response


def test_callable_prompt():
    def prompt(state):
        modified_message = f"Bar {state['messages'][-1].content}"
        return [HumanMessage(content=modified_message)]

    for agent in (
        create_react_agent(FakeToolCallingModel(), [], prompt=prompt),
        create_react_agent(FakeToolCallingModel(), [], state_modifier=prompt),
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
    prompt = RunnableLambda(
        lambda state: [HumanMessage(content=f"Baz {state['messages'][-1].content}")]
    )

    for agent in (
        create_react_agent(FakeToolCallingModel(), [], prompt=prompt),
        create_react_agent(FakeToolCallingModel(), [], state_modifier=prompt),
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


class CustomState(AgentState):
    user_name: str


class CustomStatePydantic(AgentStatePydantic):
    user_name: Optional[str] = None


@pytest.mark.skipif(
    not IS_LANGCHAIN_CORE_030_OR_GREATER,
    reason="Langchain core 0.3.0 or greater is required",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
@pytest.mark.parametrize("state_schema", [CustomState, CustomStatePydantic])
def test_react_agent_update_state(
    request: pytest.FixtureRequest,
    checkpointer_name: str,
    version: str,
    state_schema: StateSchemaType,
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        "checkpointer_" + checkpointer_name
    )

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

    if issubclass(state_schema, AgentStatePydantic):

        def prompt(state: CustomStatePydantic):
            user_name = state.user_name
            if user_name is None:
                return state.messages

            system_msg = f"User name is {user_name}"
            return [{"role": "system", "content": system_msg}] + state.messages
    else:

        def prompt(state: CustomState):
            user_name = state.get("user_name")
            if user_name is None:
                return state["messages"]

            system_msg = f"User name is {user_name}"
            return [{"role": "system", "content": system_msg}] + state["messages"]

    tool_calls = [[{"args": {}, "id": "1", "name": "get_user_name"}]]
    model = FakeToolCallingModel(tool_calls=tool_calls)
    agent = create_react_agent(
        model,
        [get_user_name],
        state_schema=state_schema,
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
@pytest.mark.parametrize(
    "checkpointer_name",
    [
        checkpointer
        for checkpointer in ALL_CHECKPOINTERS_SYNC
        if "shallow" not in checkpointer
    ],
)
@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_react_agent_parallel_tool_calls(
    request: pytest.FixtureRequest, checkpointer_name: str, version: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        "checkpointer_" + checkpointer_name
    )
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


class AgentStateExtraKey(AgentState):
    foo: int


class AgentStateExtraKeyPydantic(AgentStatePydantic):
    foo: int


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
@pytest.mark.parametrize(
    "state_schema", [AgentStateExtraKey, AgentStateExtraKeyPydantic]
)
def test_create_react_agent_inject_vars(
    version: str, state_schema: StateSchemaType
) -> None:
    store = InMemoryStore()
    namespace = ("test",)
    store.put(namespace, "test_key", {"bar": 3})

    if issubclass(state_schema, AgentStatePydantic):

        def tool1(
            some_val: int,
            state: Annotated[AgentStateExtraKeyPydantic, InjectedState],
            store: Annotated[BaseStore, InjectedStore()],
        ) -> str:
            """Tool 1 docstring."""
            store_val = store.get(namespace, "test_key").value["bar"]
            return some_val + state.foo + store_val
    else:

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
        state_schema=state_schema,
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
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_react_with_subgraph_tools(
    request: pytest.FixtureRequest, checkpointer_name: str, version: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        "checkpointer_" + checkpointer_name
    )

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


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_tool_node_node_interrupt(
    request: pytest.FixtureRequest, checkpointer_name: str, version: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        "checkpointer_" + checkpointer_name
    )

    def tool_normal(some_val: int) -> str:
        """Tool docstring."""
        return "normal"

    def tool_interrupt(some_val: int) -> str:
        """Tool docstring."""
        foo = interrupt("provide value for foo")
        return foo

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

    # TODO: figure out why this is not working w/ shallow postgres checkpointer
    if "shallow" in checkpointer_name:
        return

    state = agent.get_state(config)
    assert state.next == ("tools",)
    task = state.tasks[0]
    assert task.name == "tools"
    assert task.interrupts == (
        Interrupt(
            value="provide value for foo",
            when="during",
            resumable=True,
            ns=[AnyStr("tools:")],
        ),
    )


@pytest.mark.parametrize("tool_style", ["openai", "anthropic"])
def test_should_bind_tools(tool_style: str) -> None:
    @dec_tool
    def some_tool(some_val: int) -> str:
        """Tool docstring."""
        return "meow"

    @dec_tool
    def some_other_tool(some_val: int) -> str:
        """Tool docstring."""
        return "meow"

    model = FakeToolCallingModel(tool_style=tool_style)
    # should bind when a regular model
    assert _should_bind_tools(model, [])
    assert _should_bind_tools(model, [some_tool])

    # should bind when a seq
    seq = model | RunnableLambda(lambda message: message)
    assert _should_bind_tools(seq, [])
    assert _should_bind_tools(seq, [some_tool])

    # should not bind when a model with tools
    assert not _should_bind_tools(model.bind_tools([some_tool]), [some_tool])
    # should not bind when a seq with tools
    seq_with_tools = model.bind_tools([some_tool]) | RunnableLambda(
        lambda message: message
    )
    assert not _should_bind_tools(seq_with_tools, [some_tool])

    # should raise on invalid inputs
    with pytest.raises(ValueError):
        _should_bind_tools(model.bind_tools([some_tool]), [])
    with pytest.raises(ValueError):
        _should_bind_tools(model.bind_tools([some_tool]), [some_other_tool])
    with pytest.raises(ValueError):
        _should_bind_tools(model.bind_tools([some_tool]), [some_tool, some_other_tool])


def test_get_model() -> None:
    model = FakeToolCallingModel(tool_calls=[])
    assert _get_model(model) == model

    @dec_tool
    def some_tool(some_val: int) -> str:
        """Tool docstring."""
        return "meow"

    model_with_tools = model.bind_tools([some_tool])
    assert _get_model(model_with_tools) == model

    seq = model | RunnableLambda(lambda message: message)
    assert _get_model(seq) == model

    seq_with_tools = model.bind_tools([some_tool]) | RunnableLambda(
        lambda message: message
    )
    assert _get_model(seq_with_tools) == model

    with pytest.raises(TypeError):
        _get_model(RunnableLambda(lambda message: message))
