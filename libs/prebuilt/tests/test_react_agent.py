import dataclasses
import inspect
import json
import sys
from functools import partial
from typing import (
    Annotated,
    Literal,
    TypeVar,
)
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import InjectedToolCallId, ToolException
from langchain_core.tools import tool as dec_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Interrupt, interrupt
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import TypedDict

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
from tests.any_str import AnyStr
from tests.messages import _AnyIdHumanMessage, _AnyIdToolMessage
from tests.model import FakeToolCallingModel

pytestmark = pytest.mark.anyio

REACT_TOOL_CALL_VERSIONS = ["v1", "v2"]


def _create_mock_runtime(store: BaseStore | None = None) -> Mock:
    """Create a mock Runtime object for testing ToolNode outside of graph context.

    This helper is needed because ToolNode._func expects a Runtime parameter
    which is injected by RunnableCallable from config["configurable"]["__pregel_runtime"].
    When testing ToolNode directly (outside a graph), we need to provide this manually.
    """
    mock_runtime = Mock()
    mock_runtime.store = store
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda *args, **kwargs: None
    return mock_runtime


def _create_config_with_runtime(store: BaseStore | None = None) -> RunnableConfig:
    """Create a RunnableConfig with mock Runtime for testing ToolNode.

    Returns:
        RunnableConfig with __pregel_runtime in configurable dict.
    """
    return {"configurable": {"__pregel_runtime": _create_mock_runtime(store)}}


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_no_prompt(sync_checkpointer: BaseCheckpointSaver, version: str) -> None:
    model = FakeToolCallingModel()

    agent = create_react_agent(
        model,
        [],
        checkpointer=sync_checkpointer,
        version=version,
    )
    inputs = [HumanMessage("hi?")]
    thread = {"configurable": {"thread_id": "123"}}
    response = agent.invoke({"messages": inputs}, thread, debug=True)
    expected_response = {"messages": inputs + [AIMessage(content="hi?", id="0")]}
    assert response == expected_response

    saved = sync_checkpointer.get_tuple(thread)
    assert saved is not None
    assert saved.checkpoint["channel_values"] == {
        "messages": [
            _AnyIdHumanMessage(content="hi?"),
            AIMessage(content="hi?", id="0"),
        ],
    }
    assert saved.metadata == {
        "parents": {},
        "source": "loop",
        "step": 1,
    }
    assert saved.pending_writes == []


async def test_no_prompt_async(async_checkpointer: BaseCheckpointSaver) -> None:
    model = FakeToolCallingModel()

    agent = create_react_agent(model, [], checkpointer=async_checkpointer)
    inputs = [HumanMessage("hi?")]
    thread = {"configurable": {"thread_id": "123"}}
    response = await agent.ainvoke({"messages": inputs}, thread, debug=True)
    expected_response = {"messages": inputs + [AIMessage(content="hi?", id="0")]}
    assert response == expected_response

    saved = await async_checkpointer.aget_tuple(thread)
    assert saved is not None
    assert saved.checkpoint["channel_values"] == {
        "messages": [
            _AnyIdHumanMessage(content="hi?"),
            AIMessage(content="hi?", id="0"),
        ],
    }
    assert saved.metadata == {
        "parents": {},
        "source": "loop",
        "step": 1,
    }
    assert saved.pending_writes == []


def test_system_message_prompt():
    prompt = SystemMessage(content="Foo")
    agent = create_react_agent(FakeToolCallingModel(), [], prompt=prompt)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {
        "messages": inputs + [AIMessage(content="Foo-hi?", id="0", tool_calls=[])]
    }
    assert response == expected_response


def test_string_prompt():
    prompt = "Foo"
    agent = create_react_agent(FakeToolCallingModel(), [], prompt=prompt)
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

    agent = create_react_agent(FakeToolCallingModel(), [], prompt=prompt)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Bar hi?", id="0")]}
    assert response == expected_response


async def test_callable_prompt_async():
    async def prompt(state):
        modified_message = f"Bar {state['messages'][-1].content}"
        return [HumanMessage(content=modified_message)]

    agent = create_react_agent(FakeToolCallingModel(), [], prompt=prompt)
    inputs = [HumanMessage("hi?")]
    response = await agent.ainvoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Bar hi?", id="0")]}
    assert response == expected_response


def test_runnable_prompt():
    prompt = RunnableLambda(
        lambda state: [HumanMessage(content=f"Baz {state['messages'][-1].content}")]
    )

    agent = create_react_agent(FakeToolCallingModel(), [], prompt=prompt)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Baz hi?", id="0")]}
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
@pytest.mark.parametrize("include_builtin", [True, False])
def test_model_with_tools(tool_style: str, version: str, include_builtin: bool):
    model = FakeToolCallingModel(tool_style=tool_style)

    @dec_tool
    def tool1(some_val: int) -> str:
        """Tool 1 docstring."""
        return f"Tool 1: {some_val}"

    @dec_tool
    def tool2(some_val: int) -> str:
        """Tool 2 docstring."""
        return f"Tool 2: {some_val}"

    tools = [tool1, tool2]
    if include_builtin:
        tools.append(
            {
                "type": "mcp",
                "server_label": "atest_sever",
                "server_url": "https://some.mcp.somewhere.com/sse",
                "headers": {"foo": "bar"},
                "allowed_tools": [
                    "mcp_tool_1",
                    "set_active_account",
                    "get_url_markdown",
                    "get_url_screenshot",
                ],
                "require_approval": "never",
            }
        )
    # check valid agent constructor
    agent = create_react_agent(
        model.bind_tools(tools),
        tools,
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
        },
        config=_create_config_with_runtime(),
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

    def handle3(e: ValueError | ToolException) -> str:
        return ""

    class Handler:
        def handle(self, e: ValueError) -> str:
            return ""

    handle4 = Handler().handle

    def handle5(e: TypeError | ValueError | ToolException):
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

        def handler(e: str | int):
            return ""

        _infer_handled_types(handler)


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
    user_name: str | None = None


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
@pytest.mark.parametrize("state_schema", [CustomState, CustomStatePydantic])
def test_react_agent_update_state(
    sync_checkpointer: BaseCheckpointSaver,
    version: Literal["v1", "v2"],
    state_schema: StateSchemaType,
) -> None:
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
        checkpointer=sync_checkpointer,
        version=version,
    )
    config = {"configurable": {"thread_id": "1"}}
    # Run until interrupted
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


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_react_agent_parallel_tool_calls(
    sync_checkpointer: BaseCheckpointSaver, version: str
) -> None:
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
        checkpointer=sync_checkpointer,
        version=version,
    )
    config = {"configurable": {"thread_id": "1"}}
    query = "Get user assistance and also check the weather"
    message_types = []
    for event in agent.stream(
        {"messages": [("user", query)]}, config, stream_mode="values"
    ):
        if messages := event.get("messages"):
            message_types.append([m.type for m in messages])

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
        if messages := event.get("messages"):
            message_types.append([m.type for m in messages])

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
        pytest.param(
            _InjectedStatePydanticSchema,
            marks=pytest.mark.skipif(
                sys.version_info >= (3, 14),
                reason="Pydantic v1 not supported in Python 3.14+",
            ),
        ),
        _InjectedStatePydanticV2Schema,
        _InjectedStateDataclassSchema,
    ],
)
def test_tool_node_inject_state(schema_: type[T]) -> None:
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
        msgs: Annotated[list[AnyMessage], InjectedState("messages")],
    ) -> str:
        """Tool 1 docstring."""
        return foo

    def tool4(
        some_val: int, msgs: Annotated[list[AnyMessage], InjectedState("messages")]
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
        result = node.invoke(
            schema_(**{"messages": [msg], "foo": "bar"}),
            config=_create_config_with_runtime(),
        )
        tool_message = result["messages"][-1]
        assert tool_message.content == "bar", f"Failed for tool={tool_name}"

    tool_call = {
        "name": "tool4",
        "args": {"some_val": 1},
        "id": "some 0",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])
    result = node.invoke(
        schema_(**{"messages": [msg], "foo": ""}), config=_create_config_with_runtime()
    )
    tool_message = result["messages"][-1]
    assert tool_message.content == "hi?"

    result = node.invoke([msg], config=_create_config_with_runtime())
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
    version: Literal["v1", "v2"], state_schema: StateSchemaType
) -> None:
    """Test that the agent can inject state and store into tool functions."""
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
        ToolNode([tool1], handle_tool_errors=False),
        state_schema=state_schema,
        store=store,
        version=version,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "hi"}], "foo": 2})
    assert result["messages"] == [
        _AnyIdHumanMessage(content="hi"),
        AIMessage(content="hi", tool_calls=[tool_call], id="0"),
        _AnyIdToolMessage(content="6", name="tool1", tool_call_id="some 0"),
        AIMessage("hi-hi-6", id="1"),
    ]
    assert result["foo"] == 2


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
        node_result = node.invoke(
            {"messages": [msg]}, config=_create_config_with_runtime(store=store)
        )
        graph_result = graph.invoke({"messages": [msg]})
        for result in (node_result, graph_result):
            result["messages"][-1]
            tool_message = result["messages"][-1]
            assert tool_message.content == "Some val: 1, store val: bar", (
                f"Failed for tool={tool_name}"
            )

    tool_call = {
        "name": "tool3",
        "args": {"some_val": 1},
        "id": "some 0",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])
    node_result = node.invoke(
        {"messages": [msg], "bar": "baz"},
        config=_create_config_with_runtime(store=store),
    )
    graph_result = graph.invoke({"messages": [msg], "bar": "baz"})
    for result in (node_result, graph_result):
        result["messages"][-1]
        tool_message = result["messages"][-1]
        assert tool_message.content == "Some val: 1, store val: bar, state val: baz", (
            f"Failed for tool={tool_name}"
        )

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
        [AIMessage(content="", tool_calls=tool_calls)],
        config=_create_config_with_runtime(),
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
    agent = create_react_agent(
        model, [tool_return_direct, tool_normal], version=version
    )
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
    agent = create_react_agent(
        model, [tool_return_direct, tool_normal], version=version
    )
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
def test_react_with_subgraph_tools(
    sync_checkpointer: BaseCheckpointSaver, version: Literal["v1", "v2"]
) -> None:
    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define the subgraphs
    def add(state):
        return {"result": state["a"] + state["b"]}

    add_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(add)
        .add_edge(START, "add")
        .compile()
    )

    def multiply(state):
        return {"result": state["a"] * state["b"]}

    multiply_subgraph = (
        StateGraph(State, output_schema=Output)
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
        checkpointer=sync_checkpointer,
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
def test_react_agent_subgraph_streaming_sync(version: Literal["v1", "v2"]) -> None:
    """Test React agent streaming when used as a subgraph node sync version"""

    @dec_tool
    def get_weather(city: str) -> str:
        """Get the weather of a city."""
        return f"The weather of {city} is sunny."

    # Create a React agent
    model = FakeToolCallingModel(
        tool_calls=[
            [{"args": {"city": "Tokyo"}, "id": "1", "name": "get_weather"}],
            [],
        ]
    )

    agent = create_react_agent(
        model,
        tools=[get_weather],
        prompt="You are a helpful travel assistant.",
        version=version,
    )

    # Create a subgraph that uses the React agent as a node
    def react_agent_node(state: MessagesState, config: RunnableConfig) -> MessagesState:
        """Node that runs the React agent and collects streaming output."""
        collected_content = ""

        # Stream the agent output and collect content
        for msg_chunk, msg_metadata in agent.stream(
            {"messages": [("user", state["messages"][-1].content)]},
            config,
            stream_mode="messages",
        ):
            if hasattr(msg_chunk, "content") and msg_chunk.content:
                collected_content += msg_chunk.content

        return {"messages": [("assistant", collected_content)]}

    # Create the main workflow with the React agent as a subgraph node
    workflow = StateGraph(MessagesState)
    workflow.add_node("react_agent", react_agent_node)
    workflow.add_edge(START, "react_agent")
    workflow.add_edge("react_agent", "__end__")
    compiled_workflow = workflow.compile()

    # Test the streaming functionality
    result = compiled_workflow.invoke(
        {"messages": [("user", "What is the weather in Tokyo?")]}
    )

    # Verify the result contains expected structure
    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "What is the weather in Tokyo?"
    assert "assistant" in str(result["messages"][1])

    # Test streaming with subgraphs = True
    result = compiled_workflow.invoke(
        {"messages": [("user", "What is the weather in Tokyo?")]},
        subgraphs=True,
    )
    assert len(result["messages"]) == 2

    events = []
    for event in compiled_workflow.stream(
        {"messages": [("user", "What is the weather in Tokyo?")]},
        stream_mode="messages",
        subgraphs=False,
    ):
        events.append(event)

    assert len(events) == 0

    events = []
    for event in compiled_workflow.stream(
        {"messages": [("user", "What is the weather in Tokyo?")]},
        stream_mode="messages",
        subgraphs=True,
    ):
        events.append(event)

    assert len(events) == 3
    namespace, (msg, metadata) = events[0]
    # FakeToolCallingModel returns a single AIMessage with tool calls
    # The content of the AIMessage reflects the input message
    assert msg.content.startswith("You are a helpful travel assistant")
    namespace, (msg, metadata) = events[1]  # ToolMessage
    assert msg.content.startswith("The weather of Tokyo is sunny.")


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
async def test_react_agent_subgraph_streaming(version: Literal["v1", "v2"]) -> None:
    """Test React agent streaming when used as a subgraph node."""

    @dec_tool
    def get_weather(city: str) -> str:
        """Get the weather of a city."""
        return f"The weather of {city} is sunny."

    # Create a React agent
    model = FakeToolCallingModel(
        tool_calls=[
            [{"args": {"city": "Tokyo"}, "id": "1", "name": "get_weather"}],
            [],
        ]
    )

    agent = create_react_agent(
        model,
        tools=[get_weather],
        prompt="You are a helpful travel assistant.",
        version=version,
    )

    # Create a subgraph that uses the React agent as a node
    async def react_agent_node(
        state: MessagesState, config: RunnableConfig
    ) -> MessagesState:
        """Node that runs the React agent and collects streaming output."""
        collected_content = ""

        # Stream the agent output and collect content
        async for msg_chunk, msg_metadata in agent.astream(
            {"messages": [("user", state["messages"][-1].content)]},
            config,
            stream_mode="messages",
        ):
            if hasattr(msg_chunk, "content") and msg_chunk.content:
                collected_content += msg_chunk.content

        return {"messages": [("assistant", collected_content)]}

    # Create the main workflow with the React agent as a subgraph node
    workflow = StateGraph(MessagesState)
    workflow.add_node("react_agent", react_agent_node)
    workflow.add_edge(START, "react_agent")
    workflow.add_edge("react_agent", "__end__")
    compiled_workflow = workflow.compile()

    # Test the streaming functionality
    result = await compiled_workflow.ainvoke(
        {"messages": [("user", "What is the weather in Tokyo?")]}
    )

    # Verify the result contains expected structure
    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "What is the weather in Tokyo?"
    assert "assistant" in str(result["messages"][1])

    # Test streaming with subgraphs = True
    result = await compiled_workflow.ainvoke(
        {"messages": [("user", "What is the weather in Tokyo?")]},
        subgraphs=True,
    )
    assert len(result["messages"]) == 2

    events = []
    async for event in compiled_workflow.astream(
        {"messages": [("user", "What is the weather in Tokyo?")]},
        stream_mode="messages",
        subgraphs=False,
    ):
        events.append(event)

    assert len(events) == 0

    events = []
    async for event in compiled_workflow.astream(
        {"messages": [("user", "What is the weather in Tokyo?")]},
        stream_mode="messages",
        subgraphs=True,
    ):
        events.append(event)

    assert len(events) == 3
    namespace, (msg, metadata) = events[0]
    # FakeToolCallingModel returns a single AIMessage with tool calls
    # The content of the AIMessage reflects the input message
    assert msg.content.startswith("You are a helpful travel assistant")
    namespace, (msg, metadata) = events[1]  # ToolMessage
    assert msg.content.startswith("The weather of Tokyo is sunny.")


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_tool_node_node_interrupt(
    sync_checkpointer: BaseCheckpointSaver, version: str
) -> None:
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
        checkpointer=sync_checkpointer,
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
    assert task.interrupts == (
        Interrupt(
            value="provide value for foo",
            id=AnyStr(),
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


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_dynamic_model_basic(version: str) -> None:
    """Test basic dynamic model functionality."""

    def dynamic_model(state, runtime: Runtime):
        # Return different models based on state
        if "urgent" in state["messages"][-1].content:
            return FakeToolCallingModel(tool_calls=[])
        else:
            return FakeToolCallingModel(tool_calls=[])

    agent = create_react_agent(dynamic_model, [], version=version)

    result = agent.invoke({"messages": [HumanMessage("hello")]})
    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == "hello"

    result = agent.invoke({"messages": [HumanMessage("urgent help")]})
    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == "urgent help"


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_dynamic_model_with_tools(version: Literal["v1", "v2"]) -> None:
    """Test dynamic model with tool calling."""

    @dec_tool
    def basic_tool(x: int) -> str:
        """Basic tool."""
        return f"basic: {x}"

    @dec_tool
    def advanced_tool(x: int) -> str:
        """Advanced tool."""
        return f"advanced: {x}"

    def dynamic_model(state: dict, runtime: Runtime) -> BaseChatModel:
        # Return model with different behaviors based on message content
        if "advanced" in state["messages"][-1].content:
            return FakeToolCallingModel(
                tool_calls=[
                    [{"args": {"x": 1}, "id": "1", "name": "advanced_tool"}],
                    [],
                ]
            )
        else:
            return FakeToolCallingModel(
                tool_calls=[[{"args": {"x": 1}, "id": "1", "name": "basic_tool"}], []]
            )

    agent = create_react_agent(
        dynamic_model, [basic_tool, advanced_tool], version=version
    )

    # Test basic tool usage
    result = agent.invoke({"messages": [HumanMessage("basic request")]})
    assert len(result["messages"]) == 3
    tool_message = result["messages"][-1]
    assert tool_message.content == "basic: 1"
    assert tool_message.name == "basic_tool"

    # Test advanced tool usage
    result = agent.invoke({"messages": [HumanMessage("advanced request")]})
    assert len(result["messages"]) == 3
    tool_message = result["messages"][-1]
    assert tool_message.content == "advanced: 1"
    assert tool_message.name == "advanced_tool"


@dataclasses.dataclass
class Context:
    user_id: str


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_dynamic_model_with_context(version: str) -> None:
    """Test dynamic model using config parameters."""

    def dynamic_model(state, runtime: Runtime[Context]):
        # Use context to determine model behavior
        user_id = runtime.context.user_id
        if user_id == "user_premium":
            return FakeToolCallingModel(tool_calls=[])
        else:
            return FakeToolCallingModel(tool_calls=[])

    agent = create_react_agent(
        dynamic_model, [], context_schema=Context, version=version
    )

    # Test with basic user
    result = agent.invoke(
        {"messages": [HumanMessage("hello")]},
        context=Context(user_id="user_basic"),
    )
    assert len(result["messages"]) == 2

    # Test with premium user
    result = agent.invoke(
        {"messages": [HumanMessage("hello")]},
        context=Context(user_id="user_premium"),
    )
    assert len(result["messages"]) == 2


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_dynamic_model_with_state_schema(version: Literal["v1", "v2"]) -> None:
    """Test dynamic model with custom state schema."""

    class CustomDynamicState(AgentState):
        model_preference: str = "default"

    def dynamic_model(state: CustomDynamicState, runtime: Runtime) -> BaseChatModel:
        # Use custom state field to determine model
        if state.get("model_preference") == "advanced":
            return FakeToolCallingModel(tool_calls=[])
        else:
            return FakeToolCallingModel(tool_calls=[])

    agent = create_react_agent(
        dynamic_model, [], state_schema=CustomDynamicState, version=version
    )

    result = agent.invoke(
        {"messages": [HumanMessage("hello")], "model_preference": "advanced"}
    )
    assert len(result["messages"]) == 2
    assert result["model_preference"] == "advanced"


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_dynamic_model_with_prompt(version: Literal["v1", "v2"]) -> None:
    """Test dynamic model with different prompt types."""

    def dynamic_model(state: AgentState, runtime: Runtime) -> BaseChatModel:
        return FakeToolCallingModel(tool_calls=[])

    # Test with string prompt
    agent = create_react_agent(dynamic_model, [], prompt="system_msg", version=version)
    result = agent.invoke({"messages": [HumanMessage("human_msg")]})
    assert result["messages"][-1].content == "system_msg-human_msg"

    # Test with callable prompt
    def dynamic_prompt(state: AgentState) -> list[MessageLikeRepresentation]:
        """Generate a dynamic system message based on state."""
        return [{"role": "system", "content": "system_msg"}] + list(state["messages"])

    agent = create_react_agent(
        dynamic_model, [], prompt=dynamic_prompt, version=version
    )
    result = agent.invoke({"messages": [HumanMessage("human_msg")]})
    assert result["messages"][-1].content == "system_msg-human_msg"


async def test_dynamic_model_async() -> None:
    """Test dynamic model with async operations."""

    def dynamic_model(state: AgentState, runtime: Runtime) -> BaseChatModel:
        return FakeToolCallingModel(tool_calls=[])

    agent = create_react_agent(dynamic_model, [])

    result = await agent.ainvoke({"messages": [HumanMessage("hello async")]})
    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == "hello async"


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_dynamic_model_with_structured_response(version: str) -> None:
    """Test dynamic model with structured response format."""

    class TestResponse(BaseModel):
        message: str
        confidence: float

    def dynamic_model(state, runtime: Runtime):
        expected_response = TestResponse(message="dynamic response", confidence=0.9)
        return FakeToolCallingModel(
            tool_calls=[], structured_response=expected_response
        )

    agent = create_react_agent(
        dynamic_model, [], response_format=TestResponse, version=version
    )

    result = agent.invoke({"messages": [HumanMessage("hello")]})
    assert "structured_response" in result
    assert result["structured_response"].message == "dynamic response"
    assert result["structured_response"].confidence == 0.9


def test_dynamic_model_with_checkpointer(sync_checkpointer):
    """Test dynamic model with checkpointer."""
    call_count = 0

    def dynamic_model(state: AgentState, runtime: Runtime) -> BaseChatModel:
        nonlocal call_count
        call_count += 1
        return FakeToolCallingModel(
            tool_calls=[],
            # Incrementing the call count as it is used to assign an id
            # to the AIMessage.
            # The default reducer semantics are to overwrite an existing message
            # with the new one if the id matches.
            index=call_count,
        )

    agent = create_react_agent(dynamic_model, [], checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "test_dynamic"}}

    # First call
    result1 = agent.invoke({"messages": [HumanMessage("hello")]}, config)
    assert len(result1["messages"]) == 2  # Human + AI message

    # Second call - should load from checkpoint
    result2 = agent.invoke({"messages": [HumanMessage("world")]}, config)
    assert len(result2["messages"]) == 4

    # Dynamic model should be called each time
    assert call_count >= 2


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_dynamic_model_state_dependent_tools(version: Literal["v1", "v2"]) -> None:
    """Test dynamic model that changes available tools based on state."""

    @dec_tool
    def tool_a(x: int) -> str:
        """Tool A."""
        return f"A: {x}"

    @dec_tool
    def tool_b(x: int) -> str:
        """Tool B."""
        return f"B: {x}"

    def dynamic_model(state, runtime: Runtime):
        # Switch tools based on message history
        if any("use_b" in msg.content for msg in state["messages"]):
            return FakeToolCallingModel(
                tool_calls=[[{"args": {"x": 2}, "id": "1", "name": "tool_b"}], []]
            )
        else:
            return FakeToolCallingModel(
                tool_calls=[[{"args": {"x": 1}, "id": "1", "name": "tool_a"}], []]
            )

    agent = create_react_agent(dynamic_model, [tool_a, tool_b], version=version)

    # Ask to use tool B
    result = agent.invoke({"messages": [HumanMessage("use_b please")]})
    last_message = result["messages"][-1]
    assert isinstance(last_message, ToolMessage)
    assert last_message.content == "B: 2"

    # Ask to use tool A
    result = agent.invoke({"messages": [HumanMessage("hello")]})
    last_message = result["messages"][-1]
    assert isinstance(last_message, ToolMessage)
    assert last_message.content == "A: 1"


@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_dynamic_model_error_handling(version: Literal["v1", "v2"]) -> None:
    """Test error handling in dynamic model."""

    def failing_dynamic_model(state, runtime: Runtime):
        if "fail" in state["messages"][-1].content:
            raise ValueError("Dynamic model failed")
        return FakeToolCallingModel(tool_calls=[])

    agent = create_react_agent(failing_dynamic_model, [], version=version)

    # Normal operation should work
    result = agent.invoke({"messages": [HumanMessage("hello")]})
    assert len(result["messages"]) == 2

    # Should propagate the error
    with pytest.raises(ValueError, match="Dynamic model failed"):
        agent.invoke({"messages": [HumanMessage("fail now")]})


def test_dynamic_model_vs_static_model_behavior():
    """Test that dynamic and static models produce equivalent results when configured the same."""
    # Static model
    static_model = FakeToolCallingModel(tool_calls=[])
    static_agent = create_react_agent(static_model, [])

    # Dynamic model returning the same model
    def dynamic_model(state, runtime: Runtime):
        return FakeToolCallingModel(tool_calls=[])

    dynamic_agent = create_react_agent(dynamic_model, [])

    input_msg = {"messages": [HumanMessage("test message")]}

    static_result = static_agent.invoke(input_msg)
    dynamic_result = dynamic_agent.invoke(input_msg)

    # Results should be equivalent (content-wise, IDs may differ)
    assert len(static_result["messages"]) == len(dynamic_result["messages"])
    assert static_result["messages"][0].content == dynamic_result["messages"][0].content
    assert static_result["messages"][1].content == dynamic_result["messages"][1].content


def test_dynamic_model_receives_correct_state():
    """Test that the dynamic model function receives the correct state, not the model input."""
    received_states = []

    class CustomAgentState(AgentState):
        custom_field: str

    def dynamic_model(state, runtime: Runtime) -> BaseChatModel:
        # Capture the state that's passed to the dynamic model function
        received_states.append(state)
        return FakeToolCallingModel(tool_calls=[])

    agent = create_react_agent(dynamic_model, [], state_schema=CustomAgentState)

    # Test with initial state
    input_state = {"messages": [HumanMessage("hello")], "custom_field": "test_value"}
    agent.invoke(input_state)

    # The dynamic model function should receive the original state, not the processed model input
    assert len(received_states) == 1
    received_state = received_states[0]

    # Should have the custom field from original state
    assert "custom_field" in received_state
    assert received_state["custom_field"] == "test_value"

    # Should have the original messages
    assert len(received_state["messages"]) == 1
    assert received_state["messages"][0].content == "hello"


async def test_dynamic_model_receives_correct_state_async():
    """Test that the async dynamic model function receives the correct state, not the model input."""
    received_states = []

    class CustomAgentStateAsync(AgentState):
        custom_field: str

    def dynamic_model(state, runtime: Runtime):
        # Capture the state that's passed to the dynamic model function
        received_states.append(state)
        return FakeToolCallingModel(tool_calls=[])

    agent = create_react_agent(dynamic_model, [], state_schema=CustomAgentStateAsync)

    # Test with initial state
    input_state = {
        "messages": [HumanMessage("hello async")],
        "custom_field": "test_value_async",
    }
    await agent.ainvoke(input_state)

    # The dynamic model function should receive the original state, not the processed model input
    assert len(received_states) == 1
    received_state = received_states[0]

    # Should have the custom field from original state
    assert "custom_field" in received_state
    assert received_state["custom_field"] == "test_value_async"

    # Should have the original messages
    assert len(received_state["messages"]) == 1
    assert received_state["messages"][0].content == "hello async"


def test_pre_model_hook() -> None:
    model = FakeToolCallingModel(tool_calls=[])

    # Test `llm_input_messages`
    def pre_model_hook(state: AgentState):
        return {"llm_input_messages": [HumanMessage("Hello!")]}

    agent = create_react_agent(model, [], pre_model_hook=pre_model_hook)
    assert "pre_model_hook" in agent.nodes
    result = agent.invoke({"messages": [HumanMessage("hi?")]})
    assert result == {
        "messages": [
            _AnyIdHumanMessage(content="hi?"),
            AIMessage(content="Hello!", id="0"),
        ]
    }

    # Test `messages`
    def pre_model_hook(state: AgentState):
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage("Hello!")]
        }

    agent = create_react_agent(model, [], pre_model_hook=pre_model_hook)
    result = agent.invoke({"messages": [HumanMessage("hi?")]})
    assert result == {
        "messages": [
            _AnyIdHumanMessage(content="Hello!"),
            AIMessage(content="Hello!", id="1"),
        ]
    }


def test_post_model_hook() -> None:
    class FlagState(AgentState):
        flag: bool

    model = FakeToolCallingModel(tool_calls=[])

    def post_model_hook(state: FlagState) -> dict[str, bool]:
        return {"flag": True}

    pmh_agent = create_react_agent(
        model, [], post_model_hook=post_model_hook, state_schema=FlagState
    )

    assert "post_model_hook" in pmh_agent.nodes

    result = pmh_agent.invoke({"messages": [HumanMessage("hi?")], "flag": False})
    assert result["flag"] is True

    events = list(pmh_agent.stream({"messages": [HumanMessage("hi?")], "flag": False}))
    assert events == [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="hi?",
                        additional_kwargs={},
                        response_metadata={},
                        id="1",
                    )
                ]
            }
        },
        {"post_model_hook": {"flag": True}},
    ]


def test_post_model_hook_with_structured_output() -> None:
    class WeatherResponse(BaseModel):
        temperature: float = Field(description="The temperature in fahrenheit")

    tool_calls = [[{"args": {}, "id": "1", "name": "get_weather"}]]

    def get_weather():
        """Get the weather"""
        return "The weather is sunny and 75°F."

    expected_structured_response = WeatherResponse(temperature=75)
    model = FakeToolCallingModel(
        tool_calls=tool_calls, structured_response=expected_structured_response
    )

    class State(AgentState):
        flag: bool
        structured_response: WeatherResponse

    def post_model_hook(state: State) -> dict[str, bool] | Command:
        return {"flag": True}

    agent = create_react_agent(
        model,
        [get_weather],
        response_format=WeatherResponse,
        post_model_hook=post_model_hook,
        state_schema=State,
    )

    assert "post_model_hook" in agent.nodes
    assert "generate_structured_response" in agent.nodes

    response = agent.invoke(
        {"messages": [HumanMessage("What's the weather?")], "flag": False}
    )
    assert response["flag"] is True
    assert response["structured_response"] == expected_structured_response

    events = list(
        agent.stream({"messages": [HumanMessage("What's the weather?")], "flag": False})
    )
    assert "generate_structured_response" in events[-1]
    assert events == [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="What's the weather?",
                        additional_kwargs={},
                        response_metadata={},
                        id="2",
                        tool_calls=[
                            {
                                "name": "get_weather",
                                "args": {},
                                "id": "1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }
        },
        {"post_model_hook": {"flag": True}},
        {
            "tools": {
                "messages": [
                    _AnyIdToolMessage(
                        content="The weather is sunny and 75°F.",
                        name="get_weather",
                        tool_call_id="1",
                    ),
                ]
            }
        },
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="What's the weather?-What's the weather?-The weather is sunny and 75°F.",
                        additional_kwargs={},
                        response_metadata={},
                        id="3",
                        tool_calls=[
                            {
                                "name": "get_weather",
                                "args": {},
                                "id": "1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }
        },
        {"post_model_hook": {"flag": True}},
        {
            "generate_structured_response": {
                "structured_response": WeatherResponse(temperature=75.0)
            }
        },
    ]


@pytest.mark.parametrize(
    "state_schema", [AgentStateExtraKey, AgentStateExtraKeyPydantic]
)
def test_create_react_agent_inject_vars_with_post_model_hook(
    state_schema: StateSchemaType,
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

    def post_model_hook(state: dict) -> dict:
        """Post model hook is injecting a new foo key."""
        return {"foo": 2}

    model = FakeToolCallingModel(tool_calls=[[tool_call], []])
    agent = create_react_agent(
        model,
        ToolNode([tool1], handle_tool_errors=False),
        state_schema=state_schema,
        store=store,
        post_model_hook=post_model_hook,
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
