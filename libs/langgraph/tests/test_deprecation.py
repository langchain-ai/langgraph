from __future__ import annotations

import warnings
from typing import Any, Optional

import pytest
from langchain_core.runnables import RunnableConfig
from pytest_mock import MockerFixture
from typing_extensions import NotRequired, TypedDict

from langgraph.channels.last_value import LastValue
from langgraph.errors import NodeInterrupt
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph
from langgraph.graph.message import MessageGraph
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import Interrupt, RetryPolicy
from langgraph.warnings import LangGraphDeprecatedSinceV05, LangGraphDeprecatedSinceV10


class PlainState(TypedDict): ...


def test_add_node_retry_arg() -> None:
    builder = StateGraph(PlainState)

    with pytest.warns(
        LangGraphDeprecatedSinceV05,
        match="`retry` is deprecated and will be removed. Please use `retry_policy` instead.",
    ):
        builder.add_node("test_node", lambda state: state, retry=RetryPolicy())  # type: ignore[arg-type]


def test_task_retry_arg() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV05,
        match="`retry` is deprecated and will be removed. Please use `retry_policy` instead.",
    ):

        @task(retry=RetryPolicy())  # type: ignore[arg-type]
        def my_task(state: PlainState) -> PlainState:
            return state


def test_entrypoint_retry_arg() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV05,
        match="`retry` is deprecated and will be removed. Please use `retry_policy` instead.",
    ):

        @entrypoint(retry=RetryPolicy())  # type: ignore[arg-type]
        def my_entrypoint(state: PlainState) -> PlainState:
            return state


def test_state_graph_input_schema() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV05,
        match="`input` is deprecated and will be removed. Please use `input_schema` instead.",
    ):
        StateGraph(PlainState, input=PlainState)  # type: ignore[arg-type]


def test_state_graph_output_schema() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV05,
        match="`output` is deprecated and will be removed. Please use `output_schema` instead.",
    ):
        StateGraph(PlainState, output=PlainState)  # type: ignore[arg-type]


def test_add_node_input_schema() -> None:
    builder = StateGraph(PlainState)

    with pytest.warns(
        LangGraphDeprecatedSinceV05,
        match="`input` is deprecated and will be removed. Please use `input_schema` instead.",
    ):
        builder.add_node("test_node", lambda state: state, input=PlainState)  # type: ignore[arg-type]


def test_constants_deprecation() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="Importing Send from langgraph.constants is deprecated. Please use 'from langgraph.types import Send' instead.",
    ):
        from langgraph.constants import Send  # noqa: F401

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="Importing Interrupt from langgraph.constants is deprecated. Please use 'from langgraph.types import Interrupt' instead.",
    ):
        from langgraph.constants import Interrupt  # noqa: F401


def test_pregel_types_deprecation() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="Importing from langgraph.pregel.types is deprecated. Please use 'from langgraph.types import ...' instead.",
    ):
        from langgraph.pregel.types import StateSnapshot  # noqa: F401


def test_config_schema_deprecation() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_schema` is deprecated and will be removed. Please use `context_schema` instead.",
    ):
        builder = StateGraph(PlainState, config_schema=PlainState)
        assert builder.context_schema == PlainState

    builder.add_node("test_node", lambda state: state)
    builder.set_entry_point("test_node")
    graph = builder.compile()

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_schema` is deprecated. Use `get_context_jsonschema` for the relevant schema instead.",
    ):
        assert graph.config_schema() is not None

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`get_config_jsonschema` is deprecated. Use `get_context_jsonschema` instead.",
    ):
        graph.get_config_jsonschema()


def test_config_schema_deprecation_on_entrypoint() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_schema` is deprecated and will be removed. Please use `context_schema` instead.",
    ):

        @entrypoint(config_schema=PlainState)  # type: ignore[arg-type]
        def my_entrypoint(state: PlainState) -> PlainState:
            return state

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_schema` is deprecated. Use `get_context_jsonschema` for the relevant schema instead.",
    ):
        assert my_entrypoint.context_schema == PlainState
        assert my_entrypoint.config_schema() is not None


@pytest.mark.filterwarnings("ignore:`config_type` is deprecated")
def test_config_type_deprecation_pregel(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = NodeBuilder().subscribe_only("input").do(add_one).write_to("output")

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_type` is deprecated and will be removed. Please use `context_schema` instead.",
    ):
        instance = Pregel(
            nodes={
                "one": chain,
            },
            channels={
                "input": LastValue(int),
                "output": LastValue(int),
            },
            input_channels="input",
            output_channels="output",
            config_type=PlainState,
        )
        assert instance.context_schema == PlainState


def test_interrupt_attributes_deprecation() -> None:
    interrupt = Interrupt(value="question", id="abc")

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`interrupt_id` is deprecated. Use `id` instead.",
    ):
        interrupt.interrupt_id


def test_node_interrupt_deprecation() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="NodeInterrupt is deprecated. Please use `langgraph.types.interrupt` instead.",
    ):
        NodeInterrupt(value="test")


def test_deprecated_import() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="Importing PREVIOUS from langgraph.constants is deprecated. This constant is now private and should not be used directly.",
    ):
        from langgraph.constants import PREVIOUS  # noqa: F401


@pytest.mark.filterwarnings(
    "ignore:`durability` has no effect when no checkpointer is present"
)
def test_checkpoint_during_deprecation_state_graph() -> None:
    class CheckDurability(TypedDict):
        durability: NotRequired[str]

    def plain_node(state: CheckDurability, config: RunnableConfig) -> CheckDurability:
        return {"durability": config["configurable"]["__pregel_durability"]}

    builder = StateGraph(CheckDurability)
    builder.add_node("plain_node", plain_node)
    builder.set_entry_point("plain_node")
    graph = builder.compile()

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`checkpoint_during` is deprecated and will be removed. Please use `durability` instead.",
    ):
        result = graph.invoke({}, checkpoint_during=True)
        assert result["durability"] == "async"

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`checkpoint_during` is deprecated and will be removed. Please use `durability` instead.",
    ):
        result = graph.invoke({}, checkpoint_during=False)
        assert result["durability"] == "exit"

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`checkpoint_during` is deprecated and will be removed. Please use `durability` instead.",
    ):
        for chunk in graph.stream({}, checkpoint_during=True):  # type: ignore[arg-type]
            assert chunk["plain_node"]["durability"] == "async"

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`checkpoint_during` is deprecated and will be removed. Please use `durability` instead.",
    ):
        for chunk in graph.stream({}, checkpoint_during=False):  # type: ignore[arg-type]
            assert chunk["plain_node"]["durability"] == "exit"


def test_config_parameter_incorrect_typing() -> None:
    """Test that a warning is raised when config parameter is typed incorrectly."""
    builder = StateGraph(PlainState)

    # Test sync function with config: dict
    with pytest.warns(
        UserWarning,
        match="The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig | None', not '.*dict.*'. ",
    ):

        def sync_node_with_dict_config(state: PlainState, config: dict) -> PlainState:
            return state

        builder.add_node(sync_node_with_dict_config)

    # Test async function with config: dict
    with pytest.warns(
        UserWarning,
        match="The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig | None', not '.*dict.*'. ",
    ):

        async def async_node_with_dict_config(
            state: PlainState, config: dict
        ) -> PlainState:
            return state

        builder.add_node(async_node_with_dict_config)

    # Test with other incorrect types
    with pytest.warns(
        UserWarning,
        match="The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig | None', not '.*Any.*'. ",
    ):

        def sync_node_with_any_config(state: PlainState, config: Any) -> PlainState:
            return state

        builder.add_node(sync_node_with_any_config)

    with pytest.warns(
        UserWarning,
        match="The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig | None', not '.*Any.*'. ",
    ):

        async def async_node_with_any_config(
            state: PlainState, config: Any
        ) -> PlainState:
            return state

        builder.add_node(async_node_with_any_config)

    with warnings.catch_warnings(record=True) as w:

        def node_with_correct_config(
            state: PlainState, config: RunnableConfig
        ) -> PlainState:
            return state

        builder.add_node(node_with_correct_config)

        def node_with_optional_config(
            state: PlainState,
            config: Optional[RunnableConfig],  # noqa: UP045
        ) -> PlainState:
            return state

        builder.add_node(node_with_optional_config)

        def node_with_untyped_config(state: PlainState, config) -> PlainState:
            return state

        builder.add_node(node_with_untyped_config)

        async def async_node_with_correct_config(
            state: PlainState, config: RunnableConfig
        ) -> PlainState:
            return state

        builder.add_node(async_node_with_correct_config)

        async def async_node_with_optional_config(
            state: PlainState,
            config: Optional[RunnableConfig],  # noqa: UP045
        ) -> PlainState:
            return state

        builder.add_node(async_node_with_optional_config)

        async def async_node_with_untyped_config(
            state: PlainState, config
        ) -> PlainState:
            return state

        builder.add_node(async_node_with_untyped_config)
        assert len(w) == 0


def test_message_graph_deprecation() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="MessageGraph is deprecated in LangGraph v1.0.0, to be removed in v2.0.0. Please use StateGraph with a `messages` key instead.",
    ):
        MessageGraph()
