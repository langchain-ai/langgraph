import pytest
from pytest_mock import MockerFixture
from typing_extensions import TypedDict

from langgraph.channels.last_value import LastValue
from langgraph.errors import NodeInterrupt
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph
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


@pytest.mark.filterwarnings("ignore:`config_schema` is deprecated")
@pytest.mark.filterwarnings("ignore:`get_config_jsonschema` is deprecated")
def test_config_schema_deprecation() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_schema` is deprecated and will be removed. Please use `context_schema` instead.",
    ):
        builder = StateGraph(PlainState, config_schema=PlainState)

    builder.add_node("test_node", lambda state: state)
    builder.set_entry_point("test_node")
    graph = builder.compile()

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_schema` is deprecated. Use `get_context_jsonschema` for the relevant schema instead.",
    ):
        graph.config_schema()

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`get_config_jsonschema` is deprecated. Use `get_context_jsonschema` instead.",
    ):
        graph.get_config_jsonschema()


def test_config_type_deprecation_pregel(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = NodeBuilder().subscribe_only("input").do(add_one).write_to("output")

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_type` is deprecated and will be removed. Please use `context_schema` instead.",
    ):
        Pregel(
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


@pytest.mark.filterwarnings("ignore:`interrupt_id` is deprecated. Use `id` instead.")
def test_interrupt_attributes_deprecation() -> None:
    interrupt = Interrupt(value="question", id="abc")

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`interrupt_id` is deprecated. Use `id` instead.",
    ):
        interrupt.interrupt_id


@pytest.mark.filterwarnings("ignore:NodeInterrupt is deprecated.")
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
