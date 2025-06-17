import pytest
from typing_extensions import TypedDict

from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy
from langgraph.warnings import LangGraphDeprecatedSinceV05


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
