from typing_extensions import NotRequired, Required, TypedDict

from langgraph.graph import StateGraph
from langgraph.managed import RemainingSteps


class StatePlain(TypedDict):
    remaining_steps: RemainingSteps


class StateNotRequired(TypedDict):
    remaining_steps: NotRequired[RemainingSteps]


class StateRequired(TypedDict):
    remaining_steps: Required[RemainingSteps]


def test_managed_values_recognized() -> None:
    graph = StateGraph(StatePlain)
    assert "remaining_steps" in graph.managed

    graph = StateGraph(StateNotRequired)
    assert "remaining_steps" in graph.managed

    graph = StateGraph(StateRequired)
    assert "remaining_steps" in graph.managed
