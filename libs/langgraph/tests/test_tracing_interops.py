import json
import sys
import time
from typing import Any, Callable, Tuple, TypedDict, TypeVar
from unittest.mock import MagicMock

import langsmith as ls
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer

from langgraph.graph import StateGraph


def _get_mock_client(**kwargs: Any) -> ls.Client:
    mock_session = MagicMock()
    return ls.Client(session=mock_session, api_key="test", **kwargs)


def _get_calls(
    mock_client: Any,
    verbs: set[str] = {"POST"},
) -> list:
    return [
        c
        for c in mock_client.session.request.mock_calls
        if c.args and c.args[0] in verbs
    ]


T = TypeVar("T")


def wait_for(
    condition: Callable[[], Tuple[T, bool]],
    max_sleep_time: int = 10,
    sleep_time: int = 3,
) -> T:
    """Wait for a condition to be true."""
    start_time = time.time()
    last_e = None
    while time.time() - start_time < max_sleep_time:
        try:
            res, cond = condition()
            if cond:
                return res
        except Exception as e:
            last_e = e
            time.sleep(sleep_time)
    total_time = time.time() - start_time
    if last_e is not None:
        raise last_e
    raise ValueError(f"Callable did not return within {total_time}")


async def test_nested_tracing():
    lt_py_311 = sys.version_info < (3, 11)
    mock_client = _get_mock_client()

    class State(TypedDict):
        value: str

    @ls.traceable
    async def some_traceable(content: State):
        return await child_graph.ainvoke(content)

    async def parent_node(state: State, config: RunnableConfig) -> State:
        if lt_py_311:
            result = await some_traceable(state, langsmith_extra={"config": config})
        else:
            result = await some_traceable(state)
        return {"value": f"parent_{result['value']}"}

    async def child_node(state: State) -> State:
        return {"value": f"child_{state['value']}"}

    child_builder = StateGraph(State)
    child_builder.add_node(child_node)
    child_builder.add_edge("__start__", "child_node")
    child_graph = child_builder.compile()

    parent_builder = StateGraph(State)
    parent_builder.add_node(parent_node)
    parent_builder.add_edge("__start__", "parent_node")
    parent_graph = parent_builder.compile()

    tracer = LangChainTracer(client=mock_client)
    result = await parent_graph.ainvoke({"value": "input"}, {"callbacks": [tracer]})

    assert result == {"value": "parent_child_input"}

    def get_posts():
        post_calls = _get_calls(mock_client, verbs={"POST"})

        posts = [p for c in post_calls for p in json.loads(c.kwargs["data"])["post"]]
        names = [p.get("name") for p in posts]
        if "child_node" in names:
            return posts, True
        return None, False

    posts = wait_for(get_posts)
    # If the callbacks weren't propagated correctly, we'd
    # end up with broken dotted_orders
    parent_run = next(data for data in posts if data["name"] == "parent_node")
    child_run = next(data for data in posts if data["name"] == "child_node")
    traceable_run = next(data for data in posts if data["name"] == "some_traceable")

    assert child_run["dotted_order"].startswith(traceable_run["dotted_order"])
    assert traceable_run["dotted_order"].startswith(parent_run["dotted_order"])

    assert child_run["parent_run_id"] == traceable_run["id"]
    assert traceable_run["parent_run_id"] == parent_run["id"]
    assert parent_run["trace_id"] == child_run["trace_id"] == traceable_run["trace_id"]
