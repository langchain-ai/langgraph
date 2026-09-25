import pytest
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import TasksTransformer
from langgraph.types import Command


class Counter(TypedDict):
    count: int


def graph_with(output):
    graph = StateGraph(Counter)
    graph.add_node("work", lambda state: output)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


@pytest.mark.parametrize(
    "output,kind",
    [
        ({"count": 2}, "dict"),
        (Command(update={"count": 2}), "Command"),
        ([Command(update={"count": 2})], "list"),
    ],
)
def test_v3_task_result_retains_node_output_type(output, kind):
    run = graph_with(output).stream_events(
        {"count": 1}, version="v3", transformers=[TasksTransformer]
    )
    events = list(run.tasks)
    result = next(event for event in events if "result" in event)
    assert result["result"] == {"count": 2}
    assert result["output_type"] == kind


@pytest.mark.parametrize("output", [{"count": 2}, Command(update={"count": 2})])
def test_legacy_task_result_is_unchanged(output):
    events = list(graph_with(output).stream({"count": 1}, stream_mode="tasks"))
    result = next(event for event in events if "result" in event)
    assert "output_type" not in result


@pytest.mark.anyio
async def test_async_v3_task_output_type():
    run = await graph_with(Command(update={"count": 2})).astream_events(
        {"count": 1}, version="v3", transformers=[TasksTransformer]
    )
    events = [event async for event in run.tasks]
    result = next(event for event in events if "result" in event)
    assert result["output_type"] == "Command"


@pytest.mark.anyio
async def test_parallel_tasks_keep_independent_output_types():
    import operator
    from typing import Annotated

    class Totals(TypedDict):
        count: Annotated[int, operator.add]

    graph = StateGraph(Totals)
    graph.add_node("plain", lambda state: {"count": 1})
    graph.add_node("routed", lambda state: Command(update={"count": 2}))
    for name in ["plain", "routed"]:
        graph.add_edge(START, name)
        graph.add_edge(name, END)
    run = await graph.compile().astream_events(
        {"count": 0}, version="v3", transformers=[TasksTransformer]
    )
    results = {
        event["name"]: event["output_type"]
        async for event in run.tasks
        if "result" in event
    }
    assert results == {"plain": "dict", "routed": "Command"}


@pytest.mark.anyio
async def test_timed_async_node_retains_output_type():
    from langgraph.types import TimeoutPolicy

    async def node(state):
        return Command(update={"count": 2})

    graph = StateGraph(Counter)
    graph.add_node("timed", node, timeout=TimeoutPolicy(run_timeout=10))
    graph.add_edge(START, "timed")
    graph.add_edge("timed", END)
    run = await graph.compile().astream_events(
        {"count": 1}, version="v3", transformers=[TasksTransformer]
    )
    result = next(
        event for event in [item async for item in run.tasks] if "result" in event
    )
    assert result["output_type"] == "Command"


@pytest.mark.anyio
async def test_retry_records_only_the_successful_return_type():
    from langgraph.types import RetryPolicy

    attempts = 0

    async def node(state):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("retry this attempt")
        return Command(update={"count": 3})

    graph = StateGraph(Counter)
    graph.add_node(
        "retrying",
        node,
        retry_policy=RetryPolicy(
            initial_interval=0, max_attempts=2, retry_on=ValueError
        ),
    )
    graph.add_edge(START, "retrying")
    graph.add_edge("retrying", END)
    run = await graph.compile().astream_events(
        {"count": 1}, version="v3", transformers=[TasksTransformer]
    )
    results = [event async for event in run.tasks if "result" in event]
    assert attempts == 2
    assert len(results) == 1
    assert results[0]["result"] == {"count": 3}
    assert results[0]["output_type"] == "Command"


@pytest.mark.anyio
async def test_interrupt_has_no_successful_output_type():
    from langgraph.types import interrupt

    def node(state):
        interrupt("approval needed")
        return {"count": 3}

    graph = StateGraph(Counter)
    graph.add_node("pausing", node)
    graph.add_edge(START, "pausing")
    graph.add_edge("pausing", END)
    run = await graph.compile().astream_events(
        {"count": 1}, version="v3", transformers=[TasksTransformer]
    )
    results = [event async for event in run.tasks if "result" in event]
    assert len(results) == 1
    assert results[0]["interrupts"]
    assert "output_type" not in results[0]
