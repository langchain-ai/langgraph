import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ConversationState(MessagesState):
    pass


def build_sync_graph(*, nested: bool):
    def leaf_node(state: ConversationState) -> ConversationState:
        return {
            "messages": state["messages"]
            + [{"role": "assistant", "content": "leaf-reply"}]
        }

    leaf_graph = (
        StateGraph(ConversationState)
        .add_node("leaf", leaf_node)
        .add_edge(START, "leaf")
        .compile(checkpointer=True)
    )

    if nested:
        middle_graph = (
            StateGraph(ConversationState)
            .add_node("middle", leaf_graph)
            .add_edge(START, "middle")
            .compile(checkpointer=True)
        )
        target = middle_graph
        target_name = "agent"
    else:
        target = leaf_graph
        target_name = "agent"

    graph = (
        StateGraph(ConversationState)
        .add_node(target_name, target)
        .add_edge(START, target_name)
        .compile(checkpointer=InMemorySaver())
    )
    return graph


def build_async_graph(*, nested: bool):
    async def leaf_node(state: ConversationState) -> ConversationState:
        return {
            "messages": state["messages"]
            + [{"role": "assistant", "content": "leaf-reply"}]
        }

    leaf_graph = (
        StateGraph(ConversationState)
        .add_node("leaf", leaf_node)
        .add_edge(START, "leaf")
        .compile(checkpointer=True)
    )

    if nested:
        middle_graph = (
            StateGraph(ConversationState)
            .add_node("middle", leaf_graph)
            .add_edge(START, "middle")
            .compile(checkpointer=True)
        )
        target = middle_graph
        target_name = "agent"
    else:
        target = leaf_graph
        target_name = "agent"

    graph = (
        StateGraph(ConversationState)
        .add_node(target_name, target)
        .add_edge(START, target_name)
        .compile(checkpointer=InMemorySaver())
    )
    return graph


def invoke_twice_sync(graph) -> dict:
    config = {"configurable": {"thread_id": "thread-sync"}}
    graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)
    graph.invoke(
        {"messages": [{"role": "user", "content": "what did I say?"}]},
        config,
    )
    return config


async def invoke_twice_async(graph) -> dict:
    config = {"configurable": {"thread_id": "thread-async"}}
    await graph.ainvoke({"messages": [{"role": "user", "content": "hi"}]}, config)
    await graph.ainvoke(
        {"messages": [{"role": "user", "content": "what did I say?"}]},
        config,
    )
    return config


def assert_completed_subgraph_snapshot(state) -> None:
    assert len(state.tasks) == 1
    subgraph_task = state.tasks[0]
    assert subgraph_task.name == "agent"
    assert subgraph_task.state is not None
    assert subgraph_task.state.metadata is not None
    assert subgraph_task.state.values["messages"][-1].content == "leaf-reply"
    assert len(subgraph_task.state.values["messages"]) == 4


def assert_nested_completed_subgraph_snapshot(state) -> None:
    assert len(state.tasks) == 1
    middle_task = state.tasks[0]
    assert middle_task.name == "agent"
    assert middle_task.state is not None
    assert len(middle_task.state.tasks) == 1

    leaf_task = middle_task.state.tasks[0]
    assert leaf_task.name == "middle"
    assert leaf_task.state is not None
    assert leaf_task.state.values["messages"][-1].content == "leaf-reply"
    assert len(leaf_task.state.values["messages"]) == 4


def test_get_state_with_subgraphs_returns_completed_subgraph_state_sync() -> None:
    graph = build_sync_graph(nested=False)
    config = invoke_twice_sync(graph)

    state = graph.get_state(config, subgraphs=True)

    assert state.next == ()
    assert state.values["messages"][-1].content == "leaf-reply"
    assert_completed_subgraph_snapshot(state)


def test_get_state_with_subgraphs_returns_completed_nested_subgraph_state_sync() -> None:
    graph = build_sync_graph(nested=True)
    config = invoke_twice_sync(graph)

    state = graph.get_state(config, subgraphs=True)

    assert state.next == ()
    assert_nested_completed_subgraph_snapshot(state)


async def test_get_state_with_subgraphs_returns_completed_subgraph_state_async() -> None:
    graph = build_async_graph(nested=False)
    config = await invoke_twice_async(graph)

    state = await graph.aget_state(config, subgraphs=True)

    assert state.next == ()
    assert state.values["messages"][-1].content == "leaf-reply"
    assert_completed_subgraph_snapshot(state)


async def test_get_state_with_subgraphs_returns_completed_nested_subgraph_state_async() -> None:
    graph = build_async_graph(nested=True)
    config = await invoke_twice_async(graph)

    state = await graph.aget_state(config, subgraphs=True)

    assert state.next == ()
    assert_nested_completed_subgraph_snapshot(state)


def test_get_state_without_subgraphs_keeps_completed_tasks_hidden_sync() -> None:
    graph = build_sync_graph(nested=False)
    config = invoke_twice_sync(graph)

    state = graph.get_state(config, subgraphs=False)

    assert state.tasks == ()
    assert state.next == ()


async def test_get_state_without_subgraphs_keeps_completed_tasks_hidden_async() -> None:
    graph = build_async_graph(nested=False)
    config = await invoke_twice_async(graph)

    state = await graph.aget_state(config, subgraphs=False)

    assert state.tasks == ()
    assert state.next == ()
