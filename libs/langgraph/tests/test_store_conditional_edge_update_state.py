import pytest
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.stores import BaseStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

pytestmark = pytest.mark.anyio


class State(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]


def _build() -> StateGraph:
    def node(state: State) -> State:  # noqa: ARG001
        return {}

    def route_with_store(
        state: State,  # noqa: ARG001
        config: RunnableConfig,  # noqa: ARG001
        store: BaseStore,
    ) -> str:
        # Touch store so the injected param is provably resolved.
        assert store is not None
        return END

    return (
        StateGraph(State)
        .add_node("node", node)
        .add_edge(START, "node")
        .add_conditional_edges("node", route_with_store)
    )


def test_update_state_conditional_edge_with_store() -> None:
    """A conditional edge whose router asks for `store` (typed with
    langchain_core.stores.BaseStore) must work under update_state, not only
    under invoke (issue #6340)."""
    graph = _build().compile(checkpointer=MemorySaver(), store=InMemoryStore())
    cfg = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": [HumanMessage(content="hi")]}, cfg)
    # Previously raised: Missing required config key 'store' for 'route_with_store'
    graph.update_state(cfg, {"messages": [HumanMessage(content="bye")]})


async def test_aupdate_state_conditional_edge_with_store() -> None:
    """Async counterpart of test_update_state_conditional_edge_with_store."""
    graph = _build().compile(checkpointer=MemorySaver(), store=InMemoryStore())
    cfg = {"configurable": {"thread_id": "2"}}
    await graph.ainvoke({"messages": [HumanMessage(content="hi")]}, cfg)
    await graph.aupdate_state(cfg, {"messages": [HumanMessage(content="bye")]})
