import pytest
from langchain_core.callbacks import AsyncCallbackManager, BaseCallbackHandler

from langgraph._internal._config import get_async_callback_manager_for_config
from langgraph.graph import StateGraph

pytestmark = pytest.mark.anyio


def test_new_async_manager_includes_tags() -> None:
    config = {"callbacks": None}
    manager = get_async_callback_manager_for_config(config, tags=["x", "y"])
    assert isinstance(manager, AsyncCallbackManager)
    assert manager.inheritable_tags == ["x", "y"]


def test_new_async_manager_merges_tags_with_config() -> None:
    config = {"callbacks": None, "tags": ["a"]}
    manager = get_async_callback_manager_for_config(config, tags=["b"])
    assert manager.inheritable_tags == ["a", "b"]


async def test_with_config_callbacks_preserved_in_astream_events() -> None:
    class TrackingCallback(BaseCallbackHandler):
        def __init__(self) -> None:
            self.called = False

        def on_chain_start(self, *args, **kwargs) -> None:
            self.called = True

    builder = StateGraph(dict)
    builder.add_node("node", lambda state: state)
    builder.add_edge("__start__", "node")
    cb = TrackingCallback()
    graph = builder.compile().with_config({"callbacks": [cb]})
    async for _ in graph.astream_events({}, version="v2"):
        pass
    assert cb.called
