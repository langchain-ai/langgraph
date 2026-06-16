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


class _TrackingCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.called = False

    def on_chain_start(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.called = True


async def test_with_config_callbacks_preserved_in_astream_events() -> None:
    """A callback bound via .with_config(...) must survive when
    astream_events injects its own internal callback handler.

    Pre-fix: ensure_config overwrites the callbacks key, dropping the
    bound handler. Post-fix: the handler list is merged.
    """
    builder = StateGraph(dict)
    builder.add_node("node", lambda state: state)
    builder.add_edge("__start__", "node")
    cb = _TrackingCallback()
    graph = builder.compile().with_config({"callbacks": [cb]})
    async for _ in graph.astream_events({}, version="v2"):
        pass
    assert cb.called, "user-bound callback was dropped by ensure_config overwrite"


async def test_with_config_configurable_preserved_on_invoke() -> None:
    """A configurable key bound via .with_config(...) must survive when
    invoke-time config supplies a different configurable key.

    Pre-fix: ensure_config overwrites the entire configurable dict.
    Post-fix: the two dicts are shallow-merged per key.
    """
    builder = StateGraph(dict)
    captured: dict = {}

    def node(state, config):  # noqa: ANN001
        captured.update(config.get("configurable") or {})
        return state

    builder.add_node("node", node)
    builder.add_edge("__start__", "node")
    graph = builder.compile().with_config({"configurable": {"ls_agent_type": "root"}})
    await graph.ainvoke({}, {"configurable": {"thread_id": "T1"}})
    assert captured.get("ls_agent_type") == "root", (
        "bound configurable key was dropped by ensure_config overwrite"
    )
    assert captured.get("thread_id") == "T1", "invoke-time key not present"


async def test_with_config_metadata_preserved_on_invoke() -> None:
    """A metadata key bound via .with_config(...) must survive when
    invoke-time config supplies a different metadata key.

    Pre-fix: ensure_config overwrites the entire metadata dict.
    Post-fix: the two dicts are shallow-merged per key.
    """
    builder = StateGraph(dict)
    captured: dict = {}

    def node(state, config):  # noqa: ANN001
        captured.update(config.get("metadata") or {})
        return state

    builder.add_node("node", node)
    builder.add_edge("__start__", "node")
    graph = builder.compile().with_config({"metadata": {"user_id": "U1"}})
    await graph.ainvoke({}, {"metadata": {"correlation_id": "C1"}})
    assert captured.get("user_id") == "U1", (
        "bound metadata key was dropped by ensure_config overwrite"
    )
    assert captured.get("correlation_id") == "C1", "invoke-time key not present"


async def test_with_config_tags_preserved_on_invoke() -> None:
    """Tags bound via .with_config(...) must survive when invoke-time
    config supplies its own tags.

    Pre-fix: ensure_config overwrites the entire tags list.
    Post-fix: tags are concatenated (matching merge_configs behavior;
    no deduplication, no sorting).
    """
    builder = StateGraph(dict)
    captured: list = []

    def node(state, config):  # noqa: ANN001
        captured.extend(config.get("tags") or [])
        return state

    builder.add_node("node", node)
    builder.add_edge("__start__", "node")
    graph = builder.compile().with_config({"tags": ["bound"]})
    await graph.ainvoke({}, {"tags": ["invoke"]})
    assert "bound" in captured, "bound tag was dropped by ensure_config overwrite"
    assert "invoke" in captured, "invoke-time tag not present"
