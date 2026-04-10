import pytest
from langchain_core.callbacks import AsyncCallbackManager

from langgraph._internal._config import ensure_config, get_async_callback_manager_for_config

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


def test_ensure_config_does_not_mutate_shared_metadata() -> None:
    original_config = {"metadata": {"ls_integration": "langchain_create_agent"}}
    new_config = {"configurable": {"thread_id": "thread-1"}}
    ensure_config(original_config, new_config)

    assert original_config == {"metadata": {"ls_integration": "langchain_create_agent"}}, (
        f"ensure_config mutated original config: {original_config}"
    )


def test_ensure_config_does_not_mutate_shared_metadata_multiple_invocations() -> None:
    original_config = {"metadata": {"ls_integration": "langchain_create_agent"}}

    ensure_config(original_config, {"configurable": {"thread_id": "thread-1"}})
    ensure_config(original_config, {"configurable": {"thread_id": "thread-2"}})

    assert original_config == {"metadata": {"ls_integration": "langchain_create_agent"}}, (
        f"ensure_config mutated original config after multiple invocations: {original_config}"
    )
