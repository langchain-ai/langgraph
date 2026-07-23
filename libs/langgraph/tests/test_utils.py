import functools
import sys
import uuid
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Literal,
    Optional,
    TypeVar,
    Union,
)
from unittest.mock import MagicMock, patch

import langsmith
import pytest
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from typing_extensions import NotRequired, Required, TypedDict

from langgraph._internal._config import (
    _is_not_empty,
    _merge_callbacks,
    ensure_config,
    get_callback_manager_for_config,
    merge_configs,
)
from langgraph._internal._fields import (
    _is_optional_type,
    get_enhanced_type_hints,
    get_field_default,
)
from langgraph._internal._runnable import is_async_callable, is_async_generator
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

# ruff: noqa: UP045, UP007

pytestmark = pytest.mark.anyio


def test_is_async() -> None:
    async def func() -> None:
        pass

    assert is_async_callable(func)
    wrapped_func = functools.wraps(func)(func)
    assert is_async_callable(wrapped_func)

    def sync_func() -> None:
        pass

    assert not is_async_callable(sync_func)
    wrapped_sync_func = functools.wraps(sync_func)(sync_func)
    assert not is_async_callable(wrapped_sync_func)

    class AsyncFuncCallable:
        async def __call__(self) -> None:
            pass

    runnable = AsyncFuncCallable()
    assert is_async_callable(runnable)
    wrapped_runnable = functools.wraps(runnable)(runnable)
    assert is_async_callable(wrapped_runnable)

    class SyncFuncCallable:
        def __call__(self) -> None:
            pass

    sync_runnable = SyncFuncCallable()
    assert not is_async_callable(sync_runnable)
    wrapped_sync_runnable = functools.wraps(sync_runnable)(sync_runnable)
    assert not is_async_callable(wrapped_sync_runnable)


def test_is_generator() -> None:
    async def gen():
        yield

    assert is_async_generator(gen)

    wrapped_gen = functools.wraps(gen)(gen)
    assert is_async_generator(wrapped_gen)

    def sync_gen():
        yield

    assert not is_async_generator(sync_gen)
    wrapped_sync_gen = functools.wraps(sync_gen)(sync_gen)
    assert not is_async_generator(wrapped_sync_gen)

    class AsyncGenCallable:
        async def __call__(self):
            yield

    runnable = AsyncGenCallable()
    assert is_async_generator(runnable)
    wrapped_runnable = functools.wraps(runnable)(runnable)
    assert is_async_generator(wrapped_runnable)

    class SyncGenCallable:
        def __call__(self):
            yield

    sync_runnable = SyncGenCallable()
    assert not is_async_generator(sync_runnable)
    wrapped_sync_runnable = functools.wraps(sync_runnable)(sync_runnable)
    assert not is_async_generator(wrapped_sync_runnable)


@pytest.fixture
def rt_graph() -> CompiledStateGraph:
    class State(TypedDict):
        foo: int
        node_run_id: int

    def node(_: State):
        from langsmith import get_current_run_tree  # type: ignore

        return {"node_run_id": get_current_run_tree().id}  # type: ignore

    graph = StateGraph(State)
    graph.add_node(node)
    graph.set_entry_point("node")
    graph.add_edge("node", END)
    return graph.compile()


def test_runnable_callable_tracing_nested(rt_graph: CompiledStateGraph) -> None:
    with patch("langsmith.client.Client", spec=langsmith.Client) as mock_client:
        with patch("langchain_core.tracers.langchain.get_client") as mock_get_client:
            mock_get_client.return_value = mock_client
            with langsmith.tracing_context(enabled=True):
                res = rt_graph.invoke({"foo": 1})
    assert isinstance(res["node_run_id"], uuid.UUID)


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
async def test_runnable_callable_tracing_nested_async(
    rt_graph: CompiledStateGraph,
) -> None:
    with patch("langsmith.client.Client", spec=langsmith.Client) as mock_client:
        with patch("langchain_core.tracers.langchain.get_client") as mock_get_client:
            mock_get_client.return_value = mock_client
            with langsmith.tracing_context(enabled=True):
                res = await rt_graph.ainvoke({"foo": 1})
    assert isinstance(res["node_run_id"], uuid.UUID)


def test_is_optional_type():
    assert _is_optional_type(None)
    assert not _is_optional_type(type(None))
    assert _is_optional_type(Optional[list])
    assert not _is_optional_type(int)
    assert _is_optional_type(Optional[Literal[1, 2, 3]])
    assert not _is_optional_type(Literal[1, 2, 3])
    assert _is_optional_type(Optional[list[int]])
    assert _is_optional_type(Optional[dict[str, int]])
    assert not _is_optional_type(list[int | None])
    assert _is_optional_type(Union[str | None, int | None])
    assert _is_optional_type(Union[str | None | int | None, float | None | dict | None])
    assert not _is_optional_type(Union[str | int, float | dict])

    assert _is_optional_type(Union[int, None])
    assert _is_optional_type(Union[str, None, int])
    assert _is_optional_type(Union[None, str, int])
    assert not _is_optional_type(Union[int, str])

    assert not _is_optional_type(Any)  # Do we actually want this?
    assert _is_optional_type(Optional[Any])

    class MyClass:
        pass

    assert _is_optional_type(Optional[MyClass])
    assert not _is_optional_type(MyClass)
    assert _is_optional_type(Optional[ForwardRef("MyClass")])
    assert not _is_optional_type(ForwardRef("MyClass"))

    assert _is_optional_type(Optional[list[int] | dict[str, int | None]])
    assert not _is_optional_type(Union[list[int], dict[str, int | None]])

    assert _is_optional_type(Optional[Callable[[int], str]])
    assert not _is_optional_type(Callable[[int], str | None])

    T = TypeVar("T")
    assert _is_optional_type(Optional[T])
    assert not _is_optional_type(T)

    U = TypeVar("U", bound=T | None)  # type: ignore
    assert _is_optional_type(U)


def test_is_required():
    class MyBaseTypedDict(TypedDict):
        val_1: Required[str | None]
        val_2: Required[str]
        val_3: NotRequired[str]
        val_4: NotRequired[str | None]
        val_5: Annotated[NotRequired[int], "foo"]
        val_6: NotRequired[Annotated[int, "foo"]]
        val_7: Annotated[Required[int], "foo"]
        val_8: Required[Annotated[int, "foo"]]
        val_9: str | None
        val_10: str

    annos = MyBaseTypedDict.__annotations__
    assert get_field_default("val_1", annos["val_1"], MyBaseTypedDict) == ...
    assert get_field_default("val_2", annos["val_2"], MyBaseTypedDict) == ...
    assert get_field_default("val_3", annos["val_3"], MyBaseTypedDict) is None
    assert get_field_default("val_4", annos["val_4"], MyBaseTypedDict) is None
    # See https://peps.python.org/pep-0655/#interaction-with-annotated
    assert get_field_default("val_5", annos["val_5"], MyBaseTypedDict) is None
    assert get_field_default("val_6", annos["val_6"], MyBaseTypedDict) is None
    assert get_field_default("val_7", annos["val_7"], MyBaseTypedDict) == ...
    assert get_field_default("val_8", annos["val_8"], MyBaseTypedDict) == ...
    assert get_field_default("val_9", annos["val_9"], MyBaseTypedDict) is None
    assert get_field_default("val_10", annos["val_10"], MyBaseTypedDict) == ...

    class MyChildDict(MyBaseTypedDict):
        val_11: int
        val_11b: int | None
        val_11c: int | None | str

    class MyGrandChildDict(MyChildDict, total=False):
        val_12: int
        val_13: Required[str]

    cannos = MyChildDict.__annotations__
    gcannos = MyGrandChildDict.__annotations__
    assert get_field_default("val_11", cannos["val_11"], MyChildDict) == ...
    assert get_field_default("val_11b", cannos["val_11b"], MyChildDict) is None
    assert get_field_default("val_11c", cannos["val_11c"], MyChildDict) is None
    assert get_field_default("val_12", gcannos["val_12"], MyGrandChildDict) is None
    assert get_field_default("val_9", gcannos["val_9"], MyGrandChildDict) is None
    assert get_field_default("val_13", gcannos["val_13"], MyGrandChildDict) == ...


def test_enhanced_type_hints() -> None:
    from dataclasses import dataclass
    from typing import Annotated

    from pydantic import BaseModel, Field

    class MyTypedDict(TypedDict):
        val_1: str
        val_2: int = 42
        val_3: str = "default"

    hints = list(get_enhanced_type_hints(MyTypedDict))
    assert len(hints) == 3
    assert hints[0] == ("val_1", str, None, None)
    assert hints[1] == ("val_2", int, 42, None)
    assert hints[2] == ("val_3", str, "default", None)

    @dataclass
    class MyDataclass:
        val_1: str
        val_2: int = 42
        val_3: str = "default"

    hints = list(get_enhanced_type_hints(MyDataclass))
    assert len(hints) == 3
    assert hints[0] == ("val_1", str, None, None)
    assert hints[1] == ("val_2", int, 42, None)
    assert hints[2] == ("val_3", str, "default", None)

    class MyPydanticModel(BaseModel):
        val_1: str
        val_2: int = 42
        val_3: str = Field(default="default", description="A description")

    hints = list(get_enhanced_type_hints(MyPydanticModel))
    assert len(hints) == 3
    assert hints[0] == ("val_1", str, None, None)
    assert hints[1] == ("val_2", int, 42, None)
    assert hints[2] == ("val_3", str, "default", "A description")

    class MyPydanticModelWithAnnotated(BaseModel):
        val_1: Annotated[str, Field(description="A description")]
        val_2: Annotated[int, Field(default=42)]
        val_3: Annotated[
            str, Field(default="default", description="Another description")
        ]

    hints = list(get_enhanced_type_hints(MyPydanticModelWithAnnotated))
    assert len(hints) == 3
    assert hints[0] == ("val_1", str, None, "A description")
    assert hints[1] == ("val_2", int, 42, None)
    assert hints[2] == ("val_3", str, "default", "Another description")


def test_is_not_empty() -> None:
    assert _is_not_empty("foo")
    assert _is_not_empty("")
    assert _is_not_empty(1)
    assert _is_not_empty(0)
    assert not _is_not_empty(None)
    assert not _is_not_empty([])
    assert not _is_not_empty(())
    assert not _is_not_empty({})


def test_configurable_metadata() -> None:
    config = {
        "configurable": {
            "a-key": "foo",
            "somesecretval": "bar",
            "sometoken": "thetoken",
            "__dontinclude": "bar",
            "includeme": "hi",
            "andme": 42,
            "nested": {"foo": "bar"},
            "nooverride": -2,
            "thread_id": "th-123",
            "checkpoint_id": "ckpt-1",
            "checkpoint_ns": "ns-1",
            "task_id": "task-1",
            "run_id": "run-456",
            "assistant_id": "asst-789",
            "graph_id": "graph-0",
            "model": "gpt-4o",
            "user_id": "uid-1",
            "cron_id": "cron-1",
            "langgraph_auth_user_id": "user-1",
        },
        "metadata": {"nooverride": 18},
    }
    merged = ensure_config(config)
    metadata = merged["metadata"]
    assert set(metadata) == {
        "nooverride",
        "assistant_id",
        "thread_id",
        "checkpoint_id",
        "run_id",
        "graph_id",
        "checkpoint_ns",
        "task_id",
    }
    assert metadata["nooverride"] == 18


def test_callback_manager_copies_whitelisted_configurable_ids_to_metadata() -> None:
    config = {
        "configurable": {
            "thread_id": "th-123",
            "checkpoint_id": "ckpt-1",
            "checkpoint_ns": "ns-1",
            "task_id": "task-1",
            "run_id": "run-456",
            "assistant_id": "asst-789",
            "graph_id": "graph-0",
            "model": "gpt-4o",
            "user_id": "uid-1",
            "cron_id": "cron-1",
            "langgraph_auth_user_id": "user-1",
        },
        "metadata": {
            "thread_id": "from-metadata",
            "nooverride": 18,
        },
    }
    manager = ensure_config(config)
    callback_manager = get_callback_manager_for_config(manager)
    assert callback_manager.metadata == {
        "thread_id": "from-metadata",
        "nooverride": 18,
        "checkpoint_id": "ckpt-1",
        "checkpoint_ns": "ns-1",
        "task_id": "task-1",
        "run_id": "run-456",
        "assistant_id": "asst-789",
        "graph_id": "graph-0",
    }


def test_callback_manager_copies_configurable_ids_to_tracing_metadata() -> None:
    tracer = LangChainTracer(client=MagicMock())
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "th-123",
            "checkpoint_id": "ckpt-1",
            "checkpoint_ns": "ns-1",
            "task_id": "task-1",
            "run_id": "run-456",
            "assistant_id": "asst-789",
            "graph_id": "graph-0",
            "model": "gpt-4o",
            "user_id": "uid-1",
            "cron_id": "cron-1",
            "langgraph_auth_user_id": "user-1",
            "includeme": "hi",
            "andme": 42,
            "__dontinclude": "bar",
            "some_api_key": "secret",
            "custom_setting": {"nested": True},
        },
        "metadata": {
            "thread_id": "from-metadata",
            "user_id": "from-metadata-user",
            "includeme": "from-metadata",
        },
        "callbacks": [tracer],
    }

    manager = ensure_config(config)
    callback_manager = get_callback_manager_for_config(manager)
    handlers = callback_manager.handlers
    tracers = [handler for handler in handlers if isinstance(handler, LangChainTracer)]
    assert len(tracers) == 1
    tracer = tracers[0]
    assert tracer.tracing_metadata == {
        "checkpoint_id": "ckpt-1",
        "checkpoint_ns": "ns-1",
        "task_id": "task-1",
        "run_id": "run-456",
        "assistant_id": "asst-789",
        "graph_id": "graph-0",
        "model": "gpt-4o",
        "cron_id": "cron-1",
        "andme": 42,
        "includeme": "hi",
        "thread_id": "th-123",
        "user_id": "uid-1",
    }


class _TrackingCB(BaseCallbackHandler):
    """Minimal callback handler used only as a sentinel for merge tests."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TrackingCB) and self.tag == other.tag

    def __hash__(self) -> int:
        return hash(self.tag)


def test_merge_callbacks_none_base_list_new() -> None:
    cb = _TrackingCB("a")
    merged = _merge_callbacks(None, [cb])
    assert merged == [cb]


def test_merge_callbacks_list_base_list_new() -> None:
    a, b = _TrackingCB("a"), _TrackingCB("b")
    merged = _merge_callbacks([a], [b])
    assert merged == [a, b]


def test_merge_callbacks_list_base_manager_new() -> None:
    a = _TrackingCB("a")
    mgr = CallbackManager(handlers=[_TrackingCB("b")])
    merged = _merge_callbacks([a], mgr)
    assert isinstance(merged, CallbackManager)
    assert _TrackingCB("a") in merged.handlers
    assert _TrackingCB("b") in merged.handlers


def test_merge_callbacks_manager_base_list_new() -> None:
    mgr = CallbackManager(handlers=[_TrackingCB("a")])
    b = _TrackingCB("b")
    merged = _merge_callbacks(mgr, [b])
    assert isinstance(merged, CallbackManager)
    assert _TrackingCB("a") in merged.handlers
    assert _TrackingCB("b") in merged.handlers


def test_merge_callbacks_manager_base_manager_new() -> None:
    mgr_a = CallbackManager(handlers=[_TrackingCB("a")])
    mgr_b = CallbackManager(handlers=[_TrackingCB("b")])
    merged = _merge_callbacks(mgr_a, mgr_b)
    assert isinstance(merged, CallbackManager)
    assert _TrackingCB("a") in merged.handlers
    assert _TrackingCB("b") in merged.handlers


def test_merge_callbacks_none_base_none_new() -> None:
    merged = _merge_callbacks(None, None)
    assert merged is None


def test_ensure_config_merges_configurable_across_configs() -> None:
    a = {"configurable": {"ls_agent_type": "root"}}
    b = {"configurable": {"thread_id": "T1"}}
    merged = ensure_config(a, b)
    assert merged["configurable"]["ls_agent_type"] == "root"
    assert merged["configurable"]["thread_id"] == "T1"


def test_ensure_config_configurable_later_wins_per_key() -> None:
    a = {"configurable": {"shared": "from_a", "only_a": "A"}}
    b = {"configurable": {"shared": "from_b", "only_b": "B"}}
    merged = ensure_config(a, b)
    assert merged["configurable"]["shared"] == "from_b"  # later wins per key
    assert merged["configurable"]["only_a"] == "A"
    assert merged["configurable"]["only_b"] == "B"


def test_ensure_config_explicit_configurable_replaces_ambient() -> None:
    # An explicit checkpoint coordinate (here a new thread_id) starts a fresh
    # lineage and drops the ambient run context (e.g. a parent task's
    # checkpoint_ns), so a child graph does not inherit it.
    from langchain_core.runnables.config import var_child_runnable_config

    token = var_child_runnable_config.set(
        {"configurable": {"checkpoint_ns": "p:parent-task", "checkpoint_id": "cid"}}
    )
    try:
        merged = ensure_config({"configurable": {"thread_id": "child"}})
    finally:
        var_child_runnable_config.reset(token)
    assert merged["configurable"]["thread_id"] == "child"
    assert "checkpoint_ns" not in merged["configurable"]
    assert "checkpoint_id" not in merged["configurable"]


def test_ensure_config_ambient_inherited_when_no_explicit_configurable() -> None:
    # With no explicit configurable, the ambient run context is inherited
    # unchanged (stateless subgraph / interrupt-resume pattern).
    from langchain_core.runnables.config import var_child_runnable_config

    token = var_child_runnable_config.set(
        {"configurable": {"checkpoint_ns": "p:parent-task"}}
    )
    try:
        merged = ensure_config({"tags": ["t"]})
    finally:
        var_child_runnable_config.reset(token)
    assert merged["configurable"]["checkpoint_ns"] == "p:parent-task"


def test_ensure_config_explicit_configurables_still_merge_over_ambient() -> None:
    # A new thread_id drops the ambient, but explicit configs still shallow-merge
    # among themselves, so a with_config(...) value (ls_agent_type) survives
    # alongside an invoke-time thread_id.
    from langchain_core.runnables.config import var_child_runnable_config

    token = var_child_runnable_config.set(
        {"configurable": {"checkpoint_ns": "p:parent-task"}}
    )
    try:
        merged = ensure_config(
            {"configurable": {"ls_agent_type": "root"}},
            {"configurable": {"thread_id": "child"}},
        )
    finally:
        var_child_runnable_config.reset(token)
    assert merged["configurable"]["ls_agent_type"] == "root"
    assert merged["configurable"]["thread_id"] == "child"
    assert "checkpoint_ns" not in merged["configurable"]


def test_ensure_config_non_coordinate_config_keeps_ambient_checkpoint_ns() -> None:
    # A nested subagent is invoked with a non-coordinate configurable key
    # (ls_agent_type) and no thread_id; it must keep the inherited checkpoint_ns
    # so it stays a discoverable child of the parent run (deepagents `task` tool).
    from langchain_core.runnables.config import var_child_runnable_config

    token = var_child_runnable_config.set(
        {"configurable": {"thread_id": "parent", "checkpoint_ns": "p:parent-task"}}
    )
    try:
        merged = ensure_config({"configurable": {"ls_agent_type": "subagent"}})
    finally:
        var_child_runnable_config.reset(token)
    assert merged["configurable"]["ls_agent_type"] == "subagent"
    assert merged["configurable"]["checkpoint_ns"] == "p:parent-task"
    assert merged["configurable"]["thread_id"] == "parent"


def test_ensure_config_same_thread_id_still_clears_ambient() -> None:
    # A child that reuses the parent's thread_id is still addressing its own root
    # namespace on that thread, so the parent task's checkpoint_ns must not leak
    # in; otherwise the child writes state that get_state cannot read back.
    from langchain_core.runnables.config import var_child_runnable_config

    token = var_child_runnable_config.set(
        {"configurable": {"thread_id": "shared", "checkpoint_ns": "p:parent-task"}}
    )
    try:
        merged = ensure_config({"configurable": {"thread_id": "shared"}})
    finally:
        var_child_runnable_config.reset(token)
    assert merged["configurable"]["thread_id"] == "shared"
    assert "checkpoint_ns" not in merged["configurable"]


def test_ensure_config_merges_metadata_across_configs() -> None:
    a = {"metadata": {"user_id": "U1"}}
    b = {"metadata": {"correlation_id": "C1"}}
    merged = ensure_config(a, b)
    assert merged["metadata"]["user_id"] == "U1"
    assert merged["metadata"]["correlation_id"] == "C1"


def test_ensure_config_metadata_later_wins_per_key() -> None:
    a = {"metadata": {"shared": "from_a"}}
    b = {"metadata": {"shared": "from_b"}}
    merged = ensure_config(a, b)
    assert merged["metadata"]["shared"] == "from_b"


def test_merge_configs_merges_metadata_lc_versions() -> None:
    a = {
        "metadata": {
            "lc_versions": {"langgraph": "1.2.4"},
            "lc_agent_name": "agent",
        }
    }
    b = {"metadata": {"lc_versions": {"langchain-core": "1.2.0"}}}
    merged = merge_configs(a, b)
    assert merged["metadata"]["lc_versions"] == {
        "langgraph": "1.2.4",
        "langchain-core": "1.2.0",
    }
    assert merged["metadata"]["lc_agent_name"] == "agent"


def test_ensure_config_merges_metadata_lc_versions() -> None:
    a = {
        "metadata": {
            "lc_versions": {"langgraph": "1.2.4"},
            "lc_agent_name": "agent",
        }
    }
    b = {"metadata": {"lc_versions": {"langchain-core": "1.2.0"}}}
    merged = ensure_config(a, b)
    assert merged["metadata"]["lc_versions"] == {
        "langgraph": "1.2.4",
        "langchain-core": "1.2.0",
    }
    assert merged["metadata"]["lc_agent_name"] == "agent"


@pytest.mark.parametrize("merge", [merge_configs, ensure_config])
def test_metadata_lc_versions_later_values_win_without_recursive_merge(
    merge: Callable[..., RunnableConfig],
) -> None:
    a = {
        "metadata": {
            "lc_versions": {
                "langgraph": "1.2.4",
                "nested": {"only_a": "A", "shared": "from_a"},
            }
        }
    }
    b = {
        "metadata": {
            "lc_versions": {
                "langchain-core": "1.2.0",
                "nested": {"only_b": "B", "shared": "from_b"},
            }
        }
    }
    merged = merge(a, b)
    assert merged["metadata"]["lc_versions"] == {
        "langgraph": "1.2.4",
        "langchain-core": "1.2.0",
        "nested": {"only_b": "B", "shared": "from_b"},
    }


@pytest.mark.parametrize("merge", [merge_configs, ensure_config])
def test_metadata_nested_mappings_other_than_lc_versions_are_replaced(
    merge: Callable[..., RunnableConfig],
) -> None:
    a = {"metadata": {"versions": {"langgraph": "1.2.4"}, "mode": "bound"}}
    b = {
        "metadata": {
            "versions": {"langchain-core": "1.2.0"},
            "mode": {"source": "runtime"},
        }
    }
    merged = merge(a, b)
    assert merged["metadata"]["versions"] == {"langchain-core": "1.2.0"}
    assert merged["metadata"]["mode"] == {"source": "runtime"}


@pytest.mark.parametrize("merge", [merge_configs, ensure_config])
def test_metadata_non_mapping_values_later_wins(
    merge: Callable[..., RunnableConfig],
) -> None:
    a = {"metadata": {"lc_versions": {"langgraph": "1.2.4"}, "mode": "bound"}}
    b = {"metadata": {"lc_versions": "runtime", "mode": {"source": "runtime"}}}
    merged = merge(a, b)
    assert merged["metadata"]["lc_versions"] == "runtime"
    assert merged["metadata"]["mode"] == {"source": "runtime"}


@pytest.mark.parametrize("merge", [merge_configs, ensure_config])
def test_metadata_lc_versions_merge_does_not_mutate_inputs(
    merge: Callable[..., RunnableConfig],
) -> None:
    a_versions = {"langgraph": "1.2.4"}
    b_versions = {"langchain-core": "1.2.0"}
    a = {"metadata": {"lc_versions": a_versions}}
    b = {"metadata": {"lc_versions": b_versions}}
    merged = merge(a, b)
    assert a == {"metadata": {"lc_versions": {"langgraph": "1.2.4"}}}
    assert b == {"metadata": {"lc_versions": {"langchain-core": "1.2.0"}}}
    assert merged["metadata"]["lc_versions"] is not a_versions
    assert merged["metadata"]["lc_versions"] is not b_versions

    merged["metadata"]["lc_versions"]["langgraph"] = "changed"
    assert a_versions == {"langgraph": "1.2.4"}
    assert b_versions == {"langchain-core": "1.2.0"}


@pytest.mark.parametrize("merge", [merge_configs, ensure_config])
def test_metadata_single_sided_mapping_values_are_copied(
    merge: Callable[..., RunnableConfig],
) -> None:
    base_versions = {"langgraph": "1.2.4"}
    new_versions = {"langchain-core": "1.2.0"}

    merged_base_only = merge({"metadata": {"lc_versions": base_versions}})
    merged_new_only = merge(
        {"metadata": {}},
        {"metadata": {"lc_versions": new_versions}},
    )

    assert merged_base_only["metadata"]["lc_versions"] is not base_versions
    assert merged_new_only["metadata"]["lc_versions"] is not new_versions

    merged_base_only["metadata"]["lc_versions"]["langgraph"] = "changed"
    merged_new_only["metadata"]["lc_versions"]["langchain-core"] = "changed"
    assert base_versions == {"langgraph": "1.2.4"}
    assert new_versions == {"langchain-core": "1.2.0"}


@pytest.mark.parametrize("merge", [merge_configs, ensure_config])
def test_metadata_empty_incoming_preserves_base_lc_versions(
    merge: Callable[..., RunnableConfig],
) -> None:
    a = {"metadata": {"lc_versions": {"langgraph": "1.2.4"}}}
    b = {"metadata": {"lc_versions": {}}}
    merged = merge(a, b)
    assert merged["metadata"]["lc_versions"] == {"langgraph": "1.2.4"}


@pytest.mark.parametrize("merge", [merge_configs, ensure_config])
def test_metadata_lc_versions_accumulate_across_more_than_two_configs(
    merge: Callable[..., RunnableConfig],
) -> None:
    a = {"metadata": {"lc_versions": {"langgraph": "1.2.4"}}}
    b = {"metadata": {"lc_versions": {"langchain-core": "1.2.0"}}}
    c = {"metadata": {"lc_versions": {"langchain": "1.1.0"}}}
    merged = merge(a, b, c)
    assert merged["metadata"]["lc_versions"] == {
        "langgraph": "1.2.4",
        "langchain-core": "1.2.0",
        "langchain": "1.1.0",
    }


def test_ensure_config_merges_tags_across_configs() -> None:
    a = {"tags": ["alpha"]}
    b = {"tags": ["beta"]}
    merged = ensure_config(a, b)
    assert merged["tags"] == ["alpha", "beta"]


def test_ensure_config_tags_concat_preserves_order_and_duplicates() -> None:
    # Plain concat (matches merge_configs in this file — no dedup, no sort).
    a = {"tags": ["shared", "alpha"]}
    b = {"tags": ["shared", "beta"]}
    merged = ensure_config(a, b)
    assert merged["tags"] == ["shared", "alpha", "shared", "beta"]


def test_ensure_config_merges_callbacks_across_configs() -> None:
    a_cb = _TrackingCB("a")
    b_cb = _TrackingCB("b")
    merged = ensure_config({"callbacks": [a_cb]}, {"callbacks": [b_cb]})
    assert merged["callbacks"] == [a_cb, b_cb]


def test_ensure_config_none_inputs_ignored() -> None:
    # mixed with None should not raise
    merged = ensure_config(None, {"tags": ["t"]}, None)
    assert merged["tags"] == ["t"]


def test_ensure_config_empty_inputs() -> None:
    # everything empty -> defaults
    merged = ensure_config()
    assert merged["tags"] == []
    assert merged["configurable"] == {}
    assert merged["callbacks"] is None
