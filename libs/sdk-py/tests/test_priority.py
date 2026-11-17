import dataclasses

import pytest

import langgraph_sdk
from langgraph_sdk import Priority, PriorityContext, PrioritySettings
from langgraph_sdk.encryption import DuplicateHandlerError, LangGraphBetaWarning
from langgraph_sdk.priority import _warn_priority_beta


class _User:
    is_authenticated = True
    display_name = "user-1"
    identity = "user-1"
    permissions: tuple[str, ...] = ()

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def __iter__(self):
        return iter(("identity", "display_name", "is_authenticated", "permissions"))


def _context(thread_id: str | None) -> PriorityContext:
    return PriorityContext(
        user=_User(),
        permissions=[],
        thread_id=thread_id,
        thread_metadata={},
        assistant_id="assistant-1",
        graph_id="agent",
        assistant_metadata={"tier": "gold"},
        run_metadata={},
        cron_id=None,
    )


async def _no_params() -> PrioritySettings | None:
    return None


async def _two_params(_ctx: PriorityContext, _extra: object) -> PrioritySettings | None:
    return None


def test_priority_and_its_types_are_exported():
    assert {"Priority", "PriorityContext", "PrioritySettings"} <= set(
        langgraph_sdk.__all__
    )
    assert Priority.types.PriorityContext is PriorityContext
    assert Priority.types.PrioritySettings is PrioritySettings


def test_construction_warns_that_priority_is_beta():
    # The warning fires once per process, so an earlier test may have used it up.
    _warn_priority_beta.cache_clear()
    with pytest.warns(LangGraphBetaWarning, match="Priority API is in beta"):
        Priority()


def test_assign_registers_and_returns_the_function():
    priority = Priority()

    async def assign(_ctx: PriorityContext) -> PrioritySettings | None:
        return {"priority": 3, "fairness_group": "org-42"}

    assert priority.assign(assign) is assign
    assert priority._assign_handler is assign


def test_assign_rejects_a_sync_function():
    priority = Priority()

    def assign(_ctx: PriorityContext) -> PrioritySettings | None:
        return None

    with pytest.raises(TypeError, match="must be an async function"):
        priority.assign(assign)  # ty: ignore[invalid-argument-type]
    assert priority._assign_handler is None


@pytest.mark.parametrize("fn", [_no_params, _two_params])
def test_assign_rejects_the_wrong_number_of_parameters(fn):
    priority = Priority()

    with pytest.raises(TypeError, match="must accept exactly 1 parameter"):
        priority.assign(fn)
    assert priority._assign_handler is None


def test_assign_rejects_a_second_function():
    priority = Priority()

    @priority.assign
    async def first(_ctx: PriorityContext) -> PrioritySettings | None:
        return None

    with pytest.raises(DuplicateHandlerError):

        @priority.assign
        async def second(_ctx: PriorityContext) -> PrioritySettings | None:
            return None

    assert priority._assign_handler is first


def test_context_accepts_no_thread_id():
    # `make lint` type-checks this call too, so `thread_id` must accept None there.
    assert _context(thread_id=None).thread_id is None


def test_context_is_frozen():
    ctx = _context(thread_id="thread-1")

    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.thread_id = "thread-2"  # ty: ignore[invalid-assignment]


def test_every_settings_key_is_optional():
    assert PrioritySettings.__optional_keys__ == {"priority", "fairness_group"}
    assert PrioritySettings.__required_keys__ == frozenset()
