"""Pytest integration — generate test functions from a RegisteredCheckpointer."""

from __future__ import annotations

import inspect
from typing import Any

from langgraph.checkpoint.validation.capabilities import (
    Capability,
    DetectedCapabilities,
)
from langgraph.checkpoint.validation.initializer import RegisteredCheckpointer
from langgraph.checkpoint.validation.spec.test_copy_thread import ALL_COPY_THREAD_TESTS
from langgraph.checkpoint.validation.spec.test_delete_for_runs import (
    ALL_DELETE_FOR_RUNS_TESTS,
)
from langgraph.checkpoint.validation.spec.test_delete_thread import (
    ALL_DELETE_THREAD_TESTS,
)
from langgraph.checkpoint.validation.spec.test_get_tuple import ALL_GET_TUPLE_TESTS
from langgraph.checkpoint.validation.spec.test_list import ALL_LIST_TESTS
from langgraph.checkpoint.validation.spec.test_prune import ALL_PRUNE_TESTS
from langgraph.checkpoint.validation.spec.test_put import ALL_PUT_TESTS
from langgraph.checkpoint.validation.spec.test_put_writes import ALL_PUT_WRITES_TESTS

# Maps capability to its test list.
_TEST_LISTS: dict[Capability, list] = {
    Capability.PUT: ALL_PUT_TESTS,
    Capability.PUT_WRITES: ALL_PUT_WRITES_TESTS,
    Capability.GET_TUPLE: ALL_GET_TUPLE_TESTS,
    Capability.LIST: ALL_LIST_TESTS,
    Capability.DELETE_THREAD: ALL_DELETE_THREAD_TESTS,
    Capability.DELETE_FOR_RUNS: ALL_DELETE_FOR_RUNS_TESTS,
    Capability.COPY_THREAD: ALL_COPY_THREAD_TESTS,
    Capability.PRUNE: ALL_PRUNE_TESTS,
}


def conformance_tests(
    registered: RegisteredCheckpointer,
    *,
    capabilities: set[str] | None = None,
) -> None:
    """Generate pytest test functions in the caller's module namespace.

    Usage in a test file::

        from langgraph.checkpoint.validation.pytest import conformance_tests
        from my_checkpointer_test import my_checkpointer

        conformance_tests(my_checkpointer)

    This will inject async test functions named like
    ``test_put__test_put_returns_config`` into the calling module, which
    pytest will discover normally.
    """
    caller_frame = inspect.stack()[1]
    caller_module = caller_frame[0].f_globals

    caps_to_test: set[Capability]
    if capabilities is not None:
        caps_to_test = {Capability(c) for c in capabilities}
    else:
        caps_to_test = set(Capability)

    # Inject a module-scoped fixture that enters the lifespan once.
    import pytest

    fixture_name = f"_lifespan_{registered.name}"

    @pytest.fixture(scope="module")
    async def _lifespan_fixture():  # type: ignore[misc]
        async with registered.enter_lifespan():
            yield

    _lifespan_fixture.__name__ = fixture_name
    _lifespan_fixture.__qualname__ = fixture_name
    caller_module[fixture_name] = _lifespan_fixture

    for cap in Capability:
        if cap not in caps_to_test:
            continue
        if cap.value in registered.skip_capabilities:
            continue

        tests = _TEST_LISTS.get(cap, [])
        for test_fn in tests:

            async def _make_test(
                _fn: Any = test_fn,
                _cap: Capability = cap,
                **kwargs: Any,
            ) -> None:
                async with registered.create() as saver:
                    detected = DetectedCapabilities.from_instance(saver)
                    if _cap not in detected.detected:
                        pytest.skip(f"{_cap.value} not detected on {registered.name}")
                    await _fn(saver)

            # Mark the test as depending on the lifespan fixture.
            _make_test = pytest.mark.usefixtures(fixture_name)(_make_test)

            name = f"test_{cap.value}__{test_fn.__name__}"
            _make_test.__name__ = name
            _make_test.__qualname__ = name
            caller_module[name] = _make_test
