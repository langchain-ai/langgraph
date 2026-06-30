"""Tests for the human-in-the-loop misconfiguration warnings introduced alongside ``interrupt()``.

When a developer wires ``interrupt_before`` / ``interrupt_after`` or the runtime ``interrupt()``
helper into a graph that has no checkpointer attached, the surrounding ``GraphInterrupt`` cannot
be persisted and the pause-and-resume contract silently breaks: the exception escapes to the
caller, the run terminates, and any pending value (often a destructive tool call) is abandoned.
These tests pin down two narrow warnings -- one at compile time, one at ``interrupt()`` call time --
so the misconfiguration is visible in logs instead of only surfacing when a production user hits it.
"""

from __future__ import annotations

import warnings

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class _State(TypedDict):
    foo: str


def _noop(_state: _State) -> dict:
    return {"foo": "x"}


def _interrupting_node(_state: _State) -> dict:
    interrupt("please confirm")
    return {"foo": "x"}


def _build_simple_graph(node):
    builder = StateGraph(_State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder


class TestCompileTimeWarning:
    def test_warns_when_static_interrupt_compiled_with_checkpointer_false(self) -> None:
        # Explicit ``checkpointer=False`` is the unambiguous "I don't want a checkpointer" signal.
        # When combined with a static interrupt the graph cannot pause/resume, so we surface the
        # misconfiguration at compile time.
        builder = _build_simple_graph(_noop)
        with pytest.warns(UserWarning, match="checkpointer=False"):
            builder.compile(checkpointer=False, interrupt_before=["node"])

    def test_warns_for_interrupt_after_with_checkpointer_false(self) -> None:
        builder = _build_simple_graph(_noop)
        with pytest.warns(UserWarning, match="checkpointer=False"):
            builder.compile(checkpointer=False, interrupt_after=["node"])

    def test_does_not_warn_when_checkpointer_false_without_interrupts(self) -> None:
        # ``checkpointer=False`` is a legitimate choice for fire-and-forget graphs. With no
        # interrupts configured, there is nothing to warn about.
        builder = _build_simple_graph(_noop)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            builder.compile(checkpointer=False)
        assert not any(
            issubclass(w.category, UserWarning) and "checkpointer" in str(w.message)
            for w in recorded
        )

    def test_does_not_warn_when_checkpointer_none_with_interrupts(self) -> None:
        # ``checkpointer=None`` can legitimately mean "inherit the parent checkpointer when used
        # as a subgraph". Warning at compile time for the ``None`` case would be noisy and would
        # fire on every nested compile. Runtime ``interrupt()`` covers the ``None`` case once the
        # effective config is known.
        builder = _build_simple_graph(_noop)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            builder.compile(interrupt_before=["node"])  # checkpointer defaults to None
        assert not any(
            issubclass(w.category, UserWarning)
            and "checkpointer=False" in str(w.message)
            for w in recorded
        )

    def test_does_not_warn_when_interrupts_paired_with_real_checkpointer(self) -> None:
        builder = _build_simple_graph(_noop)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            builder.compile(
                checkpointer=InMemorySaver(),
                interrupt_before=["node"],
            )
        assert not any(
            issubclass(w.category, UserWarning) and "checkpointer" in str(w.message)
            for w in recorded
        )


class TestRuntimeWarning:
    def test_warns_when_interrupt_called_without_checkpointer(self) -> None:
        # The dynamic ``interrupt()`` helper is the more common footgun: a node calls it without
        # the developer realising a checkpointer is required. Existing Pregel behaviour is
        # preserved -- the run surfaces the interrupt through the ``__interrupt__`` channel --
        # but we emit a warning first so the missing checkpointer is visible in logs and the
        # developer learns the resulting pause is not actually resumable.
        builder = _build_simple_graph(_interrupting_node)
        graph = builder.compile()  # no checkpointer

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            graph.invoke({"foo": ""})

        matching = [
            w
            for w in recorded
            if issubclass(w.category, UserWarning) and "checkpointer" in str(w.message)
        ]
        assert matching, "expected a UserWarning about the missing checkpointer"

    def test_does_not_warn_when_interrupt_called_with_checkpointer(self) -> None:
        builder = _build_simple_graph(_interrupting_node)
        graph = builder.compile(checkpointer=InMemorySaver())

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            # First invocation pauses; the test only cares that no checkpointer warning fires.
            graph.invoke({"foo": ""}, config={"configurable": {"thread_id": "t1"}})
        assert not any(
            issubclass(w.category, UserWarning) and "checkpointer" in str(w.message)
            for w in recorded
        )
