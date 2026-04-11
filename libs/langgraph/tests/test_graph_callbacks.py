from __future__ import annotations

import sys
from typing import Any

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.callbacks import (
    GraphCallbackHandler,
    GraphInterruptEvent,
    GraphResumeEvent,
)
from langgraph.graph import START, StateGraph
from langgraph.types import Command, Interrupt, interrupt

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


class _GraphEventHandler(GraphCallbackHandler):
    def __init__(self) -> None:
        self.interrupt_events: list[GraphInterruptEvent] = []
        self.resume_events: list[GraphResumeEvent] = []

    def on_interrupt(self, event: GraphInterruptEvent) -> Any:
        self.interrupt_events.append(event)

    def on_resume(self, event: GraphResumeEvent) -> Any:
        self.resume_events.append(event)


class _LangChainCustomEventHandler(BaseCallbackHandler):
    run_inline = True

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_custom_event(self, name: str, data: Any, **kwargs: Any) -> Any:
        self.events.append(name)


class _RaisingGraphEventHandler(GraphCallbackHandler):
    def __init__(
        self,
        *,
        raise_on_interrupt: bool = False,
        raise_on_resume: bool = False,
        raise_error: bool = False,
    ) -> None:
        self.raise_on_interrupt = raise_on_interrupt
        self.raise_on_resume = raise_on_resume
        self.raise_error = raise_error

    def on_interrupt(self, event: GraphInterruptEvent) -> Any:
        if self.raise_on_interrupt:
            raise ValueError("boom-interrupt")

    def on_resume(self, event: GraphResumeEvent) -> Any:
        if self.raise_on_resume:
            raise ValueError("boom-resume")


class _AsyncRaisingGraphEventHandler(GraphCallbackHandler):
    def __init__(
        self,
        *,
        raise_on_interrupt: bool = False,
        raise_on_resume: bool = False,
        raise_error: bool = False,
    ) -> None:
        self.raise_on_interrupt = raise_on_interrupt
        self.raise_on_resume = raise_on_resume
        self.raise_error = raise_error

    async def on_interrupt(self, event: GraphInterruptEvent) -> Any:
        if self.raise_on_interrupt:
            raise ValueError("boom-interrupt")

    async def on_resume(self, event: GraphResumeEvent) -> Any:
        if self.raise_on_resume:
            raise ValueError("boom-resume")


class _State(TypedDict):
    answer: str | None


def _build_interrupt_graph() -> Any:
    def ask(state: _State) -> _State:
        answer = interrupt("Provide value")
        return {"answer": answer}

    builder = StateGraph(_State)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    return builder.compile(checkpointer=InMemorySaver())


def test_graph_callbacks_interrupt_and_resume_sync() -> None:
    graph = _build_interrupt_graph()
    handler = _GraphEventHandler()
    langchain_handler = _LangChainCustomEventHandler()
    config = {
        "configurable": {"thread_id": "graph-callback-sync"},
        "callbacks": [langchain_handler, handler],
    }

    first = graph.invoke({"answer": None}, config)
    assert "__interrupt__" in first

    assert len(handler.interrupt_events) == 1
    assert handler.interrupt_events[0].interrupts
    assert isinstance(handler.interrupt_events[0].interrupts[0], Interrupt)
    assert handler.interrupt_events[0].checkpoint_ns == ()
    assert langchain_handler.events == []

    handler.resume_events.clear()
    resumed = graph.invoke(Command(resume="done"), config)
    assert resumed == {"answer": "done"}

    assert len(handler.resume_events) == 1
    assert handler.resume_events[0].checkpoint_ns == ()
    assert langchain_handler.events == []


@pytest.mark.anyio
@NEEDS_CONTEXTVARS
async def test_graph_callbacks_interrupt_and_resume_async() -> None:
    graph = _build_interrupt_graph()
    handler = _GraphEventHandler()
    langchain_handler = _LangChainCustomEventHandler()
    config = {
        "configurable": {"thread_id": "graph-callback-async"},
        "callbacks": [langchain_handler, handler],
    }

    first = await graph.ainvoke({"answer": None}, config)
    assert "__interrupt__" in first

    assert len(handler.interrupt_events) == 1
    assert handler.interrupt_events[0].interrupts
    assert isinstance(handler.interrupt_events[0].interrupts[0], Interrupt)
    assert handler.interrupt_events[0].checkpoint_ns == ()
    assert langchain_handler.events == []

    handler.resume_events.clear()
    resumed = await graph.ainvoke(Command(resume="done"), config)
    assert resumed == {"answer": "done"}

    assert len(handler.resume_events) == 1
    assert handler.resume_events[0].checkpoint_ns == ()
    assert langchain_handler.events == []


def test_graph_callbacks_continue_when_interrupt_handler_raises_sync() -> None:
    graph = _build_interrupt_graph()
    raising_handler = _RaisingGraphEventHandler(raise_on_interrupt=True)
    recording_handler = _GraphEventHandler()

    first = graph.invoke(
        {"answer": None},
        {
            "configurable": {"thread_id": "graph-callback-sync-raises"},
            "callbacks": [raising_handler, recording_handler],
        },
    )

    assert "__interrupt__" in first
    assert len(recording_handler.interrupt_events) == 1


def test_graph_callbacks_continue_when_resume_handler_raises_sync() -> None:
    graph = _build_interrupt_graph()
    raising_handler = _RaisingGraphEventHandler(raise_on_resume=True)
    recording_handler = _GraphEventHandler()
    config = {
        "configurable": {"thread_id": "graph-callback-sync-raises-resume"},
        "callbacks": [raising_handler, recording_handler],
    }

    first = graph.invoke({"answer": None}, config)
    assert "__interrupt__" in first

    resumed = graph.invoke(Command(resume="done"), config)
    assert resumed == {"answer": "done"}
    assert len(recording_handler.resume_events) == 1


def test_graph_callbacks_raise_error_propagates_sync() -> None:
    graph = _build_interrupt_graph()
    raising_handler = _RaisingGraphEventHandler(
        raise_on_interrupt=True,
        raise_error=True,
    )

    with pytest.raises(ValueError, match="boom-interrupt"):
        graph.invoke(
            {"answer": None},
            {
                "configurable": {"thread_id": "graph-callback-sync-raise-error"},
                "callbacks": [raising_handler],
            },
        )


@pytest.mark.anyio
@NEEDS_CONTEXTVARS
async def test_graph_callbacks_continue_when_handler_raises_async() -> None:
    graph = _build_interrupt_graph()
    raising_interrupt_handler = _AsyncRaisingGraphEventHandler(raise_on_interrupt=True)
    recording_handler = _GraphEventHandler()
    config = {
        "configurable": {"thread_id": "graph-callback-async-raises-interrupt"},
        "callbacks": [raising_interrupt_handler, recording_handler],
    }

    first = await graph.ainvoke({"answer": None}, config)
    assert "__interrupt__" in first
    assert len(recording_handler.interrupt_events) == 1

    graph = _build_interrupt_graph()
    raising_resume_handler = _AsyncRaisingGraphEventHandler(raise_on_resume=True)
    recording_handler = _GraphEventHandler()
    config = {
        "configurable": {"thread_id": "graph-callback-async-raises-resume"},
        "callbacks": [raising_resume_handler, recording_handler],
    }

    first = await graph.ainvoke({"answer": None}, config)
    assert "__interrupt__" in first
    resumed = await graph.ainvoke(Command(resume="done"), config)
    assert resumed == {"answer": "done"}
    assert len(recording_handler.resume_events) == 1


@pytest.mark.anyio
@NEEDS_CONTEXTVARS
async def test_graph_callbacks_raise_error_propagates_async() -> None:
    graph = _build_interrupt_graph()
    raising_handler = _AsyncRaisingGraphEventHandler(
        raise_on_interrupt=True,
        raise_error=True,
    )

    with pytest.raises(ValueError, match="boom-interrupt"):
        await graph.ainvoke(
            {"answer": None},
            {
                "configurable": {"thread_id": "graph-callback-async-raise-error"},
                "callbacks": [raising_handler],
            },
        )


def test_graph_callbacks_accept_base_callback_manager() -> None:
    graph = _build_interrupt_graph()
    graph_handler = _GraphEventHandler()
    custom_handler = _LangChainCustomEventHandler()
    manager = CallbackManager.configure(inheritable_callbacks=[custom_handler])
    manager.add_handler(graph_handler)

    first = graph.invoke(
        {"answer": None},
        {
            "configurable": {"thread_id": "graph-callback-base-manager"},
            "callbacks": manager,
        },
    )

    assert "__interrupt__" in first
    assert len(graph_handler.interrupt_events) == 1
