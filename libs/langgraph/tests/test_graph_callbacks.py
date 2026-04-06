from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.callbacks import GraphCallbackHandler
from langgraph.graph import START, StateGraph
from langgraph.types import Command, Interrupt, interrupt


class _GraphEventHandler(GraphCallbackHandler):
    def __init__(self) -> None:
        self.interrupt_events: list[dict[str, Any]] = []
        self.resume_events: list[dict[str, Any]] = []

    def on_interrupt(
        self,
        interrupts: list[Interrupt] | tuple[Interrupt, ...],
        *,
        run_id: Any,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> Any:
        self.interrupt_events.append(
            {
                "run_id": run_id,
                "status": status,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
                "is_nested": is_nested,
                "interrupts": interrupts,
            }
        )

    def on_resume(
        self,
        *,
        run_id: Any,
        status: str,
        checkpoint_id: str,
        checkpoint_ns: tuple[str, ...],
        is_nested: bool,
    ) -> Any:
        self.resume_events.append(
            {
                "run_id": run_id,
                "status": status,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
                "is_nested": is_nested,
            }
        )


class _LangChainCustomEventHandler(BaseCallbackHandler):
    run_inline = True

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_custom_event(self, name: str, data: Any, **kwargs: Any) -> Any:
        self.events.append(name)


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
        "callbacks": [langchain_handler],
        "graph_callbacks": [handler],
    }

    first = graph.invoke({"answer": None}, config)
    assert "__interrupt__" in first

    assert len(handler.interrupt_events) == 1
    assert handler.interrupt_events[0]["is_nested"] is False
    assert handler.interrupt_events[0]["interrupts"]
    assert isinstance(handler.interrupt_events[0]["interrupts"][0], Interrupt)
    assert langchain_handler.events == []

    handler.resume_events.clear()
    resumed = graph.invoke(Command(resume="done"), config)
    assert resumed == {"answer": "done"}

    assert len(handler.resume_events) == 1
    assert handler.resume_events[0]["is_nested"] is False
    assert langchain_handler.events == []


@pytest.mark.anyio
async def test_graph_callbacks_interrupt_and_resume_async() -> None:
    graph = _build_interrupt_graph()
    handler = _GraphEventHandler()
    langchain_handler = _LangChainCustomEventHandler()
    config = {
        "configurable": {"thread_id": "graph-callback-async"},
        "callbacks": [langchain_handler],
        "graph_callbacks": [handler],
    }

    first = await graph.ainvoke({"answer": None}, config)
    assert "__interrupt__" in first

    assert len(handler.interrupt_events) == 1
    assert handler.interrupt_events[0]["is_nested"] is False
    assert handler.interrupt_events[0]["interrupts"]
    assert isinstance(handler.interrupt_events[0]["interrupts"][0], Interrupt)
    assert langchain_handler.events == []

    handler.resume_events.clear()
    resumed = await graph.ainvoke(Command(resume="done"), config)
    assert resumed == {"answer": "done"}

    assert len(handler.resume_events) == 1
    assert handler.resume_events[0]["is_nested"] is False
    assert langchain_handler.events == []
