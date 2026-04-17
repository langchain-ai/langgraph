"""Benchmark: DiffChannel vs BinaryOperatorAggregate storage and time.

Run directly:  python tests/test_diff_channel_benchmark.py
Run via pytest: pytest tests/test_diff_channel_benchmark.py -s
"""

from __future__ import annotations

import sys
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.channels.diff import DiffChannel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------


class BinaryState(TypedDict):
    messages: Annotated[list, add_messages]


class DiffState(TypedDict):
    messages: Annotated[list, DiffChannel(add_messages)]


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def _make_graph(state_cls: type) -> Any:
    def human_node(state: Any) -> dict:
        return {}  # no-op; caller provides messages via invoke

    def ai_node(state: Any) -> dict:
        last = state["messages"][-1]
        return {"messages": [AIMessage(content=f"reply-to-{last.id}")]}

    g = StateGraph(state_cls)
    g.add_node("human", human_node)
    g.add_node("ai", ai_node)
    g.add_edge("human", "ai")
    g.add_edge("ai", END)
    g.set_entry_point("human")
    return g.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _total_blob_bytes(saver: MemorySaver) -> int:
    total = 0
    for (_, _, _, _), (type_tag, blob) in saver.blobs.items():
        if blob is not None:
            total += len(blob)
    return total


def _run_turns(n_turns: int, state_cls: type) -> tuple[float, int]:
    """Run n_turns conversation turns; return (elapsed_seconds, total_blob_bytes)."""
    graph = _make_graph(state_cls)
    saver: MemorySaver = graph.checkpointer  # type: ignore[assignment]
    config = {"configurable": {"thread_id": "bench"}}

    t0 = time.perf_counter()
    for i in range(n_turns):
        graph.invoke({"messages": [HumanMessage(content=f"msg-{i}", id=f"h{i}")]}, config)
    elapsed = time.perf_counter() - t0

    blob_bytes = _total_blob_bytes(saver)
    return elapsed, blob_bytes


# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------

TURN_COUNTS = [10, 50, 100, 200, 500]


def run_benchmark() -> None:
    print()
    print("DiffChannel vs BinaryOperatorAggregate — checkpoint storage & time benchmark")
    print("=" * 76)
    header = f"{'turns':>6}  {'binary_bytes':>14}  {'diff_bytes':>12}  {'ratio':>8}  {'binary_ms':>12}  {'diff_ms':>10}"
    print(header)
    print("-" * 76)

    for turns in TURN_COUNTS:
        b_time, b_bytes = _run_turns(turns, BinaryState)
        d_time, d_bytes = _run_turns(turns, DiffState)
        ratio = b_bytes / d_bytes if d_bytes else float("inf")
        print(
            f"{turns:>6}  {b_bytes:>14,}  {d_bytes:>12,}  {ratio:>7.1f}x  "
            f"{b_time * 1000:>11.1f}ms  {d_time * 1000:>9.1f}ms"
        )

    print("=" * 76)
    print()
    print("ratio = binary_bytes / diff_bytes (higher = DiffChannel saves more space)")
    print()


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


def test_diff_channel_benchmark(capsys: Any) -> None:
    """Storage grows O(N²) for BinaryOperatorAggregate, O(N) for DiffChannel."""
    with capsys.disabled():
        run_benchmark()

    # Verify DiffChannel uses strictly less storage for 100+ turns.
    for turns in [100, 500]:
        _, b_bytes = _run_turns(turns, BinaryState)
        _, d_bytes = _run_turns(turns, DiffState)
        assert d_bytes < b_bytes, (
            f"Expected DiffChannel to use less storage at {turns} turns, "
            f"got diff={d_bytes} binary={b_bytes}"
        )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_benchmark()
    sys.exit(0)
