"""Benchmark: DeltaChannel vs BinaryOperatorAggregate storage and time.

Run directly:  python tests/test_delta_channel_benchmark.py
Run via pytest: pytest tests/test_delta_channel_benchmark.py -s
"""

from __future__ import annotations

import sys
import time
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

REHYDRATE_EVERY = 50


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------


class BinaryState(TypedDict):
    messages: Annotated[list, add_messages]


class DiffState(TypedDict):
    messages: Annotated[list, DeltaChannel(add_messages)]


class DiffRehydrateState(TypedDict):
    messages: Annotated[
        list, DeltaChannel(add_messages, snapshot_every=REHYDRATE_EVERY)
    ]


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
        graph.invoke(
            {"messages": [HumanMessage(content=f"msg-{i}", id=f"h{i}")]}, config
        )
    elapsed = time.perf_counter() - t0

    blob_bytes = _total_blob_bytes(saver)
    return elapsed, blob_bytes


# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------

TURN_COUNTS = [10, 50, 100, 200, 500]


def run_benchmark() -> None:
    print()
    print(
        "DeltaChannel vs BinaryOperatorAggregate — checkpoint storage & time benchmark"
    )
    w = 100
    print("=" * w)
    header = (
        f"{'turns':>6}  "
        f"{'bin_bytes':>12}  {'diff_bytes':>12}  {'rehy_bytes':>12}  {'bytes_ratio':>12}  "
        f"{'bin_ms':>9}  {'diff_ms':>9}  {'rehy_ms':>9}  {'time_ratio':>12}"
    )
    print(header)
    print("-" * w)

    for turns in TURN_COUNTS:
        b_time, b_bytes = _run_turns(turns, BinaryState)
        d_time, d_bytes = _run_turns(turns, DiffState)
        r_time, r_bytes = _run_turns(turns, DiffRehydrateState)
        bytes_ratio = b_bytes / d_bytes if d_bytes else float("inf")
        time_ratio = d_time / b_time if b_time else float("inf")
        print(
            f"{turns:>6}  "
            f"{b_bytes:>12,}  {d_bytes:>12,}  {r_bytes:>12,}  {bytes_ratio:>11.1f}x  "
            f"{b_time * 1000:>8.1f}ms  {d_time * 1000:>8.1f}ms  {r_time * 1000:>8.1f}ms  {time_ratio:>11.1f}x"
        )

    print("=" * w)
    print()
    print("bytes_ratio = bin_bytes / diff_bytes  (higher = more storage saved)")
    print(
        "time_ratio  = diff_ms / bin_ms        (higher = more overhead without rehydration)"
    )
    print(
        f"rehy        = DeltaChannel(snapshot_every={REHYDRATE_EVERY}) — caps chain depth"
    )
    print()


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


def test_delta_channel_benchmark(capsys: Any) -> None:
    """Storage grows O(N²) for BinaryOperatorAggregate, O(N) for DeltaChannel."""
    with capsys.disabled():
        run_benchmark()

    # Verify DeltaChannel uses strictly less storage for 100+ turns.
    for turns in [100, 500]:
        _, b_bytes = _run_turns(turns, BinaryState)
        _, d_bytes = _run_turns(turns, DiffState)
        _, r_bytes = _run_turns(turns, DiffRehydrateState)
        assert d_bytes < b_bytes, (
            f"Expected DeltaChannel to use less storage at {turns} turns, "
            f"got diff={d_bytes} binary={b_bytes}"
        )
        assert r_bytes < b_bytes, (
            f"Expected DeltaChannel(rehydrate) to use less storage at {turns} turns, "
            f"got rehydrate={r_bytes} binary={b_bytes}"
        )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_benchmark()
    sys.exit(0)
