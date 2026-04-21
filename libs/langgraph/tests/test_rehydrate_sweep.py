"""Sweep snapshot_every values to find the storage vs. time-travel tradeoff.

Run directly:  python tests/test_rehydrate_sweep.py
Run via pytest: pytest tests/test_rehydrate_sweep.py -s
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REHYDRATE_SWEEP = [5, 10, 25, 50, 100, None]  # None = no rehydration (pure diff)
TURN_COUNTS = [50, 100, 250, 500]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(snapshot_every: int | None) -> type:
    channel = DeltaChannel(add_messages, snapshot_every=snapshot_every)
    return TypedDict("S", {"messages": Annotated[list, channel]})


def _make_graph(state_cls: type) -> Any:
    def human_node(state: Any) -> dict:
        return {}

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


def _total_blob_bytes(saver: MemorySaver) -> int:
    total = 0
    for (_, _, _, _), (type_tag, blob) in saver.blobs.items():
        if blob is not None:
            total += len(blob)
    return total


def _measure_time_travel_ms(graph: Any, config: dict) -> float:
    """Time how long it takes to get state at the very first checkpoint (worst case)."""
    history = list(graph.get_state_history(config))
    if not history:
        return 0.0
    oldest = history[-1]
    t0 = time.perf_counter()
    graph.get_state(oldest.config)
    return (time.perf_counter() - t0) * 1000


def _run(n_turns: int, snapshot_every: int | None) -> tuple[float, int, float]:
    """Returns (write_ms, blob_bytes, time_travel_ms)."""
    state_cls = _make_state(snapshot_every)
    graph = _make_graph(state_cls)
    saver: MemorySaver = graph.checkpointer  # type: ignore[assignment]
    config = {"configurable": {"thread_id": "sweep"}}

    t0 = time.perf_counter()
    for i in range(n_turns):
        graph.invoke(
            {"messages": [HumanMessage(content=f"msg-{i}", id=f"h{i}")]}, config
        )
    write_ms = (time.perf_counter() - t0) * 1000

    blob_bytes = _total_blob_bytes(saver)
    tt_ms = _measure_time_travel_ms(graph, config)
    return write_ms, blob_bytes, tt_ms


# ---------------------------------------------------------------------------
# ASCII sparkline
# ---------------------------------------------------------------------------


def _sparkline(values: list[float], width: int = 20) -> str:
    bars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    chars = [bars[round((v - lo) / span * (len(bars) - 1))] for v in values]
    return "".join(chars).ljust(width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_sweep() -> None:
    label = {v: (str(v) if v is not None else "None(∞)") for v in REHYDRATE_SWEEP}

    print()
    print("snapshot_every sweep — storage vs time-travel cost")
    print("=" * 90)

    for turns in TURN_COUNTS:
        print(f"\n--- {turns} turns ---")
        col_w = 12
        header = (
            f"{'snapshot_every':>18}  "
            f"{'blob_bytes':>{col_w}}  "
            f"{'write_ms':>{col_w}}  "
            f"{'time_travel_ms':>{col_w}}"
        )
        print(header)
        print("-" * 60)

        tt_vals: list[float] = []
        byte_vals: list[int] = []
        write_vals: list[float] = []
        rows: list[tuple] = []

        for rv in REHYDRATE_SWEEP:
            write_ms, blob_bytes, tt_ms = _run(turns, rv)
            rows.append((rv, blob_bytes, write_ms, tt_ms))
            byte_vals.append(blob_bytes)
            write_vals.append(write_ms)
            tt_vals.append(tt_ms)

        for rv, blob_bytes, write_ms, tt_ms in rows:
            print(
                f"{label[rv]:>18}  "
                f"{blob_bytes:>{col_w},}  "
                f"{write_ms:>{col_w}.1f}  "
                f"{tt_ms:>{col_w}.2f}"
            )

        print()
        print(
            f"  bytes       spark: [{_sparkline(byte_vals)}]  "
            f"lo={min(byte_vals):,}  hi={max(byte_vals):,}"
        )
        print(
            f"  time-travel spark: [{_sparkline(tt_vals)}]  "
            f"lo={min(tt_vals):.2f}ms  hi={max(tt_vals):.2f}ms"
        )
        print(
            f"  write       spark: [{_sparkline(write_vals)}]  "
            f"lo={min(write_vals):.1f}ms  hi={max(write_vals):.1f}ms"
        )

    print()
    print("=" * 90)
    print(
        "snapshot_every=None means pure diff (no snapshots) — "
        "lowest storage, highest time-travel cost."
    )
    print(
        "Lower snapshot_every = more frequent full snapshots = "
        "faster time-travel, more storage."
    )
    print()


def test_rehydrate_sweep(capsys: Any) -> None:
    with capsys.disabled():
        run_sweep()


if __name__ == "__main__":
    run_sweep()
    sys.exit(0)
