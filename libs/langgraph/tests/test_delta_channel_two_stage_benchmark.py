"""Benchmark: DeltaChannel two-stage vs one-stage query path.

Compares read latency and correctness between the default one-stage
`SELECT_DELTA_COMBINED_SQL` path and the two-stage path gated by
`LG_DELTA_TWO_STAGE_QUERY=1`.

Workload: 1000 messages x ~10 KB each, snapshot_frequency=10.

Run directly:  python tests/test_delta_channel_two_stage_benchmark.py
Run via pytest: pytest tests/test_delta_channel_two_stage_benchmark.py -s
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import _messages_delta_reducer

try:
    from langgraph.checkpoint.postgres import PostgresSaver

    _POSTGRES_AVAILABLE = True
    _POSTGRES_URI = os.environ.get(
        "LANGGRAPH_BENCH_POSTGRES_URI",
        "postgres://postgres@localhost:5432/postgres?sslmode=disable",
    )
except ImportError:
    _POSTGRES_AVAILABLE = False

# ---------------------------------------------------------------------------
# Synthetic ~10 KB message payload
# ---------------------------------------------------------------------------

_PADDING = "X" * 9500

def _human_content(i: int) -> str:
    return f"human-turn-{i} {_PADDING}"

def _ai_content(i: int) -> str:
    return f"ai-turn-{i} {_PADDING}"

# ---------------------------------------------------------------------------
# State & graph
# ---------------------------------------------------------------------------

SNAPSHOT_FREQ = 10

channel = DeltaChannel(
    _messages_delta_reducer,
    snapshot_frequency=SNAPSHOT_FREQ,
)

DeltaState = TypedDict(
    "DeltaState",
    {"messages": Annotated[list, channel]},
)


def _make_graph(checkpointer: Any) -> Any:
    def human_node(state: Any) -> dict:
        return {}

    def ai_node(state: Any) -> dict:
        i = len(state["messages"]) // 2
        return {"messages": [AIMessage(content=_ai_content(i), id=f"a{i}")]}

    g = StateGraph(DeltaState)
    g.add_node("human", human_node)
    g.add_node("ai", ai_node)
    g.add_edge("human", "ai")
    g.add_edge("ai", END)
    g.set_entry_point("human")
    return g.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pg_saver(thread_id: str = "bench2stage"):
    """Create a fresh PostgresSaver, clean up thread data before and after."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with PostgresSaver.from_conn_string(_POSTGRES_URI) as saver:
            saver.setup()
            with saver._cursor() as cur:
                for tbl in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                    cur.execute(f"DELETE FROM {tbl} WHERE thread_id = %s", (thread_id,))
            yield saver
            with saver._cursor() as cur:
                for tbl in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                    cur.execute(f"DELETE FROM {tbl} WHERE thread_id = %s", (thread_id,))

    return _ctx()


def _count_rows_and_bytes(saver: Any, thread_id: str, channel_name: str) -> dict:
    """Query raw row counts and byte sizes for the thread's delta channel data."""
    stats: dict[str, Any] = {}
    with saver._cursor() as cur:
        cur.execute(
            "SELECT count(*) AS cnt, coalesce(sum(length(blob)), 0) AS bytes "
            "FROM checkpoint_blobs WHERE thread_id = %s AND channel = %s",
            (thread_id, channel_name),
        )
        row = cur.fetchone()
        stats["blob_rows"] = row["cnt"]
        stats["blob_bytes"] = row["bytes"]

        cur.execute(
            "SELECT count(*) AS cnt, coalesce(sum(length(blob)), 0) AS bytes "
            "FROM checkpoint_writes WHERE thread_id = %s AND channel = %s",
            (thread_id, channel_name),
        )
        row = cur.fetchone()
        stats["write_rows"] = row["cnt"]
        stats["write_bytes"] = row["bytes"]

        cur.execute(
            "SELECT count(*) AS cnt FROM checkpoints WHERE thread_id = %s",
            (thread_id,),
        )
        row = cur.fetchone()
        stats["checkpoint_rows"] = row["cnt"]

    return stats


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

N_TURNS = int(os.environ.get("BENCH_N_TURNS", "100"))
N_READS = 5


def run_two_stage_benchmark() -> None:
    if not _POSTGRES_AVAILABLE:
        print("PostgresSaver not available, skipping benchmark")
        return

    try:
        import psycopg
        psycopg.connect(_POSTGRES_URI).close()
    except Exception:
        print(f"Cannot connect to Postgres at {_POSTGRES_URI}, skipping")
        return

    thread_id = "bench2stage"

    print()
    print("=" * 76)
    print("DeltaChannel Two-Stage Query Benchmark")
    print(f"  {N_TURNS} messages x ~10 KB each, snapshot_frequency={SNAPSHOT_FREQ}")
    print("=" * 76)

    # Phase 1: build the state
    print(f"\nPhase 1: Writing {N_TURNS} turns...")
    with _pg_saver(thread_id) as saver:
        graph = _make_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}

        t0 = time.perf_counter()
        for i in range(N_TURNS):
            graph.invoke(
                {"messages": [HumanMessage(content=_human_content(i), id=f"h{i}")]},
                config,
            )
            if (i + 1) % 200 == 0:
                print(f"  ... {i + 1}/{N_TURNS} turns written")
        write_elapsed = time.perf_counter() - t0
        print(f"  Write phase: {write_elapsed:.1f}s ({write_elapsed/N_TURNS*1000:.1f}ms/turn)")

        # Storage stats
        stats = _count_rows_and_bytes(saver, thread_id, "messages")
        print(f"\n  Storage stats:")
        print(f"    checkpoint rows:  {stats['checkpoint_rows']}")
        print(f"    blob rows:        {stats['blob_rows']}  ({_fmt_bytes(stats['blob_bytes'])})")
        print(f"    write rows:       {stats['write_rows']}  ({_fmt_bytes(stats['write_bytes'])})")

        # Phase 2: read benchmarks — use a fresh graph for each mode to
        # avoid in-memory channel caching across measurements.
        results: dict[str, dict] = {}

        for mode_label, env_val in [("1-stage (default)", ""), ("2-stage (optimized)", "1")]:
            os.environ["LG_DELTA_TWO_STAGE_QUERY"] = env_val
            fresh_graph = _make_graph(saver)

            # Warm up
            fresh_graph.get_state(config)

            t1 = time.perf_counter()
            for _ in range(N_READS):
                state = fresh_graph.get_state(config)
            read_elapsed = (time.perf_counter() - t1) / N_READS

            msg_ids = [m.id for m in state.values["messages"]]
            results[mode_label] = {
                "read_ms": read_elapsed * 1000,
                "msg_count": len(msg_ids),
                "msg_ids": msg_ids,
            }

        # Reset env
        os.environ.pop("LG_DELTA_TWO_STAGE_QUERY", None)

        # Phase 3: report
        print(f"\n  Read latency (avg of {N_READS} get_state calls):")
        print(f"  {'mode':<30}  {'latency':>12}  {'msgs':>6}")
        print("  " + "-" * 52)
        for label, r in results.items():
            print(f"  {label:<30}  {r['read_ms']:>9.1f}ms  {r['msg_count']:>6}")

        labels = list(results.keys())
        r1 = results[labels[0]]
        r2 = results[labels[1]]
        if r1["read_ms"] > 0:
            speedup = r1["read_ms"] / r2["read_ms"]
            print(f"\n  Speedup: {speedup:.2f}x")

        # Correctness
        assert r1["msg_ids"] == r2["msg_ids"], (
            "CORRECTNESS FAILURE: message IDs differ between 1-stage and 2-stage!"
        )
        print(f"\n  Correctness: PASS (message IDs identical)")

    print()


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="slow benchmark — run manually with: python tests/test_delta_channel_two_stage_benchmark.py"
)
def test_two_stage_benchmark(capsys: Any) -> None:
    """Two-stage query should produce identical results to one-stage."""
    with capsys.disabled():
        run_two_stage_benchmark()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_two_stage_benchmark()
    sys.exit(0)
