"""Benchmark: DeltaChannel — multi-channel reads, mixed snapshot frequencies.

Run directly:  python tests/test_delta_channel_benchmark.py
Run via pytest: pytest tests/test_delta_channel_benchmark.py -s

Sweeps `(K delta channels, snapshot_frequency strategy, turn count)` and
reports per-scenario read latency, write latency, storage, and peak Python
heap usage during `get_state`.

Scenarios cover the dimensions where this branch's optimizations matter:

  * K-channel batching   — varying K (number of `DeltaChannel`s the graph
                           reads on hydrate) shows the effect of merging
                           per-channel reads into a single saver call.
  * Mixed frequencies    — channels with very different snapshot cadences
                           in one graph exercise the per-channel chain
                           bound in stage-2.
  * Turn count           — chain depth shows how paged stage-1 holds up.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import time
import tracemalloc
from typing import Annotated, Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
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
# Realistic message payload (~100 tokens / ~400 chars each)
# ---------------------------------------------------------------------------

_HUMAN_TEMPLATE = (
    "I need help understanding the implications of {topic} on our system "
    "architecture. Specifically, I'm concerned about how this interacts with "
    "our existing {concern} and whether we need to refactor the {component} "
    "layer before proceeding."
)

_TOPICS = [
    "distributed tracing",
    "eventual consistency",
    "schema migration",
    "backpressure handling",
    "idempotency guarantees",
]
_CONCERNS = ["concurrency model", "retry semantics", "ordering guarantees"]
_COMPONENTS = ["persistence", "ingestion", "routing"]


def _human_content(i: int) -> str:
    return _HUMAN_TEMPLATE.format(
        topic=_TOPICS[i % len(_TOPICS)],
        concern=_CONCERNS[i % len(_CONCERNS)],
        component=_COMPONENTS[i % len(_COMPONENTS)],
    )


# ---------------------------------------------------------------------------
# State / graph factory: K DeltaChannel fields with per-channel freqs
# ---------------------------------------------------------------------------


def _make_state_cls(freqs: list[int]) -> type:
    """Build a TypedDict with one DeltaChannel per entry in `freqs`."""
    fields: dict[str, Any] = {}
    for i, freq in enumerate(freqs):
        ch = DeltaChannel(_messages_delta_reducer, snapshot_frequency=freq)
        fields[f"ch{i}"] = Annotated[list, ch]
    return TypedDict(  # type: ignore[return-value]
        "_BenchState_" + "-".join(str(f) for f in freqs),
        fields,
    )


def _make_graph(state_cls: type, K: int, checkpointer: Any = None) -> Any:
    """Graph: one node, writes a fresh message into every channel each turn."""

    def fanout(state: Any) -> dict[str, Any]:
        i = max(len(state.get(f"ch{j}", [])) for j in range(K))
        # Each channel gets its own copy with a unique id so the reducer
        # does meaningful per-channel state accumulation.
        return {
            f"ch{j}": [HumanMessage(content=_human_content(i), id=f"c{j}_{i}")]
            for j in range(K)
        }

    g = StateGraph(state_cls)
    g.add_node("fanout", fanout)
    g.set_entry_point("fanout")
    g.add_edge("fanout", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


# ---------------------------------------------------------------------------
# Storage / memory measurement
# ---------------------------------------------------------------------------


def _inmemory_blob_bytes(saver: MemorySaver) -> int:
    return sum(
        len(blob) for (_, _, _, _), (_, blob) in saver.blobs.items() if blob is not None
    )


def _postgres_storage_bytes(saver: Any, thread_id: str) -> int:
    """Total bytes across checkpoints / checkpoint_blobs / checkpoint_writes
    rows for this `thread_id`. Uses `pg_column_size` for an in-row payload
    estimate; faster than full-table size and scoped to the thread."""
    sql = """
    SELECT COALESCE(SUM(pg_column_size(c.*)), 0)
         + COALESCE((SELECT SUM(pg_column_size(b.*)) FROM checkpoint_blobs b
                     WHERE b.thread_id = %s), 0)
         + COALESCE((SELECT SUM(pg_column_size(w.*)) FROM checkpoint_writes w
                     WHERE w.thread_id = %s), 0)
         AS total
    FROM checkpoints c
    WHERE c.thread_id = %s
    """
    with saver._cursor() as cur:
        cur.execute(sql, (thread_id, thread_id, thread_id))
        row = cur.fetchone()
    if row is None:
        return 0
    if isinstance(row, dict):
        return int(row.get("total") or 0)
    return int(row[0] or 0)


def _run_scenario(
    freqs: list[int],
    n_turns: int,
    checkpointer: Any,
    thread_id: str,
) -> dict[str, float | int]:
    """Drive `n_turns` invocations of a K-channel graph, measure
    write/read/storage/peak-memory."""
    K = len(freqs)
    state_cls = _make_state_cls(freqs)
    graph = _make_graph(state_cls, K, checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    # Write phase
    t0 = time.perf_counter()
    for i in range(n_turns):
        graph.invoke({}, config)
    write_elapsed = time.perf_counter() - t0

    # Read phase + tracemalloc peak across get_state calls
    gc.collect()
    tracemalloc.start()
    t1 = time.perf_counter()
    for _ in range(5):
        graph.get_state(config)
    read_elapsed = (time.perf_counter() - t1) / 5
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Storage
    if isinstance(graph.checkpointer, MemorySaver):
        storage = _inmemory_blob_bytes(graph.checkpointer)
    elif _POSTGRES_AVAILABLE and isinstance(graph.checkpointer, PostgresSaver):
        storage = _postgres_storage_bytes(graph.checkpointer, thread_id)
    else:
        storage = -1

    return {
        "K": K,
        "turns": n_turns,
        "write_total_s": write_elapsed,
        "write_per_invoke_ms": (write_elapsed / n_turns) * 1000,
        "read_avg_ms": read_elapsed * 1000,
        "storage_bytes": storage,
        "peak_mem_bytes": peak_bytes,
    }


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _pg_saver(thread_id: str):
    with PostgresSaver.from_conn_string(_POSTGRES_URI) as saver:
        saver.setup()
        with saver._cursor() as cur:
            for tbl in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                cur.execute(f"DELETE FROM {tbl} WHERE thread_id = %s", (thread_id,))
        yield saver
        with saver._cursor() as cur:
            for tbl in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                cur.execute(f"DELETE FROM {tbl} WHERE thread_id = %s", (thread_id,))


def _checkpointers() -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = [("InMemory", None)]
    if _POSTGRES_AVAILABLE:
        try:
            import psycopg

            psycopg.connect(_POSTGRES_URI).close()
            result.append(("Postgres", "postgres"))
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[tuple[str, list[int]]] = [
    ("K=1, freq=50", [50]),
    ("K=3, freq=50 uniform", [50, 50, 50]),
    ("K=3, freq=mixed", [50, 200, 1000]),
    ("K=8, freq=50 uniform", [50] * 8),
    ("K=8, freq=mixed", [25, 50, 100, 200, 500, 1000, 1000, 1000]),
]

TURN_COUNTS = [100, 500]


def _fmt_bytes(n: int) -> str:
    if n < 0:
        return "n/a"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def _print_scenario_table(cp_label: str, rows: list[dict]) -> None:
    print(f"\n  [{cp_label}]")
    header = (
        f"  {'scenario':<28}{'turns':>8}{'write_ms':>11}"
        f"{'read_ms':>10}{'storage':>12}{'peak_mem':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        print(
            f"  {row['scenario']:<28}"
            f"{row['turns']:>8}"
            f"{row['write_per_invoke_ms']:>10.1f}"
            f"{row['read_avg_ms']:>10.2f}"
            f"{_fmt_bytes(row['storage_bytes']):>12}"
            f"{_fmt_bytes(row['peak_mem_bytes']):>12}"
        )


def run_benchmark() -> list[dict]:
    """Run the full sweep and return all measurement rows."""
    all_rows: list[dict] = []
    for cp_label, cp_hint in _checkpointers():
        rows: list[dict] = []
        for scenario_label, freqs in SCENARIOS:
            for turns in TURN_COUNTS:
                thread_id = f"bench-{scenario_label.replace(' ', '_')}-{turns}".lower()
                if cp_hint is None:
                    saver_ctx: Any = contextlib.nullcontext(None)
                else:
                    saver_ctx = _pg_saver(thread_id)
                with saver_ctx as saver:
                    measured = _run_scenario(freqs, turns, saver, thread_id)
                measured["scenario"] = scenario_label
                measured["saver"] = cp_label
                rows.append(measured)
                all_rows.append(measured)
        _print_scenario_table(cp_label, rows)
    return all_rows


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="slow benchmark — run manually with: python tests/test_delta_channel_benchmark.py"
)
def test_delta_channel_benchmark(capsys: Any) -> None:
    """Manual benchmark — see module docstring."""
    with capsys.disabled():
        run_benchmark()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("DeltaChannel benchmark — multi-channel reads, mixed frequencies")
    print("=" * 78)
    run_benchmark()
    sys.exit(0)
