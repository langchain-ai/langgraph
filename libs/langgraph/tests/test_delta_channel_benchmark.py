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

import argparse
import contextlib
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
        try:
            yield saver
        finally:
            saver.delete_thread(thread_id)


def _checkpointers() -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = [("InMemory", None)]
    if _POSTGRES_AVAILABLE:
        try:
            import psycopg  # noqa: PLC0415

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


def test_scripted_snapshot_sweep() -> None:
    saver = MemorySaver()
    graph = _scripted_graph(2, "tool_calls", saver)
    config = {"configurable": {"thread_id": "scripted-test"}}
    for _ in range(3):
        graph.invoke({}, config)
    state = graph.get_state(config).values
    assert len(state["messages"]) == 12
    assert len(state["files"]) == 3
    assert state["messages"][1].tool_calls[0]["id"] == "call-0"
    assert state["messages"][2].tool_call_id == "call-0"
    assert graph.get_state(config).values == state
    assert _memory_storage_bytes(saver, "scripted-test") > 0


def test_sweep_metrics_and_cleanup() -> None:
    rows = list(_measure_sweep(2, [2, 4], "mock700", "memory"))
    assert [row["read_samples"] for row in rows] == [2, 4]
    assert rows[1]["storage_bytes"] > rows[0]["storage_bytes"]
    assert rows[1]["read_max_ms"] >= rows[1]["read_p95_ms"] >= rows[1]["read_p50_ms"]
    assert 1 <= rows[1]["slowest_update"] <= 4
    assert _percentile([1, 2, 3, 4], 0.95) == 4
    points = [
        {"storage_bytes": 100, "read_max_ms": 1, "frequency": 25},
        {"storage_bytes": 50, "read_max_ms": 2, "frequency": 50},
        {"storage_bytes": 10, "read_max_ms": 10, "frequency": 100},
    ]
    assert _pareto_knee(points, "read_max_ms")["frequency"] == 50


FREQUENCIES = list(range(25, 501, 25))
LENGTHS = [50, 200, 1000, 5000, 10000]
PROFILES = ["mock700", "tool_calls"]


def _files_delta(state: dict, writes: Sequence[dict]) -> dict:
    return {**state, **{key: value for write in writes for key, value in write.items()}}


def _script(i: int, profile: str) -> list:
    """Return reproducible, ID-stable messages without contacting a provider."""
    if profile == "mock700":
        return [
            HumanMessage(content="h" * 350, id=f"human-{i}"),
            AIMessage(content="a" * 350, id=f"answer-{i}"),
        ]
    call_id = f"call-{i}"
    return [
        HumanMessage(content=_human_content(i), id=f"human-{i}"),
        AIMessage(
            content="I will inspect the source and write a report.",
            id=f"call-message-{i}",
            tool_calls=[
                {"name": "read_file", "args": {"path": f"/src/{i}.py"}, "id": call_id}
            ],
        ),
        ToolMessage(
            content=(
                "def transform(record):\n    return record.get('value', 0)\n" * 64
            ),
            tool_call_id=call_id,
            id=f"tool-{i}",
        ),
        AIMessage(
            content="The implementation is deterministic.\n" * 16, id=f"answer-{i}"
        ),
    ]


def _scripted_graph(freq: int, profile: str, saver: Any) -> Any:
    """Append one scripted conversation and file update per graph invocation."""
    fields = {
        "messages": Annotated[
            list, DeltaChannel(_messages_delta_reducer, snapshot_frequency=freq)
        ],
        "files": Annotated[dict, DeltaChannel(_files_delta, snapshot_frequency=freq)],
    }
    state_cls = TypedDict("ScriptedState", fields)
    counter = itertools.count()

    def turn(state: dict) -> dict:
        i = next(counter)
        script = _script(i, profile)
        model = GenericFakeChatModel(
            messages=iter(m for m in script if isinstance(m, AIMessage))
        )
        history = list(state.get("messages", []))
        emitted = []
        for message in script:
            actual = (
                model.invoke(history) if isinstance(message, AIMessage) else message
            )
            emitted.append(actual)
            history.append(actual)
        content = "report\n" * (16 if profile == "mock700" else 256)
        return {
            "messages": emitted,
            "files": {
                f"/reports/{i}.txt": {
                    "content": content.splitlines(),
                    "created_at": "2026-01-01T00:00:00Z",
                    "modified_at": "2026-01-01T00:00:00Z",
                }
            },
        }

    builder = StateGraph(state_cls)
    builder.add_node("scripted_chat", turn)
    builder.set_entry_point("scripted_chat")
    builder.add_edge("scripted_chat", END)
    return builder.compile(checkpointer=saver)


def _percentile(values: list[float], percentile: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * percentile) - 1)]


def _memory_storage_bytes(saver: MemorySaver, thread_id: str) -> int:
    """Count serialized payload bytes across all three memory saver stores."""
    blobs = sum(
        len(blob)
        for (tid, *_), (_, blob) in saver.blobs.items()
        if tid == thread_id and blob is not None
    )
    checkpoints = sum(
        len(checkpoint[1]) + len(metadata[1])
        for ns in saver.storage[thread_id].values()
        for checkpoint, metadata, _ in ns.values()
    )
    writes = sum(
        len(value[2][1])
        for (tid, *_), batch in saver.writes.items()
        if tid == thread_id
        for value in batch.values()
    )
    return blobs + checkpoints + writes


def _measure_sweep(freq: int, lengths: list[int], profile: str, backend: str):
    """Measure every prefix read, including pre-snapshot phases, without tracing."""
    thread_id = f"delta-sweep-{uuid.uuid4()}"
    ctx = (
        _pg_saver(thread_id)
        if backend == "postgres"
        else contextlib.nullcontext(MemorySaver())
    )
    with ctx as saver:
        graph = _scripted_graph(freq, profile, saver)
        config = {"configurable": {"thread_id": thread_id}}
        warm_config = {"configurable": {"thread_id": f"{thread_id}-warmup"}}
        try:
            warm_graph = _scripted_graph(freq, profile, saver)
            warm_graph.invoke({}, warm_config)
            warm_graph.get_state(warm_config)
        finally:
            saver.delete_thread(warm_config["configurable"]["thread_id"])
        reads: list[float] = []
        writes: list[float] = []
        slowest_update = 0
        slowest_config = config
        peak_read = -1.0
        try:
            for update in range(1, max(lengths) + 1):
                start = time.perf_counter()
                graph.invoke({}, config)
                writes.append((time.perf_counter() - start) * 1000)
                start = time.perf_counter()
                snapshot = graph.get_state(config)
                elapsed = (time.perf_counter() - start) * 1000
                reads.append(elapsed)
                if elapsed > peak_read:
                    peak_read = elapsed
                    slowest_update = update
                    slowest_config = snapshot.config
                if update not in lengths:
                    continue
                storage = (
                    _postgres_storage_bytes(saver, thread_id)
                    if backend == "postgres"
                    else _memory_storage_bytes(saver, thread_id)
                )
                gc.collect()
                tracemalloc.start()
                try:
                    graph.get_state(slowest_config)
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                yield {
                    "backend": backend,
                    "profile": profile,
                    "frequency": freq,
                    "updates": update,
                    "read_samples": len(reads),
                    "read_p50_ms": statistics.median(reads),
                    "read_p95_ms": _percentile(reads, 0.95),
                    "read_max_ms": max(reads),
                    "slowest_update": slowest_update,
                    "slowest_checkpoint_id": slowest_config["configurable"][
                        "checkpoint_id"
                    ],
                    "write_p50_ms": statistics.median(writes),
                    "write_p95_ms": _percentile(writes, 0.95),
                    "write_max_ms": max(writes),
                    "write_total_s": sum(writes) / 1000,
                    "storage_bytes": storage,
                    "slowest_read_peak_heap_bytes": peak,
                }
        finally:
            saver.delete_thread(thread_id)


def _pareto_knee(rows: list[dict], latency: str) -> dict:
    """Pick the nondominated point nearest the min-max normalized ideal."""
    frontier = [
        r
        for r in rows
        if not any(
            float(o["storage_bytes"]) <= float(r["storage_bytes"])
            and float(o[latency]) <= float(r[latency])
            and (
                float(o["storage_bytes"]) < float(r["storage_bytes"])
                or float(o[latency]) < float(r[latency])
            )
            for o in rows
        )
    ]
    bounds = {
        k: (min(float(r[k]) for r in rows), max(float(r[k]) for r in rows))
        for k in ("storage_bytes", latency)
    }

    def score(row: dict) -> float:
        return sum(
            ((float(row[k]) - lo) / (hi - lo or 1)) ** 2
            for k, (lo, hi) in bounds.items()
        )

    return min(frontier, key=score)


def _plot_sweep(csv_path: Path) -> None:
    """Render dependency-free SVG frequency and Pareto plots for each group."""
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    groups = sorted({(r["backend"], r["profile"], r["updates"]) for r in rows})
    for backend, profile, updates in groups:
        group = sorted(
            (
                r
                for r in rows
                if (r["backend"], r["profile"], r["updates"])
                == (backend, profile, updates)
            ),
            key=lambda r: int(r["frequency"]),
        )
        svg = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="420" viewBox="0 0 1100 420"><rect width="1100" height="420" fill="white"/>'
        ]
        svg.append(
            f'<text x="20" y="25" font-family="sans-serif" font-size="16">{backend} / {profile} / {updates} updates (n={len(group)} frequencies)</text>'
        )
        for panel, (xkey, ykeys, title) in enumerate(
            [
                (
                    "frequency",
                    ["storage_bytes", "read_p50_ms", "read_max_ms"],
                    "Frequency vs independently normalized metrics",
                ),
                (
                    "storage_bytes",
                    ["read_p50_ms", "read_max_ms"],
                    "Storage vs latency (Pareto tradeoff)",
                ),
            ]
        ):
            left = 60 + panel * 550
            svg.append(
                f'<path d="M {left} 60 V 340 H {left + 450}" fill="none" stroke="black"/>'
            )
            svg.append(
                f'<text x="{left}" y="50" font-family="sans-serif" font-size="12">{title}</text>'
            )
            xs = [float(r[xkey]) for r in group]
            xmin, xmax = min(xs), max(xs)
            ymax = max(float(r[k]) for r in group for k in ykeys)
            for color, key in zip(["#176baf", "#c14b14", "#6a378b"], ykeys):
                scale = max(float(r[key]) for r in group) if panel == 0 else ymax

                def coords(row: dict, scale: float = scale) -> tuple[float, float]:
                    return left + (float(row[xkey]) - xmin) / (
                        xmax - xmin or 1
                    ) * 450, 340 - float(row[key]) / (scale or 1) * 260

                points = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(coords, group))
                svg.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
                )
                if panel == 1:
                    knee = _pareto_knee(group, key)
                    x, y = coords(knee)
                    svg.append(
                        f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/><text x="{x + 6}" y="{y - 8}" font-family="sans-serif" font-size="11">f={knee["frequency"]}</text>'
                    )
                svg.append(
                    f'<text x="{left}" y="{365 + ykeys.index(key) * 15}" fill="{color}" font-family="sans-serif" font-size="11">{key}</text>'
                )
            svg.append(
                f'<text x="{left}" y="355" font-family="sans-serif" font-size="10">{xkey}: {xmin:.0f} to {xmax:.0f}; y: 0 to {1 if panel == 0 else ymax:.2f}{" normalized" if panel == 0 else " ms"}</text>'
            )
        svg.append("</svg>")
        csv_path.with_name(
            f"{csv_path.stem}-{backend}-{profile}-{updates}.svg"
        ).write_text("".join(svg))


def run_snapshot_sweep(args: argparse.Namespace) -> None:
    """Append completed grid points and regenerate plots after every point."""
    if args.backend == "postgres" and not _POSTGRES_AVAILABLE:
        raise RuntimeError("Postgres dependencies unavailable; no fallback is allowed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata = {
        "schema": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "backend": args.backend,
        "profiles": args.profiles,
        "frequencies": args.frequencies,
        "lengths": args.lengths,
        "script": "deterministic, no provider; IDs follow update index",
        "read_sampling": "one get_state after every invocation, prefix distribution",
        "write_sampling": "end-to-end graph.invoke including hydrate, scripted model and checkpoint I/O",
        "storage": "postgres pg_column_size row estimate (not relation disk bytes); memory serialized payloads across all stores",
        "max_supersteps_since_snapshot": os.environ.get(
            "LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT", "5000"
        ),
    }
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Resume metadata differs; use a new output path")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    existing = []
    if args.output.exists():
        with args.output.open() as handle:
            existing = list(csv.DictReader(handle))
    done = {(r["profile"], int(r["frequency"]), int(r["updates"])) for r in existing}
    for profile in args.profiles:
        for freq in args.frequencies:
            missing = [n for n in args.lengths if (profile, freq, n) not in done]
            if not missing:
                continue
            for row in _measure_sweep(freq, missing, profile, args.backend):
                fresh = not args.output.exists() or args.output.stat().st_size == 0
                with args.output.open("a", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(row), lineterminator="\n"
                    )
                    if fresh:
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                print(json.dumps(row), flush=True)
                _plot_sweep(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-sweep", action="store_true")
    parser.add_argument("--backend", choices=["memory", "postgres"], default="memory")
    parser.add_argument("--frequencies", type=int, nargs="+", default=FREQUENCIES)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--profiles", choices=PROFILES, nargs="+", default=PROFILES)
    parser.add_argument("--output", type=Path, default=Path("delta-sweep.csv"))
    args = parser.parse_args()
    if any(value <= 0 for value in args.frequencies + args.lengths):
        parser.error("frequencies and lengths must be positive")
    if args.snapshot_sweep:
        run_snapshot_sweep(args)
    else:
        run_benchmark()
    sys.exit(0)
