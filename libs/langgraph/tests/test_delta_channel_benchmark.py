"""Benchmark: DeltaChannel snapshot_frequency — storage vs. read-depth tradeoff.

Run directly:  python tests/test_delta_channel_benchmark.py
Run via pytest: pytest tests/test_delta_channel_benchmark.py -s

Part 1 — baseline (original): DeltaChannel(inf) vs add_messages (BinOp).
Part 2 — snapshot_frequency sweep: shows the storage/read-latency tradeoff
  across frequencies [1, 5, 10, 50, inf] at scale.

Key insight:
  snapshot_frequency=inf  → O(N) storage, O(N) read depth (pure delta)
  snapshot_frequency=N    → O(N²/N) storage, O(N) read depth bounded by freq
  snapshot_frequency=1    → O(N²) storage, O(1) read depth (full snapshot)
"""

from __future__ import annotations

import contextlib
import math
import sys
import time
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

try:
    from langgraph.checkpoint.postgres import PostgresSaver

    _POSTGRES_AVAILABLE = True
    _POSTGRES_URI = "postgres://sydney_runkle@localhost:5441/postgres?sslmode=disable"
except ImportError:
    _POSTGRES_AVAILABLE = False

# ---------------------------------------------------------------------------
# Realistic message payload (~100 tokens / ~400 chars each)
# ---------------------------------------------------------------------------

_HUMAN_TEMPLATE = (
    "I need help understanding the implications of {topic} on our system architecture. "
    "Specifically, I'm concerned about how this interacts with our existing {concern} "
    "and whether we need to refactor the {component} layer before proceeding. "
    "We've had prior incidents in this area and want to be deliberate. "
    "What should we prioritize first, and are there known failure modes we should design around from the start?"
)

_AI_TEMPLATE = (
    "Great question about {topic}. The key insight here is that {concern} introduces "
    "a subtle ordering dependency that most teams overlook until they hit it in production. "
    "For your {component} layer specifically, I'd recommend starting with a careful audit "
    "of the interface boundaries before making any structural changes. This will give you "
    "a clear picture of the blast radius and let you sequence the migration safely."
)

_TOPICS = [
    "distributed tracing",
    "eventual consistency",
    "schema migration",
    "backpressure handling",
    "idempotency guarantees",
    "cache invalidation",
    "connection pooling",
    "rate limiting",
    "circuit breaking",
    "observability pipelines",
]

_CONCERNS = [
    "concurrency model",
    "retry semantics",
    "state management",
    "error propagation",
    "latency budget",
]

_COMPONENTS = [
    "persistence",
    "routing",
    "ingestion",
    "aggregation",
    "serialization",
]


def _human_content(i: int) -> str:
    return _HUMAN_TEMPLATE.format(
        topic=_TOPICS[i % len(_TOPICS)],
        concern=_CONCERNS[i % len(_CONCERNS)],
        component=_COMPONENTS[i % len(_COMPONENTS)],
    )


def _ai_content(i: int) -> str:
    return _AI_TEMPLATE.format(
        topic=_TOPICS[i % len(_TOPICS)],
        concern=_CONCERNS[i % len(_CONCERNS)],
        component=_COMPONENTS[i % len(_COMPONENTS)],
    )


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------


class BinaryState(TypedDict):
    messages: Annotated[list, add_messages]


class DeltaState(TypedDict):
    messages: Annotated[list, DeltaChannel(list, add_messages)]


def _make_delta_state(snapshot_frequency: int | float) -> type:
    """Create a TypedDict with DeltaChannel at the given snapshot_frequency."""
    channel = DeltaChannel(list, add_messages, snapshot_frequency=snapshot_frequency)
    # Use the functional TypedDict form so the Annotated type is stored as an
    # already-evaluated object rather than a forward-reference string (which
    # would fail when get_type_hints tries to resolve 'snapshot_frequency').
    return TypedDict(  # type: ignore[return-value]
        f"DeltaState_freq{snapshot_frequency}",
        {"messages": Annotated[list, channel]},
    )


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def _make_graph(state_cls: type, checkpointer: Any = None) -> Any:
    def human_node(state: Any) -> dict:
        return {}

    def ai_node(state: Any) -> dict:
        i = len(state["messages"]) // 2
        return {"messages": [AIMessage(content=_ai_content(i), id=f"a{i}")]}

    g = StateGraph(state_cls)
    g.add_node("human", human_node)
    g.add_node("ai", ai_node)
    g.add_edge("human", "ai")
    g.add_edge("ai", END)
    g.set_entry_point("human")
    return g.compile(checkpointer=checkpointer or MemorySaver())


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _total_blob_bytes(saver: MemorySaver) -> int:
    total = 0
    for (_, _, _, _), (type_tag, blob) in saver.blobs.items():
        if blob is not None:
            total += len(blob)
    return total


def _run_turns(
    n_turns: int,
    state_cls: type,
    checkpointer: Any = None,
) -> tuple[float, float, int]:
    """Run n_turns conversation turns.

    Returns (write_elapsed_s, read_elapsed_s, total_blob_bytes).
    Read latency is the average of 5 get_state calls after the full history
    is built — forces state rehydration including ancestor replay if needed.
    """
    graph = _make_graph(state_cls, checkpointer)
    config = {"configurable": {"thread_id": "bench"}}

    t0 = time.perf_counter()
    for i in range(n_turns):
        graph.invoke(
            {"messages": [HumanMessage(content=_human_content(i), id=f"h{i}")]},
            config,
        )
    write_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(5):
        graph.get_state(config)
    read_elapsed = (time.perf_counter() - t1) / 5

    blob_bytes = (
        _total_blob_bytes(graph.checkpointer)
        if isinstance(graph.checkpointer, MemorySaver)
        else -1
    )
    return write_elapsed, read_elapsed, blob_bytes


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def _approx_tokens(n_turns: int) -> str:
    tokens = n_turns * 200
    if tokens >= 1_000_000:
        return f"~{tokens / 1_000_000:.1f}M tok"
    if tokens >= 1_000:
        return f"~{tokens / 1_000:.0f}K tok"
    return f"~{tokens} tok"


# ---------------------------------------------------------------------------
# Checkpointer factories
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _pg_saver(thread_id: str = "bench"):
    """Context manager that yields a fresh PostgresSaver and cleans up after."""
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
    """Return (label, saver_or_None) pairs for available checkpointers."""
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
# Part 1: baseline DeltaChannel(inf) vs add_messages
# ---------------------------------------------------------------------------

BASELINE_TURN_COUNTS = [10, 25, 50, 100, 500]
DELTA_ONLY_TURN_COUNTS = [1000]


def _run_baseline_for_checkpointer(cp_label: str, cp_hint: Any) -> None:
    W = 72

    def _make_saver():
        if cp_hint is None:
            return contextlib.nullcontext(None)
        return _pg_saver()

    rows: list[tuple[int, Any, Any, Any, Any, Any, Any]] = []
    for turns in BASELINE_TURN_COUNTS:
        with _make_saver() as saver:
            b_wt, b_rt, b_bytes = _run_turns(turns, BinaryState, saver)
        with _make_saver() as saver:
            d_wt, d_rt, d_bytes = _run_turns(turns, DeltaState, saver)
        rows.append((turns, b_bytes, d_bytes, b_rt, d_rt, b_wt, d_wt))
    for turns in DELTA_ONLY_TURN_COUNTS:
        with _make_saver() as saver:
            d_wt, d_rt, d_bytes = _run_turns(turns, DeltaState, saver)
        rows.append((turns, None, d_bytes, None, d_rt, None, d_wt))

    def _bytes_or_na(v: Any) -> str:
        if v is None or v < 0:
            return "n/a"
        return _fmt_bytes(v)

    def _ms_or_na(v: Any) -> str:
        return "n/a" if v is None else f"{v * 1000:.1f}ms"

    print(f"\n  [{cp_label}] Storage (blob bytes)")
    print(
        f"  {'turns':>6}  {'ctx':>10}  {'add_msgs':>12}  {'delta(inf)':>12}  {'savings':>8}"
    )
    print("  " + "-" * (W - 2))
    for turns, b_bytes, d_bytes, b_rt, d_rt, b_wt, d_wt in rows:
        if b_bytes is None or b_bytes < 0 or d_bytes is None or d_bytes < 0:
            ratio_str = "n/a"
        else:
            ratio = b_bytes / d_bytes if d_bytes else float("inf")
            ratio_str = f"{ratio:.0f}x"
        print(
            f"  {turns:>6}  {_approx_tokens(turns):>10}  "
            f"{_bytes_or_na(b_bytes):>12}  {_bytes_or_na(d_bytes):>12}  {ratio_str:>8}"
        )

    print(f"\n  [{cp_label}] Read latency (avg of 5 get_state calls)")
    print(f"  {'turns':>6}  {'ctx':>10}  {'add_msgs':>12}  {'delta(inf)':>12}")
    print("  " + "-" * (W - 2))
    for turns, b_bytes, d_bytes, b_rt, d_rt, b_wt, d_wt in rows:
        print(
            f"  {turns:>6}  {_approx_tokens(turns):>10}  "
            f"{_ms_or_na(b_rt):>12}  {_ms_or_na(d_rt):>12}"
        )


def run_baseline_benchmark() -> None:
    print()
    print("Part 1 — DeltaChannel(inf) vs add_messages: storage & latency")
    print("=" * 72)
    for cp_label, cp_hint in _checkpointers():
        _run_baseline_for_checkpointer(cp_label, cp_hint)
    print()


# ---------------------------------------------------------------------------
# Part 2: snapshot_frequency sweep
# ---------------------------------------------------------------------------

# Frequencies to test. 1 = always snapshot (like BinOp), inf = pure delta.
SNAPSHOT_FREQUENCIES: list[int | float] = [1, 5, 10, 50, math.inf]

# Turn counts for the sweep — high enough to show storage divergence.
SWEEP_TURN_COUNTS = [50, 100, 500]


def _freq_label(freq: int | float) -> str:
    if freq == math.inf:
        return "inf"
    return str(int(freq))


def _run_sweep_for_checkpointer(cp_label: str, cp_hint: Any) -> None:
    def _make_saver():
        if cp_hint is None:
            return contextlib.nullcontext(None)
        return _pg_saver()

    # Collect results: {turns: {freq_label: (write_s, read_s, bytes)}}
    results: dict[int, dict[str, tuple[float, float, int]]] = {}
    for turns in SWEEP_TURN_COUNTS:
        results[turns] = {}
        for freq in SNAPSHOT_FREQUENCIES:
            state_cls = _make_delta_state(freq)
            with _make_saver() as saver:
                wt, rt, bb = _run_turns(turns, state_cls, saver)
            results[turns][_freq_label(freq)] = (wt, rt, bb)

    freq_labels = [_freq_label(f) for f in SNAPSHOT_FREQUENCIES]
    col_w = 12

    header = f"  {'turns':>6}  {'ctx':>10}" + "".join(
        f"  {f'freq={freq_label}':>{col_w}}" for freq_label in freq_labels
    )

    print(f"\n  [{cp_label}] Storage (blob bytes) — lower is better")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for turns in SWEEP_TURN_COUNTS:
        row = f"  {turns:>6}  {_approx_tokens(turns):>10}"
        for label in freq_labels:
            _, _, bb = results[turns][label]
            row += f"  {_fmt_bytes(bb) if bb >= 0 else 'n/a':>{col_w}}"
        print(row)

    print(f"\n  [{cp_label}] Read latency (avg of 5 get_state) — lower is better")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for turns in SWEEP_TURN_COUNTS:
        row = f"  {turns:>6}  {_approx_tokens(turns):>10}"
        for label in freq_labels:
            _, rt, _ = results[turns][label]
            row += f"  {f'{rt * 1000:.1f}ms':>{col_w}}"
        print(row)

    print(
        f"\n  [{cp_label}] Per-invoke write latency (total / turns) — lower is better"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for turns in SWEEP_TURN_COUNTS:
        row = f"  {turns:>6}  {_approx_tokens(turns):>10}"
        for label in freq_labels:
            wt, _, _ = results[turns][label]
            row += f"  {f'{(wt / turns) * 1000:.1f}ms':>{col_w}}"
        print(row)


def run_snapshot_freq_benchmark() -> None:
    print()
    print("Part 2 — DeltaChannel snapshot_frequency sweep")
    print("Lower freq → fewer snapshots → less storage but deeper read replay")
    print("=" * 80)
    for cp_label, cp_hint in _checkpointers():
        _run_sweep_for_checkpointer(cp_label, cp_hint)
    print()
    print("Legend:")
    print(
        "  freq=1    snapshot every write (full blob always — same as add_messages / BinOp)"
    )
    print("  freq=N    snapshot every N writes; read walks at most N ancestor writes")
    print("  freq=inf  pure delta; read walks entire ancestor chain")
    print()


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="slow benchmark — run manually with: python tests/test_delta_channel_benchmark.py"
)
def test_delta_channel_baseline_benchmark(capsys: Any) -> None:
    """DeltaChannel(inf) uses less storage than add_messages at scale."""
    with capsys.disabled():
        run_baseline_benchmark()

    for turns in [25, 50]:
        _, _, b_bytes = _run_turns(turns, BinaryState)
        _, _, d_bytes = _run_turns(turns, DeltaState)
        assert d_bytes < b_bytes, (
            f"DeltaChannel should use less storage at {turns} turns, "
            f"got delta={d_bytes} binary={b_bytes}"
        )


@pytest.mark.skip(
    reason="slow benchmark — run manually with: python tests/test_delta_channel_benchmark.py"
)
def test_snapshot_freq_benchmark(capsys: Any) -> None:
    """snapshot_frequency trades storage for bounded read depth."""
    with capsys.disabled():
        run_snapshot_freq_benchmark()

    # Correctness: results at all frequencies should agree on final state.
    n_turns = 20
    states: dict[str, list] = {}
    for freq in SNAPSHOT_FREQUENCIES:
        state_cls = _make_delta_state(freq)
        graph = _make_graph(state_cls)
        config = {"configurable": {"thread_id": "correctness"}}
        for i in range(n_turns):
            graph.invoke(
                {"messages": [HumanMessage(content=_human_content(i), id=f"h{i}")]},
                config,
            )
        state = graph.get_state(config)
        states[_freq_label(freq)] = [m.id for m in state.values["messages"]]

    ref = states["inf"]
    for label, msg_ids in states.items():
        assert msg_ids == ref, (
            f"freq={label} produced different message IDs than freq=inf"
        )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_baseline_benchmark()
    run_snapshot_freq_benchmark()
    sys.exit(0)
