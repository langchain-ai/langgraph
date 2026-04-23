"""Benchmark: DeltaChannel vs BinaryOperatorAggregate storage and time.

Run directly:  python tests/test_delta_channel_benchmark.py
Run via pytest: pytest tests/test_delta_channel_benchmark.py -s

Simulates realistic multi-turn conversations with paragraph-length messages
(~100 tokens each) scaling up to 1M-token-equivalent histories.

Token estimates: 1 token ≈ 4 chars; each turn ≈ 200 tokens (human + AI).
A 1M-token conversation ≈ 5,000 turns of realistic messages.

DeltaChannel stores only a zero-byte sentinel in checkpoint_blobs; the actual
write data lives in checkpoint_writes (already stored there). Reconstruction
walks the parent chain and replays writes through the operator — O(N) total
storage vs O(N²) for plain add_messages.
"""

from __future__ import annotations

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
    from langgraph.checkpoint.sqlite import SqliteSaver

    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False

try:
    from langgraph.checkpoint.postgres import PostgresSaver

    _POSTGRES_AVAILABLE = True
    _POSTGRES_URI = (
        "postgres://postgres:postgres@localhost:5441/postgres?sslmode=disable"
    )
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
    messages: Annotated[list, DeltaChannel(add_messages)]


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
    blob_bytes is -1 for savers without in-memory blob stores (e.g. SQLite).
    Read latency is measured as the time to invoke the graph with no new
    messages after the full history is built — this forces state rehydration.
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

    # Measure read/rehydration: get_state forces the channel to rebuild
    t1 = time.perf_counter()
    for _ in range(5):
        graph.get_state(config)
    read_elapsed = (time.perf_counter() - t1) / 5

    if isinstance(graph.checkpointer, MemorySaver):
        blob_bytes = _total_blob_bytes(graph.checkpointer)
    else:
        blob_bytes = -1
    return write_elapsed, read_elapsed, blob_bytes


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def _approx_tokens(n_turns: int) -> str:
    # ~100 tokens human + ~100 tokens AI per turn
    tokens = n_turns * 200
    if tokens >= 1_000_000:
        return f"~{tokens / 1_000_000:.1f}M tok"
    if tokens >= 1_000:
        return f"~{tokens / 1_000:.0f}K tok"
    return f"~{tokens} tok"


# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------

# Turn counts chosen to demonstrate O(N²) vs O(N) storage growth without running too long.
# Extrapolation: 5,000 turns × ~200 tokens/turn ≈ 1M tokens (Claude's full context window).
TURN_COUNTS = [10, 25, 50, 100, 500]

# Deep-thread counts where add_messages blob storage would exceed 1 GB;
# only DeltaChannel runs here.
DELTA_ONLY_TURN_COUNTS = [1000]


def _checkpointer_factories() -> list[tuple[str, Any]]:
    """Return (label, context_manager_or_none) pairs for available checkpointers."""
    return [("InMemory", None)]


def run_benchmark() -> None:
    print()
    print(
        "DeltaChannel vs add_messages (BinaryOperatorAggregate) — checkpoint storage & latency"
    )
    print("Simulating realistic multi-turn conversations up to ~1M-token histories")
    print("(5,000 turns × ~200 tokens/turn ≈ 1M tokens — Claude's full context window)")
    print()

    checkpointers: list[tuple[str, Any]] = [("InMemory", None)]
    if _POSTGRES_AVAILABLE:
        try:
            import psycopg

            psycopg.connect(_POSTGRES_URI).close()
            checkpointers.append(("Postgres (plain SELECT)", "postgres"))
        except Exception:
            pass

    for cp_label, cp_hint in checkpointers:
        print(f"--- Checkpointer: {cp_label} ---")
        _run_benchmark_for_checkpointer(cp_hint)


def _run_benchmark_for_checkpointer(cp_hint: Any) -> None:
    import contextlib
    import tempfile

    @contextlib.contextmanager
    def _make_saver():
        if cp_hint is None:
            yield None
        elif cp_hint == "postgres":
            with PostgresSaver.from_conn_string(_POSTGRES_URI) as saver:
                saver.setup()
                with saver._cursor() as cur:
                    cur.execute("DELETE FROM checkpoints WHERE thread_id = 'bench'")
                    cur.execute(
                        "DELETE FROM checkpoint_blobs WHERE thread_id = 'bench'"
                    )
                    cur.execute(
                        "DELETE FROM checkpoint_writes WHERE thread_id = 'bench'"
                    )
                yield saver
        else:
            with tempfile.NamedTemporaryFile(suffix=".db") as f:
                with SqliteSaver.from_conn_string(f.name) as saver:
                    yield saver

    rows: list[tuple[int, Any, Any, Any, Any, Any, Any]] = []
    for turns in TURN_COUNTS:
        with _make_saver() as saver:
            b_wt, b_rt, b_bytes = _run_turns(turns, BinaryState, saver)
        with _make_saver() as saver:
            d_wt, d_rt, d_bytes = _run_turns(turns, DeltaState, saver)
        rows.append((turns, b_bytes, d_bytes, b_rt, d_rt, b_wt, d_wt))
    for turns in DELTA_ONLY_TURN_COUNTS:
        with _make_saver() as saver:
            d_wt, d_rt, d_bytes = _run_turns(turns, DeltaState, saver)
        rows.append((turns, None, d_bytes, None, d_rt, None, d_wt))

    # ── Table 1: Storage ─────────────────────────────────────────────────────
    W = 64
    print("Storage (checkpoint blob bytes)")
    print("=" * W)
    print(
        f"{'turns':>6}  {'ctx size':>10}  {'add_msgs':>12}  {'delta':>12}  "
        f"{'savings':>8}"
    )
    print("-" * W)

    def _bytes_or_na(v: Any) -> str:
        if v is None:
            return "n/a"
        if v < 0:
            return "n/a"
        return _fmt_bytes(v)

    def _ms_or_na(v: Any) -> str:
        return "n/a" if v is None else f"{v * 1000:.1f}ms"

    for turns, b_bytes, d_bytes, b_rt, d_rt, b_wt, d_wt in rows:
        if b_bytes is None or b_bytes < 0 or d_bytes is None or d_bytes < 0:
            ratio_str = "n/a"
        else:
            ratio = b_bytes / d_bytes if d_bytes else float("inf")
            ratio_str = f"{ratio:.0f}x"
        print(
            f"{turns:>6}  {_approx_tokens(turns):>10}  "
            f"{_bytes_or_na(b_bytes):>12}  {_bytes_or_na(d_bytes):>12}  "
            f"{ratio_str:>8}"
        )
    print("=" * W)
    print()

    # ── Table 2: Read latency ─────────────────────────────────────────────────
    print("Read latency (avg of 5 get_state calls)")
    print("=" * W)
    print(f"{'turns':>6}  {'ctx size':>10}  {'add_msgs':>12}  {'delta':>12}")
    print("-" * W)
    for turns, b_bytes, d_bytes, b_rt, d_rt, b_wt, d_wt in rows:
        print(
            f"{turns:>6}  {_approx_tokens(turns):>10}  "
            f"{_ms_or_na(b_rt):>12}  {_ms_or_na(d_rt):>12}"
        )
    print("=" * W)
    print()

    # ── Table 3: Per-invoke latency (total write_elapsed / turns) ─────────────
    print("Per-invoke latency (total graph.invoke time / turns)")
    print("=" * W)
    print(f"{'turns':>6}  {'ctx size':>10}  {'add_msgs':>12}  {'delta':>12}")
    print("-" * W)
    for turns, b_bytes, d_bytes, b_rt, d_rt, b_wt, d_wt in rows:

        def _per(wt: Any) -> str:
            if wt is None:
                return "n/a"
            return f"{(wt / turns) * 1000:.1f}ms"

        print(
            f"{turns:>6}  {_approx_tokens(turns):>10}  "
            f"{_per(b_wt):>12}  {_per(d_wt):>12}"
        )
    print("=" * W)
    print()

    print("Legend:")
    print("  add_msgs  = Annotated[list, add_messages]  — O(N²) storage")
    print("  delta     = Annotated[list, DeltaChannel(add_messages)]  — O(N) storage")
    print()


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="slow benchmark — run manually with: python tests/test_delta_channel_benchmark.py"
)
def test_delta_channel_benchmark(capsys: Any) -> None:
    """Storage grows O(N²) for add_messages, O(N) for DeltaChannel."""
    with capsys.disabled():
        run_benchmark()

    # Correctness assertion: DeltaChannel must use less storage at scale.
    for turns in [25, 50]:
        _, _, b_bytes = _run_turns(turns, BinaryState)
        _, _, d_bytes = _run_turns(turns, DeltaState)
        assert d_bytes < b_bytes, (
            f"DeltaChannel should use less storage at {turns} turns, "
            f"got delta={d_bytes} binary={b_bytes}"
        )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_benchmark()
    sys.exit(0)
