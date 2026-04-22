"""Benchmark: add_messages fast-path optimizations.

Both implementations are inlined so the benchmark is self-contained and
immune to import-cache or installed-vs-local confusion.

Run directly:
    python tests/test_add_messages_benchmark.py

Or via pytest (correctness only, numbers printed to stdout):
    pytest tests/test_add_messages_benchmark.py -s -v
"""

import statistics
import time
import tracemalloc
import uuid
from typing import cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)

from langgraph.graph.message import REMOVE_ALL_MESSAGES

# ── original implementation (pre-optimisation) ────────────────────────────────


def _add_messages_original(left, right):
    remove_all_idx = None
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    left = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(left)
    ]
    right = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(right)
    ]
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for idx, m in enumerate(right):
        if m.id is None:
            m.id = str(uuid.uuid4())
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx
    if remove_all_idx is not None:
        return right[remove_all_idx + 1 :]
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )
            merged_by_id[m.id] = len(merged)
            merged.append(m)
    return [m for m in merged if m.id not in ids_to_remove]


# ── optimised implementation ──────────────────────────────────────────────────


def _add_messages_optimized(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]

    # Optimisation 1: skip conversion + ID assignment on left when it already
    # contains fully-resolved BaseMessage objects (the common case after the
    # first call, since add_messages always returns list[BaseMessage] with IDs).
    if (
        left
        and isinstance(left[0], BaseMessage)
        and not isinstance(left[0], BaseMessageChunk)
    ):
        left = cast(list[BaseMessage], left)
    else:
        left = [
            message_chunk_to_message(cast(BaseMessageChunk, m))
            for m in convert_to_messages(left)
        ]
        for m in left:
            if m.id is None:
                m.id = str(uuid.uuid4())

    # always normalise right — it's fresh external input
    right = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(right)
    ]
    remove_all_idx = None
    has_remove = False
    for idx, m in enumerate(right):
        if m.id is None:
            m.id = str(uuid.uuid4())
        if isinstance(m, RemoveMessage):
            has_remove = True
            if m.id == REMOVE_ALL_MESSAGES:
                remove_all_idx = idx

    if remove_all_idx is not None:
        return right[remove_all_idx + 1 :]

    # Optimisation 2: pure-append fast path — no removals and no ID overlaps.
    # Builds one set over left instead of copying left + building a full dict.
    if not has_remove:
        left_ids = {m.id for m in left}
        if not any(m.id in left_ids for m in right):
            return left + right

    # slow path: updates or removals present — full indexed merge
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )
            merged_by_id[m.id] = len(merged)
            merged.append(m)
    return [m for m in merged if m.id not in ids_to_remove]


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_messages(n: int) -> list[BaseMessage]:
    return [
        (HumanMessage if i % 2 == 0 else AIMessage)(
            content=f"message {i}", id=str(uuid.uuid4())
        )
        for i in range(n)
    ]


def _bench_time(fn, left, right, *, iters: int = 2_000) -> float:
    """Return median latency in microseconds."""
    for _ in range(100):
        fn(list(left), list(right))
    times = []
    for _ in range(iters):
        left_copy, right_copy = list(left), list(right)
        t0 = time.perf_counter()
        fn(left_copy, right_copy)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e6


def _bench_memory(fn, left, right) -> int:
    """Return peak memory allocated during a single call (bytes)."""
    # one warm-up so any lazy init is excluded
    fn(list(left), list(right))
    left_copy, right_copy = list(left), list(right)
    tracemalloc.start()
    tracemalloc.clear_traces()
    fn(left_copy, right_copy)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


# ── scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    ("pure append  1 → 1  msg", 1, 1, "append"),
    ("pure append  10 → 1  msg", 10, 1, "append"),
    ("pure append  100 → 1  msg", 100, 1, "append"),
    ("pure append  1000 → 1  msg", 1000, 1, "append"),
    ("pure append  1000 → 5  msgs", 1000, 5, "append"),
    ("update existing  100 → 1  msg", 100, 1, "update"),
    ("remove message  100 → 1  msg", 100, 1, "remove"),
]


def _make_inputs(n_left, n_right, mode):
    left = _make_messages(n_left)
    right = _make_messages(n_right)
    if mode == "update":
        right[0] = AIMessage(content="updated", id=left[0].id)
    elif mode == "remove":
        right = [RemoveMessage(id=left[0].id)]
    return left, right


# ── main output ───────────────────────────────────────────────────────────────

COL = 36


def run_benchmarks() -> None:
    print()
    print("=" * 88)
    print("add_messages benchmark — time (µs, median of 2 000 iterations)")
    print("=" * 88)
    print(f"{'Scenario':<{COL}} {'Original':>10} {'Optimized':>11} {'Speedup':>8}")
    print("-" * 88)

    for label, n_left, n_right, mode in SCENARIOS:
        left, right = _make_inputs(n_left, n_right, mode)
        t_orig = _bench_time(_add_messages_original, left, right)
        t_opt = _bench_time(_add_messages_optimized, left, right)
        print(f"{label:<{COL}} {t_orig:>10.2f} {t_opt:>11.2f} {t_orig / t_opt:>7.2f}x")

    print()
    print("=" * 88)
    print("add_messages benchmark — peak memory allocated per call (bytes)")
    print("=" * 88)
    print(f"{'Scenario':<{COL}} {'Original':>10} {'Optimized':>11} {'Reduction':>10}")
    print("-" * 88)

    for label, n_left, n_right, mode in SCENARIOS:
        left, right = _make_inputs(n_left, n_right, mode)
        m_orig = _bench_memory(_add_messages_original, left, right)
        m_opt = _bench_memory(_add_messages_optimized, left, right)
        reduction = (1 - m_opt / m_orig) * 100 if m_orig else 0.0
        print(f"{label:<{COL}} {m_orig:>10,} {m_opt:>11,} {reduction:>9.1f}%")

    print()
    print("=" * 88)
    print("Simulated long thread — 200 steps × 2 msgs appended per step")
    print("=" * 88)
    for name, fn in [
        ("original", _add_messages_original),
        ("optimized", _add_messages_optimized),
    ]:
        state: list = []
        t0 = time.perf_counter()
        for step in range(200):
            new_msgs = [
                HumanMessage(content=f"step {step} human", id=str(uuid.uuid4())),
                AIMessage(content=f"step {step} ai", id=str(uuid.uuid4())),
            ]
            state = fn(state, new_msgs)
        elapsed = (time.perf_counter() - t0) * 1_000
        print(f"  {name:<12}  {elapsed:.2f} ms  ({len(state)} messages)")
    print()


# ── pytest entry-points ───────────────────────────────────────────────────────


def test_add_messages_correctness():
    """Optimised implementation must match original output for every scenario."""
    for label, n_left, n_right, mode in SCENARIOS:
        left, right = _make_inputs(n_left, n_right, mode)
        expected = _add_messages_original(list(left), list(right))
        actual = _add_messages_optimized(list(left), list(right))
        assert len(actual) == len(expected), f"[{label}] length mismatch"
        for a, b in zip(actual, expected):
            assert type(a) is type(b), f"[{label}] type mismatch"
            assert a.id == b.id, f"[{label}] id mismatch"
            assert a.content == b.content, f"[{label}] content mismatch"


def test_add_messages_benchmark(capsys):
    run_benchmarks()
    out = capsys.readouterr().out
    assert "Speedup" in out
    assert "Optimized" in out


if __name__ == "__main__":
    run_benchmarks()
