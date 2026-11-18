"""TRANSCRIPT capability tests: entries stored by `aput`, read by `aget_transcript`.

The cases live in `transcript_cases.json` so that stores written in other
languages can run the same ones. Each case runs against fresh ids. A `put` step
stores one checkpoint whose `messages` channel holds a `_TranscriptWrite`; a
`read` step compares values, and `expect_later` compares `has_later_entries`.
An entry `{"seq": n, "text": s}` has `value = s`.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, TranscriptConflictError
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import (
    _TranscriptEntry,
    _TranscriptRef,
    _TranscriptWrite,
)

from langgraph.checkpoint.conformance.test_utils import (
    generate_checkpoint,
    generate_metadata,
)

CASES: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "transcript_cases.json").read_text()
)


async def _run_case(saver: BaseCheckpointSaver, case: dict[str, Any]) -> None:
    names: dict[tuple[str, str], str] = {}

    def ident(kind: str, name: str) -> str:
        if (kind, name) not in names:
            names[kind, name] = (
                str(uuid6(clock_seq=-1)) if kind == "checkpoint" else str(uuid4())
            )
        return names[kind, name]

    def config(thread: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": ident("thread", thread), "checkpoint_ns": ""}
        }

    for i, step in enumerate(case["steps"]):
        if "put" in step:
            put = step["put"]
            writes = [
                _TranscriptWrite(
                    transcript_id=ident("transcript", w["transcript"]),
                    head=w["head"],
                    entries=[
                        _TranscriptEntry(e["seq"], e["text"]) for e in w["entries"]
                    ],
                    parent_transcript_id=ident("transcript", w["parent"])
                    if "parent" in w
                    else None,
                    fork_seq=w.get("fork_seq"),
                )
                for w in put["writes"]
            ]
            values = {f"messages{n or ''}": w for n, w in enumerate(writes)}
            checkpoint = generate_checkpoint(
                checkpoint_id=ident("checkpoint", put["checkpoint"]),
                channel_values=values,
                channel_versions={k: 1 for k in values},
            )
            try:
                await saver.aput(
                    config(put["thread"]),
                    checkpoint,
                    generate_metadata(step=i),
                    {k: 1 for k in values},
                )
            except TranscriptConflictError:
                assert step.get("expect") == "conflict", (
                    f"step {i}: unexpected conflict"
                )
            else:
                assert step.get("expect") != "conflict", (
                    f"step {i}: expected a conflict"
                )
        else:
            read = step["read"]
            result = await saver.aget_transcript(
                config=config(read["thread"]),
                ref=_TranscriptRef(
                    ident("transcript", read["transcript"]), read["head"]
                ),
            )
            got = [e.value for e in result.entries]
            assert got == step["expect"], (
                f"step {i}: expected {step['expect']}, got {got}"
            )
            seqs = [e.seq for e in result.entries]
            assert seqs == list(range(1, len(seqs) + 1)), f"step {i}: seqs {seqs}"
            if "expect_later" in step:
                assert result.has_later_entries == step["expect_later"], (
                    f"step {i}: has_later_entries is {result.has_later_entries}"
                )


async def test_pointer_is_stored_in_place_of_the_write(
    saver: BaseCheckpointSaver,
) -> None:
    """The checkpoint read back holds a `_TranscriptRef`, not the entries."""
    cfg: RunnableConfig = {
        "configurable": {"thread_id": str(uuid4()), "checkpoint_ns": ""}
    }
    write = _TranscriptWrite("t", 1, [_TranscriptEntry(1, "a")])
    checkpoint = generate_checkpoint(
        channel_values={"messages": write}, channel_versions={"messages": 1}
    )
    stored = await saver.aput(cfg, checkpoint, generate_metadata(), {"messages": 1})
    tup = await saver.aget_tuple(stored)
    assert tup is not None
    ref = tup.checkpoint["channel_values"]["messages"]
    assert ref == _TranscriptRef("t", 1), f"expected a pointer to (t, 1), got {ref!r}"


def _case_test(case: dict[str, Any]) -> Callable[[BaseCheckpointSaver], Any]:
    async def test(saver: BaseCheckpointSaver) -> None:
        await _run_case(saver, case)

    test.__name__ = f"test_{case['name']}"
    return test


ALL_TRANSCRIPT_TESTS = [
    *(_case_test(case) for case in CASES),
    test_pointer_is_stored_in_place_of_the_write,
]


async def run_transcript_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all transcript tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_TRANSCRIPT_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("transcript", test_fn.__name__, True, None)
        except Exception:
            failed += 1
            failures.append(f"{test_fn.__name__}: {traceback.format_exc()}")
            if on_test_result:
                on_test_result(
                    "transcript", test_fn.__name__, False, traceback.format_exc()
                )
    return passed, failed, failures
