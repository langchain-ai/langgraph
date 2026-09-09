"""The per-task scratchpad resume lookups can be resolved either by scanning
`pending_writes` or from values computed once per superstep (#8855).

Both paths must agree in every case - the precomputed path is only an
optimization, never a behaviour change.
"""
import pytest

from langgraph.pregel import _algo
from langgraph.pregel._algo import _scratchpad

RESUME = _algo.RESUME
NULL_TASK_ID = _algo.NULL_TASK_ID


def _precompute(writes):
    """Mirror of the hoisted pass in `prepare_next_tasks`."""
    null_w = None
    resume_ws = []
    for w in writes:
        if w[1] == RESUME:
            resume_ws.append(w)
            if w[0] == NULL_TASK_ID and null_w is None:
                null_w = w
    return null_w, resume_ws


CASES = {
    "empty": [],
    "no_resume_writes": [("task0", "chan", 1), ("task1", "chan", 2)],
    "global_only": [(NULL_TASK_ID, RESUME, "global")],
    "task_only": [("task0", RESUME, "r0"), ("task1", RESUME, "r1")],
    "mixed": (
        [(f"task{i}", "chan", i) for i in range(50)]
        + [(NULL_TASK_ID, RESUME, "global"), ("task0", RESUME, "r0")]
    ),
    "list_values": [(NULL_TASK_ID, RESUME, ["a", "b"]), ("task1", RESUME, ["x"])],
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("resume_map", [None, {"h": "mapped"}, {"other": "mapped"}])
def test_precomputed_matches_scan(name, resume_map):
    writes = CASES[name]
    null_w, resume_ws = _precompute(writes)

    for task_id in ("task0", "task1", "no-such-task"):
        # fresh copies: consuming a resume value mutates pending_writes
        scanned = _scratchpad(None, list(writes), task_id, "h", resume_map, 0, 1)
        precomputed = _scratchpad(
            None,
            list(writes),
            task_id,
            "h",
            resume_map,
            0,
            1,
            null_resume_write=null_w,
            resume_writes=resume_ws,
        )
        assert precomputed.resume == scanned.resume, f"{name}/{task_id}"
        assert precomputed.get_null_resume() == scanned.get_null_resume()
        assert precomputed.get_null_resume(True) == scanned.get_null_resume(True)


def test_resume_map_applies_even_without_resume_writes():
    """`resume_map` is applied whenever pending writes exist, not only when a
    RESUME write is present - the precomputed path must preserve that."""
    writes = [("task0", "chan", 1)]
    null_w, resume_ws = _precompute(writes)
    assert resume_ws == []

    scanned = _scratchpad(None, list(writes), "task0", "h", {"h": "mapped"}, 0, 1)
    precomputed = _scratchpad(
        None,
        list(writes),
        "task0",
        "h",
        {"h": "mapped"},
        0,
        1,
        null_resume_write=null_w,
        resume_writes=resume_ws,
    )
    assert scanned.resume == ["mapped"]
    assert precomputed.resume == scanned.resume
