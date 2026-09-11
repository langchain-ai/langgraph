"""WRITE_OWNERSHIP capability tests -- aclaim_write_ownership(thread_id).

The failure these clauses describe passes every other capability in this suite.

Two processes, one `thread_id`, `durability="sync"`. Worker A is frozen inside a model
call; worker B resumes the thread with `invoke(None, config)` and finishes; then A is
thawed and allowed to write. Both ran the same task at the same checkpoint, so both call
`put_writes` with the same deterministic `task_id`, and `INSERT ... ON CONFLICT DO
NOTHING` makes that first-writer-wins -- A's write is discarded while A reports success.
A then writes its own child of that checkpoint, so the chain forks, and because
checkpoint ids are time-ordered A's branch (written last) becomes the tip. B's terminal
checkpoint is now unreachable and the answer B returned to its caller contradicts the
thread. Both processes exit 0. Nothing raises and nothing logs.

Every clause below is a property that failure violates. `test_..._cannot_fork_the_chain`
is that scenario reduced to three writes and is the one to read first.

The contract, its two deadlines, and why the refusal is allowed to be deferred are in
`langgraph.checkpoint.conformance.ownership`. Read that module before changing an
assertion here; several of these shapes are the way they are because the obvious version
excludes a real implementation.

Reads go through `saver` -- the factory instance -- never through a claimed writer, for
two reasons. `aclaim_write_ownership` is only required to return something that can
write, so a writer may expose no read methods at all; and a saver whose refusal is a
server-side error may leave the superseded owner's connection unusable, which would turn
a read through it into a different failure than the one under test.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph.checkpoint.conformance.ownership import (
    is_stale_write_owner_rejection,
    observe_rejection,
)
from langgraph.checkpoint.conformance.test_utils import (
    generate_checkpoint,
    generate_config,
    generate_metadata,
)

# --------------------------------------------------------------------- helpers ---


async def _put(writer, tid: str, *, parent_id: str | None = None, step: int = 0):
    """Write one checkpoint to *tid* through *writer*. Returns the stored config."""
    config = generate_config(tid)
    if parent_id is not None:
        config["configurable"]["checkpoint_id"] = parent_id
    return await writer.aput(config, generate_checkpoint(), generate_metadata(step=step), {})


async def _checkpoint_ids(saver: BaseCheckpointSaver, tid: str) -> list[str]:
    """Every checkpoint id on *tid*, newest first (the order `alist` yields)."""
    return [tup.checkpoint["id"] async for tup in saver.alist(generate_config(tid))]


async def _tip_id(saver: BaseCheckpointSaver, tid: str) -> str | None:
    """The checkpoint a resume would load -- what `aget_tuple` without an id returns."""
    tup = await saver.aget_tuple(generate_config(tid))
    return None if tup is None else tup.checkpoint["id"]


async def _pending_writes(saver: BaseCheckpointSaver, tid: str, checkpoint_id: str):
    """`pending_writes` at a specific checkpoint, or `[]`."""
    tup = await saver.aget_tuple(generate_config(tid, checkpoint_id=checkpoint_id))
    if tup is None or tup.pending_writes is None:
        return []
    return list(tup.pending_writes)


# ----------------------------------------------------------------- the clauses ---


async def test_claim_returns_a_writer(saver: BaseCheckpointSaver) -> None:
    """A claim yields something that can write."""
    writer = await saver.aclaim_write_ownership(str(uuid4()))
    assert writer is not None, "aclaim_write_ownership returned None"
    for method in ("aput", "aput_writes"):
        assert callable(getattr(writer, method, None)), (
            f"the object returned by aclaim_write_ownership has no callable {method!r}; "
            f"the suite writes only through it"
        )


async def test_current_owner_can_put(saver: BaseCheckpointSaver) -> None:
    """The current owner's checkpoint is durable. Ownership must not cost liveness."""
    tid = str(uuid4())
    owner = await saver.aclaim_write_ownership(tid)
    stored = await _put(owner, tid)

    cp_id = stored["configurable"]["checkpoint_id"]
    assert await _tip_id(saver, tid) == cp_id, (
        f"the current owner's checkpoint is not the tip: "
        f"expected {cp_id}, got {await _tip_id(saver, tid)}"
    )


async def test_current_owner_can_put_writes(saver: BaseCheckpointSaver) -> None:
    """The current owner's pending write is durable."""
    tid = str(uuid4())
    owner = await saver.aclaim_write_ownership(tid)
    stored = await _put(owner, tid)
    await owner.aput_writes(stored, [("ch", "owned")], str(uuid4()))

    writes = await _pending_writes(saver, tid, stored["configurable"]["checkpoint_id"])
    assert len(writes) == 1, f"expected 1 pending write, got {len(writes)}"
    assert writes[0][2] == "owned", f"value mismatch: {writes[0][2]!r}"


async def test_superseded_owner_put_is_rejected(saver: BaseCheckpointSaver) -> None:
    """A superseded owner learns it lost, at the write or at the barrier.

    This is the clause a silent short-circuit fails. Refusing the write is not enough:
    a loser that is not told keeps running and reports success to its caller.
    """
    tid = str(uuid4())
    stale = await saver.aclaim_write_ownership(tid)
    await _put(stale, tid)
    await saver.aclaim_write_ownership(tid)  # supersedes `stale`

    await observe_rejection(stale, lambda: _put(stale, tid, step=1), what="aput")


async def test_superseded_owner_put_is_not_durable(saver: BaseCheckpointSaver) -> None:
    """The refused checkpoint reached no storage. The rejection is not advisory."""
    tid = str(uuid4())
    stale = await saver.aclaim_write_ownership(tid)
    first = await _put(stale, tid)
    await saver.aclaim_write_ownership(tid)

    before = await _checkpoint_ids(saver, tid)
    try:
        await observe_rejection(stale, lambda: _put(stale, tid, step=1), what="aput")
    except AssertionError:
        raise  # a silent or unrecognisable refusal is test_..._is_rejected's business
    after = await _checkpoint_ids(saver, tid)

    assert after == before, (
        f"a superseded owner's aput changed durable state: {before} -> {after}"
    )
    assert after == [first["configurable"]["checkpoint_id"]], (
        f"expected only the pre-takeover checkpoint to survive, got {after}"
    )


async def test_superseded_owner_cannot_fork_the_chain(
    saver: BaseCheckpointSaver,
) -> None:
    """The reproduction, in three writes.

    A owns the thread and writes `c1`. B takes over and writes `c2` as a child of `c1`.
    A -- still holding `c1` as its in-memory tip, exactly as a thawed worker does --
    writes its own child of `c1`. Unfenced, that child is written last, so its
    time-ordered id makes it the tip and B's `c2` becomes unreachable.

    Afterwards the tip must still be `c2`, and the thread must hold exactly the two
    checkpoints that were legitimately written.
    """
    tid = str(uuid4())

    a = await saver.aclaim_write_ownership(tid)
    c1 = (await _put(a, tid, step=0))["configurable"]["checkpoint_id"]

    b = await saver.aclaim_write_ownership(tid)
    c2 = (await _put(b, tid, parent_id=c1, step=1))["configurable"]["checkpoint_id"]
    assert await _tip_id(saver, tid) == c2, "setup failed: B's checkpoint is not the tip"

    await observe_rejection(
        a, lambda: _put(a, tid, parent_id=c1, step=1), what="a forking aput"
    )

    tip = await _tip_id(saver, tid)
    assert tip == c2, (
        f"the chain forked: the tip is {tip}, not the current owner's checkpoint {c2}. "
        f"A resume would load a superseded worker's state and the current owner's "
        f"terminal checkpoint is unreachable."
    )
    assert sorted(await _checkpoint_ids(saver, tid)) == sorted([c1, c2]), (
        f"unexpected checkpoints on the thread: {await _checkpoint_ids(saver, tid)}, "
        f"expected exactly {[c1, c2]}"
    )


async def test_superseded_owner_put_writes_is_rejected(
    saver: BaseCheckpointSaver,
) -> None:
    """A superseded owner learns it lost on the `aput_writes` path too.

    Both write paths need the clause: `put_writes` is where the deterministic `task_id`
    collision happens, and it is the path whose loss is silent today.
    """
    tid = str(uuid4())
    stale = await saver.aclaim_write_ownership(tid)
    first = await _put(stale, tid)
    await saver.aclaim_write_ownership(tid)

    await observe_rejection(
        stale,
        lambda: stale.aput_writes(first, [("ch", "stale")], str(uuid4())),
        what="aput_writes",
    )


async def test_superseded_owner_put_writes_is_not_durable(
    saver: BaseCheckpointSaver,
) -> None:
    """A superseded owner's write is absent even when nothing collides with it.

    A fresh `task_id`, so first-writer-wins cannot be what hides it: an unfenced saver
    stores this row and the transcript then contains a message from a worker that no
    longer owns the thread.
    """
    tid = str(uuid4())
    stale = await saver.aclaim_write_ownership(tid)
    first = await _put(stale, tid)
    cp_id = first["configurable"]["checkpoint_id"]
    await saver.aclaim_write_ownership(tid)

    await observe_rejection(
        stale,
        lambda: stale.aput_writes(first, [("ch", "stale")], str(uuid4())),
        what="aput_writes",
    )

    writes = await _pending_writes(saver, tid, cp_id)
    assert not any(w[2] == "stale" for w in writes), (
        f"a superseded owner's write is durable at {cp_id}: {writes}"
    )


async def test_superseded_owner_cannot_displace_the_current_owners_write(
    saver: BaseCheckpointSaver,
) -> None:
    """Same checkpoint, same `task_id`, both owners -- the current one's value survives.

    This is the collision itself. The two workers ran the same task at the same
    checkpoint, so their `task_id`s are equal by construction; whichever statement the
    saver chooses for the conflict, the value that ends up stored must be the current
    owner's.
    """
    tid = str(uuid4())
    stale = await saver.aclaim_write_ownership(tid)
    first = await _put(stale, tid)
    cp_id = first["configurable"]["checkpoint_id"]

    current = await saver.aclaim_write_ownership(tid)
    task_id = str(uuid4())
    await current.aput_writes(first, [("ch", "current")], task_id)

    await observe_rejection(
        stale,
        lambda: stale.aput_writes(first, [("ch", "stale")], task_id),
        what="a colliding aput_writes",
    )

    writes = await _pending_writes(saver, tid, cp_id)
    values = [w[2] for w in writes]
    assert values == ["current"], (
        f"expected the current owner's value alone at {cp_id}, got {values}"
    )


async def test_superseded_owner_cannot_displace_a_special_channel_write(
    saver: BaseCheckpointSaver,
) -> None:
    """The same collision on a channel whose conflict resolution is last-writer-wins.

    `WRITES_IDX_MAP`'s channels -- `__error__`, `__scheduled__`, `__interrupt__`,
    `__resume__` -- are special in the base package, and a saver may reasonably treat a
    write to them as *replacing* rather than duplicating: reserved negative indices mean
    the later write is the truer one. At least one bundled saver switches statement on
    exactly this condition, from a first-writer-wins insert to an upsert, whenever every
    channel in the batch is one of these four.

    Which makes this the one collision where a superseded owner can overwrite live data
    rather than merely be dropped, and `__resume__` is the worst of the four to lose: it
    carries the answer a human gave to an interrupt. So the clause is separate from the
    ordinary-channel one, and named, because the two fail for different reasons.
    """
    tid = str(uuid4())
    stale = await saver.aclaim_write_ownership(tid)
    first = await _put(stale, tid)
    cp_id = first["configurable"]["checkpoint_id"]

    current = await saver.aclaim_write_ownership(tid)
    task_id = str(uuid4())
    await current.aput_writes(first, [("__resume__", "current")], task_id)

    await observe_rejection(
        stale,
        lambda: stale.aput_writes(first, [("__resume__", "stale")], task_id),
        what="a colliding aput_writes on __resume__",
    )

    values = [w[2] for w in await _pending_writes(saver, tid, cp_id) if w[1] == "__resume__"]
    assert values == ["current"], (
        f"a superseded owner overwrote __resume__ at {cp_id}: expected ['current'], "
        f"got {values}"
    )


async def test_superseded_owner_stays_superseded(saver: BaseCheckpointSaver) -> None:
    """The refusal is not one-shot: a second attempt is refused the same way.

    A saver whose fence is a one-time flag, or whose first rejection leaves it unable to
    classify the second, fails here rather than in production.
    """
    tid = str(uuid4())
    stale = await saver.aclaim_write_ownership(tid)
    first = await _put(stale, tid)
    await saver.aclaim_write_ownership(tid)

    await observe_rejection(stale, lambda: _put(stale, tid, step=1), what="aput")
    await observe_rejection(
        stale, lambda: _put(stale, tid, step=2), what="a second aput"
    )

    assert await _checkpoint_ids(saver, tid) == [first["configurable"]["checkpoint_id"]], (
        "a repeatedly-refused owner still changed durable state"
    )


async def test_ownership_is_per_thread(saver: BaseCheckpointSaver) -> None:
    """Claiming one thread does not supersede another thread's owner.

    Ownership scoped too broadly -- per saver, per connection, per process -- passes
    every clause above and breaks every deployment that runs more than one run at a
    time.
    """
    tid1, tid2 = str(uuid4()), str(uuid4())
    owner1 = await saver.aclaim_write_ownership(tid1)
    await saver.aclaim_write_ownership(tid2)

    stored = await _put(owner1, tid1, step=1)
    assert await _tip_id(saver, tid1) == stored["configurable"]["checkpoint_id"], (
        f"claiming {tid2} superseded the owner of {tid1}"
    )


async def test_new_owner_can_write_after_superseding(
    saver: BaseCheckpointSaver,
) -> None:
    """Recovery works: the taking-over owner writes and its writes are durable.

    A fence that refuses everyone is trivially safe and useless. This is the clause that
    makes the capability about *transferring* ownership rather than locking a thread.
    """
    tid = str(uuid4())
    old = await saver.aclaim_write_ownership(tid)
    c1 = (await _put(old, tid))["configurable"]["checkpoint_id"]

    new = await saver.aclaim_write_ownership(tid)
    c2 = (await _put(new, tid, parent_id=c1, step=1))["configurable"]["checkpoint_id"]
    await new.aput_writes(
        generate_config(tid, checkpoint_id=c2), [("ch", "recovered")], str(uuid4())
    )

    assert await _tip_id(saver, tid) == c2, "the new owner's checkpoint is not the tip"
    writes = await _pending_writes(saver, tid, c2)
    assert [w[2] for w in writes] == ["recovered"], (
        f"the new owner's pending write is missing: {writes}"
    )


async def test_duplicate_put_writes_from_current_owner_is_not_a_rejection(
    saver: BaseCheckpointSaver,
) -> None:
    """Idempotency and ownership are different axes, and this clause pins them apart.

    `test_put_writes_idempotent` stays exactly as correct as it was: absorbing a
    duplicate write from the owner that already made it is not the same event as
    refusing a superseded owner, and a saver that conflates them turns every retry into
    a lost run.
    """
    tid = str(uuid4())
    owner = await saver.aclaim_write_ownership(tid)
    first = await _put(owner, tid)
    task_id = str(uuid4())

    await owner.aput_writes(first, [("ch", "v")], task_id)
    try:
        await owner.aput_writes(first, [("ch", "v")], task_id)
    except BaseException as exc:  # noqa: BLE001 -- classifying, then re-raising
        if is_stale_write_owner_rejection(exc):
            raise AssertionError(
                "a duplicate aput_writes from the *current* owner was refused as a "
                "stale-owner write. Ownership is about who may write, not about "
                "whether this write was already made."
            ) from exc
        raise

    barrier = getattr(owner, "acheck_write_ownership", None)
    if barrier is not None:
        await barrier()  # must not raise: the owner never lost anything


async def test_reads_do_not_require_ownership(saver: BaseCheckpointSaver) -> None:
    """A non-owner can still read the thread.

    Load-bearing twice over. Operationally, a worker has to be able to reconcile against
    a thread's state -- and an operator to inspect it -- without taking it away from
    whoever is running it. And within this suite, every clause above reads through the
    unclaimed factory instance, so a saver that fenced its read paths would fail them all
    for the wrong reason.
    """
    tid = str(uuid4())
    owner = await saver.aclaim_write_ownership(tid)
    stored = await _put(owner, tid)
    cp_id = stored["configurable"]["checkpoint_id"]

    tup = await saver.aget_tuple(generate_config(tid))
    assert tup is not None, "a non-owner's aget_tuple returned None for a written thread"
    assert tup.checkpoint["id"] == cp_id, "a non-owner read the wrong checkpoint"
    assert await _checkpoint_ids(saver, tid) == [cp_id], (
        "a non-owner's alist did not see the owner's checkpoint"
    )


ALL_WRITE_OWNERSHIP_TESTS = [
    test_claim_returns_a_writer,
    test_current_owner_can_put,
    test_current_owner_can_put_writes,
    test_superseded_owner_put_is_rejected,
    test_superseded_owner_put_is_not_durable,
    test_superseded_owner_cannot_fork_the_chain,
    test_superseded_owner_put_writes_is_rejected,
    test_superseded_owner_put_writes_is_not_durable,
    test_superseded_owner_cannot_displace_the_current_owners_write,
    test_superseded_owner_cannot_displace_a_special_channel_write,
    test_superseded_owner_stays_superseded,
    test_ownership_is_per_thread,
    test_new_owner_can_write_after_superseding,
    test_duplicate_put_writes_from_current_owner_is_not_a_rejection,
    test_reads_do_not_require_ownership,
]


async def run_write_ownership_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all write_ownership tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_WRITE_OWNERSHIP_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("write_ownership", test_fn.__name__, True, None)
        except Exception as e:
            failed += 1
            msg = f"{test_fn.__name__}: {e}"
            failures.append(msg)
            if on_test_result:
                on_test_result(
                    "write_ownership", test_fn.__name__, False, traceback.format_exc()
                )
    return passed, failed, failures
