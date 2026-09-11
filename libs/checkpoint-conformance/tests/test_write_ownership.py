"""Self-tests for the WRITE_OWNERSHIP capability. No database, no network.

Five savers, because a conformance clause that nothing fails is decoration -- and because
the two most contestable choices in this capability should be settled by execution rather
than by argument:

* `OwnedInMemorySaver` -- a reference implementation, about forty lines over
  `InMemorySaver`. It exists to show that the contract is small, and to give CI a green
  fixture.
* `HostileInMemorySaver` -- advertises the capability and does nothing to honour it,
  which is what today's savers effectively do. It must fail, and this file asserts
  *which* clauses it fails, so a later change that quietly weakens one of them turns this
  file red rather than passing.
* `SilentInMemorySaver` -- **the contested one.** It genuinely refuses: a superseded
  owner's writes reach no storage. It just never says so, which is the house style of the
  nearest prior art upstream. It must fail, and it must fail for the observability reason
  and not for a durability reason. If a maintainer disagrees with one thing in this
  capability it will be this, so the disagreement is a named test rather than a paragraph.
* `DeferredInMemorySaver` -- refuses durably and reports only at the barrier, never from
  the write. It must **pass**. This is the shape a server-side fence in psycopg pipeline
  mode is forced into, and if it did not pass, the contract would exclude the strongest
  available implementation of the property it tests.
* plain `InMemorySaver` -- must be untouched: the capability undetected, and
  `passed_all_base()` still true. That is the "extended, never base" property, executed
  rather than asserted in prose.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.capabilities import (
    BASE_CAPABILITIES,
    EXTENDED_CAPABILITIES,
    Capability,
    DetectedCapabilities,
)
from langgraph.checkpoint.conformance.ownership import (
    StaleWriteOwnerError,
    is_stale_write_owner_rejection,
)
from langgraph.checkpoint.conformance.spec.test_write_ownership import (
    ALL_WRITE_OWNERSHIP_TESTS,
)

CAP = Capability.WRITE_OWNERSHIP.value


# ------------------------------------------------- the reference implementation ---


class _Owner:
    """A write handle valid only while it holds the thread's current token."""

    def __init__(self, saver: "OwnedInMemorySaver", thread_id: str, token: int) -> None:
        self._saver = saver
        self._thread_id = thread_id
        self._token = token

    def _assert_current(self) -> None:
        current = self._saver._tokens.get(self._thread_id)
        if current != self._token:
            raise StaleWriteOwnerError(
                self._thread_id,
                f"held token {self._token}, current token is {current}",
            )

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any):
        self._assert_current()
        return await self._saver.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        self._assert_current()
        return await self._saver.aput_writes(config, writes, task_id, task_path)

    async def acheck_write_ownership(self) -> None:
        """The second deadline. Redundant here -- this implementation refuses eagerly --
        and implemented anyway, because a saver whose refusal arrives from a later flush
        needs it and the suite must exercise the path."""
        self._assert_current()


class OwnedInMemorySaver(InMemorySaver):
    """`InMemorySaver` plus per-thread write ownership."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tokens: dict[str, int] = {}

    async def aclaim_write_ownership(self, thread_id: str) -> _Owner:
        token = self._tokens.get(thread_id, 0) + 1
        self._tokens[thread_id] = token
        return _Owner(self, thread_id, token)


class HostileInMemorySaver(InMemorySaver):
    """Advertises the capability, enforces nothing. The negative control."""

    async def aclaim_write_ownership(self, thread_id: str) -> "HostileInMemorySaver":
        return self


# ------------------------------------------------------ the two shape controls ---


class _QuietOwner(_Owner):
    """Refuses durably, reports nothing. Neither deadline is met."""

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any):
        try:
            self._assert_current()
        except StaleWriteOwnerError:
            return config  # dropped, and the caller is none the wiser
        return await self._saver.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        try:
            self._assert_current()
        except StaleWriteOwnerError:
            return None
        return await self._saver.aput_writes(config, writes, task_id, task_path)

    async def acheck_write_ownership(self) -> None:
        return None  # "everything is fine"


class _DeferredOwner(_QuietOwner):
    """Refuses durably and reports at the barrier only -- the second deadline."""

    async def acheck_write_ownership(self) -> None:
        self._assert_current()


class SilentInMemorySaver(OwnedInMemorySaver):
    """Real enforcement, no signal. Must fail -- on observability, not durability."""

    async def aclaim_write_ownership(self, thread_id: str) -> _QuietOwner:
        token = self._tokens.get(thread_id, 0) + 1
        self._tokens[thread_id] = token
        return _QuietOwner(self, thread_id, token)


class DeferredInMemorySaver(OwnedInMemorySaver):
    """Real enforcement, reported at the barrier. Must pass."""

    async def aclaim_write_ownership(self, thread_id: str) -> _DeferredOwner:
        token = self._tokens.get(thread_id, 0) + 1
        self._tokens[thread_id] = token
        return _DeferredOwner(self, thread_id, token)


@checkpointer_test(name="OwnedInMemorySaver")
async def owned_checkpointer():
    yield OwnedInMemorySaver()


@checkpointer_test(name="HostileInMemorySaver")
async def hostile_checkpointer():
    yield HostileInMemorySaver()


@checkpointer_test(name="SilentInMemorySaver")
async def silent_checkpointer():
    yield SilentInMemorySaver()


@checkpointer_test(name="DeferredInMemorySaver")
async def deferred_checkpointer():
    yield DeferredInMemorySaver()


@checkpointer_test(name="PlainInMemorySaver")
async def plain_checkpointer():
    yield InMemorySaver()


# ------------------------------------------------------------------- the marker ---


def test_marker_recognises_the_canonical_error():
    assert is_stale_write_owner_rejection(StaleWriteOwnerError("t"))


def test_marker_recognises_a_savers_own_error_type():
    """The point of duck-typing: no runtime dependency on this package."""

    class MySaverLostTheThread(RuntimeError):
        is_stale_write_owner_error = True

    assert is_stale_write_owner_rejection(MySaverLostTheThread("gone"))


def test_marker_does_not_recognise_unrelated_errors():
    for exc in (RuntimeError("boom"), ValueError("nope"), KeyError("k")):
        assert not is_stale_write_owner_rejection(exc)


# --------------------------------------------------------------------- detection ---


def test_capability_is_extended_not_base():
    assert Capability.WRITE_OWNERSHIP in EXTENDED_CAPABILITIES
    assert Capability.WRITE_OWNERSHIP not in BASE_CAPABILITIES


def test_detection_is_purely_additive():
    """Detected by defining the method; absent on savers that do not.

    `aclaim_write_ownership` does not exist on `BaseCheckpointSaver`, so this needs no
    base-class change and cannot alter how any existing saver is classified.
    """
    from langgraph.checkpoint.base import BaseCheckpointSaver

    assert not hasattr(BaseCheckpointSaver, "aclaim_write_ownership")
    assert (
        Capability.WRITE_OWNERSHIP
        in DetectedCapabilities.from_instance(OwnedInMemorySaver()).detected
    )
    assert (
        Capability.WRITE_OWNERSHIP
        not in DetectedCapabilities.from_instance(InMemorySaver()).detected
    )


# ------------------------------------------------------------------ the verdicts ---


@pytest.mark.asyncio
async def test_reference_implementation_passes():
    report = await validate(owned_checkpointer, capabilities={CAP})
    result = report.results[CAP]
    assert result.detected
    assert result.passed is True, f"failures: {result.failures}"
    assert result.tests_passed == len(ALL_WRITE_OWNERSHIP_TESTS)


#: What a saver that ignores supersession must fail. Named rather than counted so that a
#: clause weakened into vacuity is visible here.
EXPECTED_HOSTILE_FAILURES = {
    "test_superseded_owner_put_is_rejected",
    "test_superseded_owner_put_is_not_durable",
    "test_superseded_owner_cannot_fork_the_chain",
    "test_superseded_owner_put_writes_is_rejected",
    "test_superseded_owner_put_writes_is_not_durable",
    "test_superseded_owner_cannot_displace_the_current_owners_write",
    "test_superseded_owner_cannot_displace_a_special_channel_write",
    "test_superseded_owner_stays_superseded",
}


@pytest.mark.asyncio
async def test_hostile_saver_fails_and_fails_exactly_the_ownership_clauses():
    """The clause has teeth -- and only where it should.

    Seven of fourteen. The other seven passing is the interesting half: this capability
    is about *transferring* ownership, so a saver that refused every write, or one whose
    ownership was scoped to the whole process, would fail clauses a hostile saver passes.
    """
    report = await validate(hostile_checkpointer, capabilities={CAP})
    result = report.results[CAP]
    assert result.detected
    assert result.passed is False, "a saver that ignores supersession must not conform"

    failed = {f.split(":", 1)[0] for f in result.failures}
    assert failed == EXPECTED_HOSTILE_FAILURES, (
        f"unexpected failure set.\n"
        f"  missing: {EXPECTED_HOSTILE_FAILURES - failed}\n"
        f"  extra:   {failed - EXPECTED_HOSTILE_FAILURES}"
    )


@pytest.mark.asyncio
async def test_a_silent_refusal_does_not_conform():
    """The deliberate disagreement, as a test.

    This saver's enforcement is real -- the superseded write reaches no storage. It fails
    only because the loser is never told, so it goes on believing it owns the thread and
    reports success to its caller. That is the state that produces a forked chain in the
    first place, so "refused but silent" is not a partial pass here; it is a fail.

    Asserted to fail exactly the clauses a hostile saver fails, which is the point: from
    the loser's side, silence and no enforcement at all are the same event.
    """
    report = await validate(silent_checkpointer, capabilities={CAP})
    result = report.results[CAP]
    assert result.passed is False, "a silent refusal must not conform"

    failed = {f.split(":", 1)[0] for f in result.failures}
    assert failed == EXPECTED_HOSTILE_FAILURES, (
        f"a silent refusal should fail exactly the clauses a hostile saver does.\n"
        f"  missing: {EXPECTED_HOSTILE_FAILURES - failed}\n"
        f"  extra:   {failed - EXPECTED_HOSTILE_FAILURES}"
    )
    assert all("returned normally" in f for f in result.failures), (
        f"expected every failure to be about observability, got: {result.failures}"
    )


@pytest.mark.asyncio
async def test_a_deferred_refusal_conforms():
    """The second deadline is real, not a courtesy.

    Nothing here raises from `aput` or `aput_writes`; the refusal is only ever visible
    from `acheck_write_ownership()`. A server-side fence executed inside psycopg's
    pipeline mode has no other option -- libpq defers the error past the statement that
    caused it -- so a contract that demanded an eager raise would exclude it. This clause
    is what stops that requirement being reintroduced by accident.
    """
    report = await validate(deferred_checkpointer, capabilities={CAP})
    result = report.results[CAP]
    assert result.passed is True, f"failures: {result.failures}"
    assert result.tests_passed == len(ALL_WRITE_OWNERSHIP_TESTS)


@pytest.mark.asyncio
async def test_plain_saver_is_unaffected():
    """No existing saver is reclassified, and base conformance is untouched."""
    report = await validate(plain_checkpointer)
    assert report.results[CAP].detected is False
    assert report.results[CAP].passed is None
    assert report.passed_all_base(), f"base broke: {report.to_dict()}"


@pytest.mark.asyncio
async def test_reference_implementation_still_passes_base():
    """Adding ownership does not cost base conformance."""
    report = await validate(owned_checkpointer)
    assert report.passed_all_base(), f"base broke: {report.to_dict()}"
    assert report.results[CAP].passed is True
