"""The write-ownership contract: who may write a thread, and how a loser finds out.

This module exists because the suite has no vocabulary for a saver *refusing* a write.
`aput` returns a `RunnableConfig`, `aput_writes` returns `None`, `ERROR` and `INTERRUPT`
are payload channel names a saver must store rather than refusals it may issue, and
`langgraph.checkpoint` ships no errors module at all. So the contract needs a name for
the refusal before it can have a test.

**Why the marker attribute rather than a shared base class.** A saver that advertises
this capability must be able to raise a recognisable rejection *without importing a test
package at runtime*. Requiring `isinstance(exc, StaleWriteOwnerError)` would make
`langgraph-checkpoint-conformance` a production dependency of every conforming saver,
which is backwards. So recognition is duck-typed on one attribute
(`is_stale_write_owner_error = True`), `StaleWriteOwnerError` below is the canonical
implementation for anyone who wants it, and a saver is free to set the attribute on an
exception class of its own. If this contract is ever promoted into
`langgraph-checkpoint`, the marker keeps both spellings working through the transition.

**Why the refusal may be deferred.** The obvious contract -- "the write call raises" --
is not implementable on at least one real saver. A server-side fence executed inside
`psycopg`'s pipeline mode does not surface its error from the statement that failed:
libpq defers it to the next `Sync`, so it arrives from the enclosing context manager's
exit, and it was measured arriving from a *later* write block entirely in 4 of 11 cases.
A contract that demands an eager raise would therefore exclude the strongest available
implementation of the very property it is testing.

So the contract is two-part, and the second part is what makes it observable:

1. **Durability.** A write from a superseded owner must not become durable. This is the
   part that matters, and it is checked by reading the thread back.
2. **Observability, by a stated deadline.** The superseded owner must be able to learn
   it lost -- either because the write raised, or because an explicit
   `acheck_write_ownership()` barrier raises. Returning normally with no barrier is a
   *silent* refusal and fails the suite: the runtime then still believes it owns the
   thread, which is the state that produces a forked checkpoint chain.

That second clause is a deliberate, narrow disagreement with the nearest prior art
upstream (short-circuiting stale writes after `delete_thread` and returning normally).
Silence is cheaper to implement and strictly worse to operate: the loser reports success
to its caller and keeps going.

**What this is not.** Not serialization -- ownership answers *who* may write, not in what
order, and an unfenced deployment must behave exactly as it does today. Not idempotency
-- absorbing a duplicate write and refusing a superseded owner are different axes, and
`test_duplicate_put_writes_from_current_owner_is_not_a_rejection` pins them apart. Not
exactly-once side effects, which no checkpointer can offer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "STALE_WRITE_OWNER_MARKER",
    "StaleWriteOwnerError",
    "is_stale_write_owner_rejection",
    "SupportsWriteOwnership",
    "observe_rejection",
]

#: The attribute a rejection carries so the suite can recognise it without an import.
#: Set it to ``True`` on the exception class a saver raises for a superseded owner.
STALE_WRITE_OWNER_MARKER = "is_stale_write_owner_error"


class StaleWriteOwnerError(Exception):
    """A superseded write owner attempted a write.

    Canonical implementation of the rejection. A saver may raise this, or may raise its
    own exception type with :data:`STALE_WRITE_OWNER_MARKER` set truthy on the class --
    the suite treats the two identically, so advertising the capability never requires
    importing this package at runtime.
    """

    is_stale_write_owner_error = True

    def __init__(self, thread_id: str | None = None, detail: str | None = None) -> None:
        parts = ["write refused: this owner has been superseded"]
        if thread_id is not None:
            parts.append(f"thread_id={thread_id!r}")
        if detail:
            parts.append(detail)
        super().__init__(" -- ".join(parts))
        self.thread_id = thread_id


def is_stale_write_owner_rejection(exc: BaseException) -> bool:
    """Whether *exc* is a saver refusing a write from a superseded owner.

    Duck-typed on :data:`STALE_WRITE_OWNER_MARKER` for the reason in the module
    docstring. Deliberately not an ``isinstance`` check, and deliberately not a match on
    an error code: a fenced Postgres saver surfaces the same logical refusal as at least
    two unrelated psycopg exception types depending on whether the connection was opened
    with a connection-level pipeline, and a suite that classified by type or SQLSTATE
    alone would read a *working* fence as a passing one on one of those branches.
    """
    return getattr(exc, STALE_WRITE_OWNER_MARKER, False) is True


@runtime_checkable
class SupportsWriteOwnership(Protocol):
    """What a saver advertising ``write_ownership`` must expose.

    Detection is override-based and keyed on ``aclaim_write_ownership``, so the name has
    to be distinct: an ownership clause cannot hide inside ``aput``/``aput_writes`` and
    still be auto-detected. The method is absent from ``BaseCheckpointSaver``, which is
    exactly why adding this capability needs no base-class change --
    ``capabilities._is_overridden`` returns ``True`` for a method the base class does not
    define at all.
    """

    async def aclaim_write_ownership(self, thread_id: str) -> Any:
        """Take write ownership of *thread_id*, superseding any current owner.

        Returns an object exposing ``aput`` and ``aput_writes`` whose writes to
        *thread_id* are accepted only while it remains the current owner. It may be
        ``self``, a view over the same connection, or a fresh saver -- the suite does not
        care, and only ever writes through the returned object.

        Claiming must be monotonic: once superseded, an owner cannot become current
        again. A previous holder may of course claim afresh and receive a *new*
        ownership; the old handle stays dead.
        """
        ...


async def observe_rejection(
    writer: Any,
    write: Callable[[], Awaitable[Any]],
    *,
    what: str,
) -> BaseException:
    """Perform *write* from a superseded owner and return the rejection it produced.

    Accepts either deadline: the write itself raising, or the optional
    ``acheck_write_ownership()`` barrier raising afterwards. Raises ``AssertionError``
    -- with the diagnosis, not just a boolean -- for the three ways this can go wrong:
    an unrecognisable exception, a silent refusal with no barrier to ask, and a barrier
    that reports everything is fine.
    """
    try:
        await write()
    except BaseException as exc:  # noqa: BLE001 -- classifying, then re-raising
        if is_stale_write_owner_rejection(exc):
            return exc
        raise AssertionError(
            f"{what} from a superseded owner raised {type(exc).__name__}: {exc} -- "
            f"which the suite cannot identify as a stale-write-owner rejection, because "
            f"the exception carries no truthy {STALE_WRITE_OWNER_MARKER!r} attribute. "
            f"Set that attribute on the exception class (see "
            f"langgraph.checkpoint.conformance.ownership.StaleWriteOwnerError)."
        ) from exc

    barrier = getattr(writer, "acheck_write_ownership", None)
    if barrier is None:
        raise AssertionError(
            f"{what} from a superseded owner returned normally, and the writer exposes "
            f"no acheck_write_ownership() barrier -- so a superseded worker has no way "
            f"to learn it lost the thread and will report success to its caller. A "
            f"refusal that is invisible to the loser does not satisfy this capability; "
            f"either raise from the write, or expose the barrier."
        )

    try:
        await barrier()
    except BaseException as exc:  # noqa: BLE001 -- classifying, then re-raising
        if is_stale_write_owner_rejection(exc):
            return exc
        raise AssertionError(
            f"acheck_write_ownership() after {what} from a superseded owner raised "
            f"{type(exc).__name__}: {exc} -- not identifiable as a stale-write-owner "
            f"rejection (no truthy {STALE_WRITE_OWNER_MARKER!r} attribute)."
        ) from exc

    raise AssertionError(
        f"{what} from a superseded owner was refused silently: the write returned "
        f"normally and acheck_write_ownership() then reported ownership intact. The "
        f"loser believes it still owns the thread."
    )
