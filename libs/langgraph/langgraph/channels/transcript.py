from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import _get_overwrite
from langgraph.errors import InvalidUpdateError

__all__ = ("TranscriptChannel", "TranscriptEntry", "TranscriptRef")

EntryKind = Literal["append", "replace", "remove", "reset"]


@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    """One reducer output. `reset` carries the full list in `value`."""

    seq: int
    kind: EntryKind
    item_id: str = ""
    value: Any = None


@dataclass(slots=True)
class TranscriptRef:
    """Checkpoint value of a `TranscriptChannel`: a pointer into a log.

    Written: `entries` holds entries with seq in (base_head, head].
    Read: `entries` holds every entry in [1, head], fork ancestors included.
    """

    transcript_id: str
    head: int
    base_head: int = 0
    entries: list[TranscriptEntry] = field(default_factory=list)
    parent_transcript_id: str = ""
    fork_seq: int = 0
    is_tip: bool = True


def fold(entries: Sequence[TranscriptEntry]) -> list[Any]:
    """Latest version per id, in first-appearance order, removals dropped."""
    items: dict[str, Any] = {}
    for e in entries:
        if e.kind == "reset":
            items = {_item_id(m): m for m in e.value}
        elif e.kind == "remove":
            items.pop(e.item_id, None)
        else:
            items[e.item_id] = e.value
    return list(items.values())


def diff(old: list[Any], new: list[Any]) -> list[tuple[EntryKind, str, Any]]:
    """Entries that turn `old` into `new` under `fold`.

    Falls back to a single `reset` when `new` reorders surviving items.
    """
    old_by_id = {_item_id(m): m for m in old}
    new_ids = [_item_id(m) for m in new]
    kept = [i for i in new_ids if i in old_by_id]
    added = [i for i in new_ids if i not in old_by_id]
    surviving = set(new_ids)
    if kept + added != new_ids or kept != [i for i in old_by_id if i in surviving]:
        return [("reset", "", list(new))]
    out: list[tuple[EntryKind, str, Any]] = [
        ("remove", i, None) for i in old_by_id if i not in surviving
    ]
    for m in new:
        i = _item_id(m)
        prev = old_by_id.get(i, MISSING)
        if prev is MISSING:
            out.append(("append", i, m))
        elif prev is not m and prev != m:
            out.append(("replace", i, m))
    return out


def _item_id(item: Any) -> str:
    item_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
    if not item_id:
        raise InvalidUpdateError("TranscriptChannel items must have an id")
    return str(item_id)


def _default_reducer(left: Any, right: Any) -> Any:
    from langgraph.graph.message import add_messages

    return add_messages(left, right)


class TranscriptChannel(BaseChannel[list, Any, TranscriptRef]):
    """List channel persisted as an append-only log of reducer outputs.

    The checkpoint stores `(transcript_id, head)`. A checkpointer that
    supports transcripts persists `entries` atomically with the checkpoint and
    returns them on read, so loading any checkpoint costs O(len(transcript))
    and storage grows linearly. Loading a checkpoint that is not the
    transcript's tip forks into a new transcript.

    Items must carry stable ids (messages get them from `add_messages`).
    """

    __slots__ = (
        "value",
        "reducer",
        "transcript_id",
        "head",
        "base_head",
        "pending",
        "parent_transcript_id",
        "fork_seq",
    )

    def __init__(
        self,
        reducer: Callable[[Any, Any], Any] = _default_reducer,
        typ: Any = list,
    ) -> None:
        super().__init__(typ)
        self.reducer = reducer
        self.value: list[Any] = []
        self.transcript_id = str(uuid.uuid4())
        self.head = 0
        self.base_head = 0
        self.pending: list[TranscriptEntry] = []
        self.parent_transcript_id = ""
        self.fork_seq = 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TranscriptChannel) and self.reducer == other.reducer

    @property
    def ValueType(self) -> Any:
        return list

    @property
    def UpdateType(self) -> Any:
        return Any

    def _new(self) -> Self:
        new = self.__class__(self.reducer, self.typ)
        new.key = self.key
        return new

    def copy(self) -> Self:
        new = self._new()
        new.value = list(self.value)
        new.transcript_id = self.transcript_id
        new.head = self.head
        new.base_head = self.base_head
        new.pending = list(self.pending)
        new.parent_transcript_id = self.parent_transcript_id
        new.fork_seq = self.fork_seq
        return new

    def from_checkpoint(self, checkpoint: Any) -> Self:
        new = self._new()
        if checkpoint is MISSING:
            return new
        if not isinstance(checkpoint, TranscriptRef):
            # Plain list from a regular channel: start a transcript from it.
            new.value = list(checkpoint)
            new._append(diff([], new.value))
            return new
        new.value = fold(checkpoint.entries)
        new.head = new.base_head = checkpoint.head
        if checkpoint.is_tip:
            new.transcript_id = checkpoint.transcript_id
        else:
            new.parent_transcript_id = checkpoint.transcript_id
            new.fork_seq = checkpoint.head
        return new

    def get(self) -> list[Any]:
        return self.value

    def is_available(self) -> bool:
        return True

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        new = self.value
        for v in values:
            is_overwrite, overwrite = _get_overwrite(v)
            new = list(overwrite) if is_overwrite else self.reducer(new, v)
        self._append(diff(self.value, new))
        self.value = new
        return True

    def _append(self, changes: list[tuple[EntryKind, str, Any]]) -> None:
        for kind, item_id, value in changes:
            self.head += 1
            self.pending.append(TranscriptEntry(self.head, kind, item_id, value))

    def checkpoint(self) -> TranscriptRef:
        return TranscriptRef(
            transcript_id=self.transcript_id,
            head=self.head,
            base_head=self.base_head,
            entries=list(self.pending),
            parent_transcript_id=self.parent_transcript_id,
            fork_seq=self.fork_seq,
        )
