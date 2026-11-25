from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import _get_overwrite
from langgraph.errors import InvalidUpdateError

__all__ = ("TranscriptChannel", "TranscriptEntry", "TranscriptRef")

EntryKind = Literal["append", "reset"]


@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    """One log entry. `append` carries a message, `reset` the full list."""

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


def _coerce(update: Any) -> list[Any]:
    from langchain_core.messages import (
        BaseMessageChunk,
        convert_to_messages,
        message_chunk_to_message,
    )

    items = update if isinstance(update, list) else [update]
    out = []
    for m in convert_to_messages(items):
        m = message_chunk_to_message(m) if isinstance(m, BaseMessageChunk) else m
        if m.id is None:
            m.id = str(uuid.uuid4())
        out.append(m)
    return out


def _is_remove(m: Any) -> bool:
    from langchain_core.messages import RemoveMessage

    return isinstance(m, RemoveMessage)


def _remove_all_id() -> str:
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    return REMOVE_ALL_MESSAGES


class TranscriptChannel(BaseChannel[list, Any, TranscriptRef]):
    """Message list persisted as an append-only log.

    The checkpoint stores `(transcript_id, head)`. A checkpointer that
    supports transcripts persists new entries atomically with the checkpoint
    and returns them on read, so storage grows linearly and loading any
    checkpoint costs O(len(transcript)).

    Writes contain only new or changed messages:

    * a new id appends;
    * an existing id (edit) or a `RemoveMessage` forks: a new transcript
      shares the prefix before the edited message and replays what follows;
    * `REMOVE_ALL_MESSAGES` or `Overwrite` writes one `reset` entry.

    Loading a checkpoint that is not its transcript's tip also forks.
    """

    __slots__ = (
        "value",
        "pos",
        "anchor",
        "resets",
        "transcript_id",
        "head",
        "base_head",
        "pending",
        "parent_transcript_id",
        "fork_seq",
    )

    def __init__(self, typ: Any = list) -> None:
        super().__init__(typ)
        self.value: list[Any] = []
        # id -> index in value
        self.pos: dict[str, int] = {}
        # id -> seq of the entry that introduced it (an append or a reset)
        self.anchor: dict[str, int] = {}
        # seqs of reset entries
        self.resets: set[int] = set()
        self.transcript_id = str(uuid.uuid4())
        self.head = 0
        # Highest seq known committed for this channel's lineage.
        self.base_head = 0
        self.pending: list[TranscriptEntry] = []
        # Set only for a fork this channel created and has not seen committed.
        self.parent_transcript_id = ""
        self.fork_seq = 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TranscriptChannel)

    @property
    def ValueType(self) -> Any:
        return list

    @property
    def UpdateType(self) -> Any:
        return Any

    def _new(self) -> Self:
        new = self.__class__(self.typ)
        new.key = self.key
        return new

    def copy(self) -> Self:
        new = self._new()
        new.value = list(self.value)
        new.pos = dict(self.pos)
        new.anchor = dict(self.anchor)
        new.resets = set(self.resets)
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
            new._reset(_coerce(list(checkpoint)))
            return new
        for e in checkpoint.entries:
            if e.kind == "reset":
                new._load_reset(e.seq, e.value)
            else:
                new._load_append(e.seq, e.value)
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
        for v in values:
            is_overwrite, overwrite = _get_overwrite(v)
            if is_overwrite:
                self._reset(_coerce(list(overwrite)))
                continue
            msgs = _coerce(v)
            remove_all = [
                i
                for i, m in enumerate(msgs)
                if _is_remove(m) and m.id == _remove_all_id()
            ]
            if remove_all:
                self._reset(msgs[remove_all[-1] + 1 :])
                continue
            for m in msgs:
                self._write(m)
        return True

    def _write(self, m: Any) -> None:
        i = self.pos.get(m.id)
        if i is None:
            if _is_remove(m):
                raise InvalidUpdateError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )
            self._append(m)
            return
        current = self.value[i]
        if current is m or (not _is_remove(m) and current == m):
            # Resend of a message already in the transcript.
            return
        replacement = [] if _is_remove(m) else [m]
        self._fork_at(m.id, replacement)

    def _fork_at(self, item_id: str, replacement: list[Any]) -> None:
        """Fork before `item_id`, swap it for `replacement`, replay the rest."""
        # The fork point must already be committed, so it is at most base_head.
        fork = min(self.anchor[item_id] - 1, self.base_head)
        prefix = [m for m in self.value if self.anchor[m.id] <= fork]
        # Entries in (fork, head] that brought in the tail, in order. After a
        # reset no older item survives, so a reset group is the whole list.
        groups: list[tuple[int, list[Any]]] = []
        for m in self.value:
            a = self.anchor[m.id]
            if a <= fork:
                continue
            if groups and groups[-1][0] == a:
                groups[-1][1].append(m)
            else:
                groups.append((a, [m]))

        # Items surviving the latest reset after the fork point all anchor at
        # or after it, so replay starts with that reset, even when empty.
        latest_reset = max((r for r in self.resets if r > fork), default=None)
        if latest_reset is not None and (not groups or groups[0][0] != latest_reset):
            groups.insert(0, (latest_reset, []))
        old_resets = self.resets

        if self.parent_transcript_id:
            # Fork created this run: base_head == fork_seq, so branch off its parent.
            parent = self.parent_transcript_id
        elif fork == 0:
            parent = ""
        else:
            parent = self.transcript_id
        resets = {r for r in self.resets if r <= fork}
        anchors = {m.id: self.anchor[m.id] for m in prefix}

        self.transcript_id = str(uuid.uuid4())
        self.parent_transcript_id = parent
        self.fork_seq = self.head = self.base_head = fork
        self.pending = []
        self.value = prefix
        self.pos = {m.id: i for i, m in enumerate(prefix)}
        self.anchor = anchors
        self.resets = resets
        for a, group in groups:
            swapped = [
                n for m in group for n in (replacement if m.id == item_id else [m])
            ]
            if a in old_resets:
                self._reset(swapped)
            else:
                for m in swapped:
                    self._append(m)

    def _append(self, m: Any) -> None:
        self.head += 1
        self.pending.append(TranscriptEntry(self.head, "append", m.id, m))
        self._load_append(self.head, m)

    def _reset(self, msgs: list[Any]) -> None:
        self.head += 1
        self.pending.append(TranscriptEntry(self.head, "reset", "", list(msgs)))
        self._load_reset(self.head, msgs)

    def _load_append(self, seq: int, m: Any) -> None:
        self.pos[m.id] = len(self.value)
        self.anchor[m.id] = seq
        self.value.append(m)

    def _load_reset(self, seq: int, msgs: list[Any]) -> None:
        self.value = list(msgs)
        self.pos = {m.id: i for i, m in enumerate(self.value)}
        self.anchor = dict.fromkeys(self.pos, seq)
        self.resets.add(seq)

    def checkpoint(self) -> TranscriptRef:
        return TranscriptRef(
            transcript_id=self.transcript_id,
            head=self.head,
            base_head=self.base_head,
            entries=list(self.pending),
            parent_transcript_id=self.parent_transcript_id,
            fork_seq=self.fork_seq,
        )
