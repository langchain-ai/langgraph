from collections.abc import Sequence
from typing import (
    Any,
    NamedTuple,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import Self

ERROR = "__error__"
SCHEDULED = "__scheduled__"
INTERRUPT = "__interrupt__"
RESUME = "__resume__"
TASKS = "__pregel_tasks"


class _DeltaSnapshot(NamedTuple):
    """Snapshot blob for a DeltaChannel with finite snapshot_frequency.

    Stored in checkpoint_blobs via the `EXT_DELTA_SNAPSHOT` msgpack ext code.
    The ancestor walk in `BaseCheckpointSaver.get_delta_channel_history` terminates
    when it encounters this type (any non-empty channel_values entry stops
    the walk for that channel).

    `from_checkpoint` reconstructs the channel value directly from `.value`
    without replaying writes — the snapshot IS the accumulated state.
    """

    value: Any


class _TranscriptEntry(NamedTuple):
    """One immutable entry of a transcript.

    The writer assigns `seq`. The saver serializes `value` with its own serde
    and decides whether a resend at the same seq is unchanged.
    """

    seq: int
    value: Any


class _TranscriptRef(NamedTuple):
    """Pointer to a transcript, stored as the checkpoint's value for a channel."""

    transcript_id: str
    head: int


class _TranscriptWrite(NamedTuple):
    """What a channel hands `put` for a transcript.

    The saver stores `entries` in the checkpoint's transaction and keeps
    `_TranscriptRef(transcript_id, head)` as the channel's value. Rules:

    - Every entry's seq is above `fork_seq` (or 0) and at most `head`.
    - An entry at or below the head of the parent checkpoint's pointer to this
      transcript was committed with it and is skipped unread, so a channel can
      resend everything since it loaded at no cost.
    - Any other entry at a stored seq must have the same content, compared as
      the saver's serialized plaintext (before encryption). Different content
      raises `TranscriptConflictError`.
    - Any other entry is appended at one past the transcript's last seq.
    - `head` is at most the transcript's last seq once the entries are stored.
    - `parent_transcript_id` and `fork_seq` are set together, only on a fork, and
      are the same on every write to it. The parent must exist in the same
      thread and namespace with entries through `fork_seq`.

    Breaking any other rule raises `ValueError`. A rejected put stores nothing.
    """

    transcript_id: str
    head: int
    entries: Sequence[_TranscriptEntry] = ()
    parent_transcript_id: str | None = None
    fork_seq: int | None = None


class _Transcript(NamedTuple):
    """Entries `1..head` of a transcript, with ancestors' shared prefixes resolved.

    `has_later_entries` is true when the transcript holds an entry of its own
    above the requested head. A writer that loaded such a head forks instead of
    appending.
    """

    entries: list[_TranscriptEntry]
    has_later_entries: bool


Value = TypeVar("Value", covariant=True)
Update = TypeVar("Update", contravariant=True)
C = TypeVar("C")


class ChannelProtocol(Protocol[Value, Update, C]):
    # Mirrors langgraph.channels.base.BaseChannel
    @property
    def ValueType(self) -> Any: ...

    @property
    def UpdateType(self) -> Any: ...

    def checkpoint(self) -> C | None: ...

    def from_checkpoint(self, checkpoint: C | None) -> Self: ...

    def update(self, values: Sequence[Update]) -> bool: ...

    def get(self) -> Value: ...

    def consume(self) -> bool: ...


@runtime_checkable
class SendProtocol(Protocol):
    # Mirrors langgraph.constants.Send
    node: str
    arg: Any

    def __hash__(self) -> int: ...

    def __repr__(self) -> str: ...

    def __eq__(self, value: object) -> bool: ...
