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

    The writer assigns `seq`. The saver serializes `value` with its own serde.
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
    `_TranscriptRef(transcript_id, head)` as the channel's value. A write carries
    exactly the entries its checkpoint adds:

    - The base is the head of the parent checkpoint's pointer to this transcript.
      For a transcript the put creates, it is `fork_seq`, or 0 without a parent.
    - `entries` have seqs `base + 1` through `head`, in order, with no gaps and
      nothing at or below the base. A write with no entries has `head == base`.
    - An entry at a seq the transcript already holds raises
      `TranscriptConflictError`, whatever its content.
    - A transcript the parent checkpoint does not point at can only be created,
      not written to.
    - `parent_transcript_id` and `fork_seq` are set together, on the write that
      creates a fork. Later writes may omit them; if given, they must match. The
      parent must already be stored, before this put, in the same thread and
      namespace with entries through `fork_seq`.
    - Only a channel in `new_versions` may carry entries, and one put writes each
      transcript at most once.
    - A put whose checkpoint id is already stored is a replay: its entries are
      not checked or stored.

    Breaking any other rule raises `ValueError`. A rejected put stores nothing.
    A bare `_TranscriptRef` in a checkpoint is checked as a write with no entries.
    """

    transcript_id: str
    head: int
    entries: Sequence[_TranscriptEntry] = ()
    parent_transcript_id: str | None = None
    fork_seq: int | None = None


class _Transcript(NamedTuple):
    """Entries `1..head` of a transcript, with ancestors' shared prefixes resolved.

    Reading a head past the stored entries raises `ValueError`.

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
