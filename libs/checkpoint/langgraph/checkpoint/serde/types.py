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
    """One immutable entry of a transcript. The writer assigns `seq` and `hash`."""

    seq: int
    hash: bytes
    payload: bytes


class _TranscriptRef(NamedTuple):
    """Pointer to a transcript, stored as the checkpoint's value for a channel."""

    transcript_id: str
    head: int


class _TranscriptWrite(NamedTuple):
    """What a channel hands `put` for a transcript.

    A saver that supports transcripts stores `entries` with the checkpoint and
    keeps `_TranscriptRef(transcript_id, head)` as the channel's value.
    `parent_transcript_id` and `fork_seq` are set only by a transcript forked
    from another, and are the same on every write to it.
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
