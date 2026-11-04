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


class LogAppend(NamedTuple):
    """One task write to a log channel, as handed to `put_writes`.

    !!! warning "Prototype"

        Part of the transcript log prototype. Shape and semantics may change.

    `start` is the log's length at the checkpoint the step started from (the
    position the step's entries begin at). `entries` are the appended entries
    as JSON text, exactly as written. `exec` identifies the execution that
    wrote them, so a log-aware saver can keep two executions from one
    checkpoint apart.
    """

    start: int
    entries: tuple[str, ...]
    exec: str = ""


class LogStep(NamedTuple):
    """A log channel's value in a checkpoint: the step that produced its version.

    !!! warning "Prototype"

        Part of the transcript log prototype. Shape and semantics may change.

    The step appended `count` entries starting at position `start`, so the
    log as of the checkpoint has `start + count` entries. `entries` carries
    the step's entries inline when they did not reach the saver through
    `put_writes` (graph input, `update_state`, `durability="exit"`); it is
    `None` when they did.
    """

    start: int
    count: int
    exec: str = ""
    entries: tuple[str, ...] | None = None

    @property
    def end(self) -> int:
        return self.start + self.count

    def stripped(self) -> "LogStep":
        return LogStep(self.start, self.count, self.exec, None)


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
