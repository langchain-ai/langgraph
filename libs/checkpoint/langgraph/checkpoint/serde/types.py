import dataclasses
from collections.abc import Sequence
from typing import (
    Any,
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


class _DeltaSentinel:
    """Singleton marker stored (as zero bytes) in checkpoint_blobs for a
    DeltaChannel field. The actual per-step writes live in checkpoint_writes
    and are replayed through the reducer at load time.

    Compare with `is DELTA_SENTINEL` — `loads_typed` always returns the same
    module-level instance.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "DELTA_SENTINEL"


DELTA_SENTINEL = _DeltaSentinel()


@dataclasses.dataclass
class DeltaChannelWrites:
    """In-memory wrapper around per-step writes reconstructed by a saver.
    Consumed by `DeltaChannel.from_checkpoint`. Never serialized — if this
    reaches the wire, something upstream forgot to unwrap it.
    """

    writes: list[Any]


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
