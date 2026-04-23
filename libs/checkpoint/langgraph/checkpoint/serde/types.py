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


class _SeedUnset:
    """Marker used as the default for `DeltaChannelWrites.seed`.

    Distinct from `None`, which is a legitimate pre-delta value
    (e.g. an `Optional` field whose accumulated value really was `None`).
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "SEED_UNSET"


SEED_UNSET = _SeedUnset()


@dataclasses.dataclass
class DeltaChannelWrites:
    """In-memory wrapper around per-step writes reconstructed by a saver.
    Consumed by `DeltaChannel.from_checkpoint`. Never serialized — if this
    reaches the wire, something upstream forgot to unwrap it.

    `seed` is the value from which chain replay should begin. When the saver
    encounters a pre-delta blob during the ancestor walk, it uses that blob
    as the seed and stops walking further back (the older chain is
    represented by the seed). `SEED_UNSET` means "no seed — replay from the
    channel's empty value".
    """

    writes: list[Any]
    seed: Any = SEED_UNSET


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
