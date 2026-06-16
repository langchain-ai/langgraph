from __future__ import annotations

import collections.abc
import copy as _copy
from collections.abc import Callable, Sequence
from typing import Any, Generic

from langgraph.checkpoint.base import PendingWrite
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.channels.binop import _get_overwrite, _operators_equal, _strip_extras
from langgraph.errors import (
    EmptyChannelError,
    ErrorCode,
    InvalidUpdateError,
    create_error_message,
)

__all__ = ("DeltaChannel",)


class DeltaChannel(Generic[Value], BaseChannel[Any, Any, Any]):
    """Reducer channel that stores only a sentinel in checkpoint blobs and
    reconstructs state by replaying ancestor writes through the reducer.

    !!! warning "Beta"

        `DeltaChannel` is in beta. The API and on-disk representation may
        change in future releases. Threads written with `DeltaChannel` today
        are expected to remain readable, but the surrounding contract
        (`BaseCheckpointSaver.get_delta_channel_history`, the
        `_DeltaSnapshot` blob shape, the `counters_since_delta_snapshot`
        metadata field) is not yet stable.

    The reducer receives the current accumulated value and a batch of writes
    in one call: `reducer(state, [write1, write2, ...]) -> new_state`.

    Reducers must be deterministic and batching-invariant (associative across
    folds): applying two consecutive write batches separately must produce the
    same state as applying their concatenation once:

        reducer(reducer(state, xs), ys) == reducer(state, xs + ys)

    This lets LangGraph replay checkpointed writes in larger batches than they
    were originally produced without changing reconstructed state.

    Snapshot cadence is driven by two counters: per-channel update count and
    total supersteps since last snapshot. `create_checkpoint` writes a full
    `_DeltaSnapshot` blob when EITHER the update count reaches
    `snapshot_frequency` OR the supersteps count reaches the system-wide
    `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` bound (default 5000), bounding
    replay depth even for channels that stop receiving writes.

    Parameters:
        reducer: `(state, list[writes]) -> new_state`. Must be deterministic
            and batching-invariant as described above.
        typ: The value type (e.g. `list`, `dict`). Inferred automatically
            from the outer type when used inside `Annotated[T, DeltaChannel(...)]`.
        snapshot_frequency: Every Nth update to this channel writes a snapshot
            blob (default `1000`). Must be a positive int.
    """

    __slots__ = ("value", "reducer", "snapshot_frequency")
    value: Value | Any

    def __init__(
        self,
        reducer: Callable[[Any, Sequence[Any]], Any],
        typ: type[Value] | None = None,
        *,
        snapshot_frequency: int = 1000,
    ) -> None:
        if snapshot_frequency <= 0:
            raise ValueError(
                f"snapshot_frequency must be a positive int, got {snapshot_frequency}"
            )
        if typ is None:
            typ = list  # type: ignore[assignment]  # placeholder; overridden by _is_field_channel
        super().__init__(typ)
        self.reducer = reducer
        self.snapshot_frequency = snapshot_frequency
        typ = _strip_extras(typ)
        if typ in (collections.abc.Sequence, collections.abc.MutableSequence):
            typ = list
        if typ in (collections.abc.Set, collections.abc.MutableSet):
            typ = set
        if typ in (collections.abc.Mapping, collections.abc.MutableMapping):
            typ = dict
        self.typ = typ
        self.value: Any = MISSING

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeltaChannel):
            return False
        if self.snapshot_frequency != other.snapshot_frequency:
            return False
        return _operators_equal(self.reducer, other.reducer)

    @property
    def ValueType(self) -> Any:
        return self.typ

    @property
    def UpdateType(self) -> Any:
        return self.typ

    def copy(self) -> Self:
        new = self.__class__(
            self.reducer, self.typ, snapshot_frequency=self.snapshot_frequency
        )
        new.key = self.key
        new.value = self.value if self.value is MISSING else _copy.copy(self.value)
        return new

    def from_checkpoint(self, checkpoint: Any) -> Self:
        """Initialize from a stored blob.

        Blob types:
          * `MISSING`: start empty; caller replays writes.
          * `_DeltaSnapshot(value)`: restore value directly from snapshot.
          * plain value (migration from old `BinaryOperatorAggregate` blobs):
            use directly.
        """
        new = self.__class__(
            self.reducer, self.typ, snapshot_frequency=self.snapshot_frequency
        )
        new.key = self.key
        if checkpoint is MISSING:
            new.value = self.typ()
        elif isinstance(checkpoint, _DeltaSnapshot):
            new.value = checkpoint.value
        else:
            new.value = checkpoint
        return new

    def replay_writes(self, writes: Sequence[PendingWrite]) -> None:
        """Apply ancestor writes oldest-to-newest via a single reducer call.

        If any write is an Overwrite, the last one in the sequence acts as
        the reset point: its value becomes the new base and only writes
        after it are passed to the reducer.
        """
        values = [v for _, _, v in writes]
        if not values:
            return
        base = self.value
        start = 0
        for i, v in enumerate(values):
            is_ow, ow_value = _get_overwrite(v)
            if is_ow:
                base = _copy.copy(ow_value) if ow_value is not None else self.typ()
                start = i + 1
        remaining = values[start:]
        self.value = self.reducer(base, remaining) if remaining else base

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        overwrite_idx: int | None = None
        for i, v in enumerate(values):
            is_ow, _ = _get_overwrite(v)
            if is_ow:
                if overwrite_idx is not None:
                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
                overwrite_idx = i
        if overwrite_idx is not None:
            _, overwrite_value = _get_overwrite(values[overwrite_idx])
            base = (
                _copy.copy(overwrite_value)
                if overwrite_value is not None
                else self.typ()
            )
            remaining = [v for i, v in enumerate(values) if i != overwrite_idx]
            self.value = self.reducer(base, remaining) if remaining else base
            return True
        base = self.typ() if self.value is MISSING else self.value
        self.value = self.reducer(base, list(values))
        return True

    def get(self) -> Any:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        """Return stored representation: always `MISSING`.

        Snapshot decisions live in `create_checkpoint` (which has the channel
        version) and write `_DeltaSnapshot(ch.get())` directly into
        `channel_values`. For non-snapshot steps the channel does not appear
        in `channel_values`; reconstruction walks ancestor writes via the
        saver's `get_delta_channel_history`.
        """
        return MISSING
