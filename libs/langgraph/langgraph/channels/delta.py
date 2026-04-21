from __future__ import annotations

import collections.abc
from collections.abc import Callable, Sequence
from copy import copy
from typing import Any, Generic, Literal

from langgraph.checkpoint.base import DeltaChainValue, DeltaValue
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.channels.binop import _get_overwrite, _strip_extras
from langgraph.errors import EmptyChannelError

__all__ = ("DeltaChannel",)


def _copy_value(value: Any) -> Any:
    if value is MISSING:
        return value
    try:
        return value.copy()
    except AttributeError:
        return copy(value)


class DeltaChannel(Generic[Value], BaseChannel[list[Value], Value, DeltaValue]):
    """A channel that stores only per-step write deltas in checkpoints.

    Reconstructs the full accumulated list at load time by replaying the
    chain of deltas through the operator. Use with append-style reducers
    (e.g. `add_messages`) on long-running threads to reduce checkpoint
    storage from O(N²) to O(N).

    Works with all checkpointers. Savers with a dedicated blob store
    (InMemorySaver, PostgresSaver) use an O(1) fast-path per chain step;
    all others (SQLite, MongoDB, etc.) fall back to get_tuple traversal.

    Use `snapshot_every=N` to cap chain traversal depth at N steps. Every N
    steps a full snapshot is written as the chain root; subsequent deltas
    chain back to it, so `get_state` / reload never traverses more than N
    checkpoints regardless of thread length. Recommended for savers without
    a dedicated blob store.

    Usage::

        class State(TypedDict):
            messages: Annotated[list[AnyMessage], DeltaChannel(add_messages)]
            # Cap reconstruction depth (recommended for SQLite / MongoDB savers):
            messages: Annotated[list[AnyMessage], DeltaChannel(add_messages, snapshot_every=50)]
    """

    __slots__ = (
        "value",
        "operator",
        "snapshot_every",
        "_pending",
        "_base_version",
        "_last_checkpoint_id",
        "_overwritten",
        "_steps_since_snapshot",
    )

    def __init__(
        self,
        operator: Callable[[list[Value], Any], list[Value]],
        typ: type = list,
        *,
        snapshot_every: int | None = None,
    ) -> None:
        typ = _strip_extras(typ)
        if typ in (
            collections.abc.Sequence,
            collections.abc.MutableSequence,
        ):
            typ = list
        super().__init__(typ)
        self.operator = operator
        self.snapshot_every = snapshot_every
        try:
            self.value: list[Value] = typ()
        except Exception:
            self.value = []
        self._pending: list[Any] = []
        self._base_version: str | None = None
        self._last_checkpoint_id: str | None = None
        self._overwritten: bool = False
        self._steps_since_snapshot: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeltaChannel):
            return False
        if self.snapshot_every != other.snapshot_every:
            return False
        if (
            self.operator.__name__ != "<lambda>"
            and other.operator.__name__ != "<lambda>"
        ):
            return self.operator is other.operator
        return True

    @property
    def ValueType(self) -> Any:
        return list[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Any:
        return self.typ | list[self.typ]  # type: ignore[name-defined]

    @property
    def checkpoint_hydration_kind(self) -> Literal["delta"]:
        return "delta"

    def copy(self) -> Self:
        new = DeltaChannel(self.operator, self.typ, snapshot_every=self.snapshot_every)
        new.key = self.key
        new.value = _copy_value(self.value)
        new._pending = self._pending[:]
        new._base_version = self._base_version
        new._last_checkpoint_id = self._last_checkpoint_id
        new._overwritten = self._overwritten
        new._steps_since_snapshot = self._steps_since_snapshot
        return new

    def from_checkpoint(self, checkpoint: Any) -> Self:
        new = DeltaChannel(self.operator, self.typ, snapshot_every=self.snapshot_every)
        new.key = self.key
        if checkpoint is MISSING:
            pass
        elif isinstance(checkpoint, DeltaChainValue):
            accumulated: list[Value] = (
                checkpoint.base if checkpoint.base is not None else new.typ()
            )
            for step_writes in checkpoint.deltas:
                for write in step_writes:
                    accumulated = new.operator(accumulated, write)
            new.value = accumulated
            # Seed the counter from actual chain depth so rehydration fires at
            # the right time regardless of how many prior invocations there were.
            new._steps_since_snapshot = len(checkpoint.deltas)
        elif isinstance(checkpoint, DeltaValue):
            # Should never reach here — checkpoint hydration should assemble
            # DeltaValues into DeltaChainValue before calling from_checkpoint.
            raise AssertionError(
                "DeltaChannel.from_checkpoint received a raw DeltaValue. "
                "This is a bug in checkpoint hydration — chain assembly should "
                "have occurred before from_checkpoint was called."
            )
        else:
            # Backwards compat: plain value from old BinaryOperatorAggregate checkpoint
            # or a full snapshot emitted by DeltaChannel.
            new.value = _copy_value(checkpoint)
        new._pending = []
        new._base_version = None  # set by the subsequent after_checkpoint() call
        new._overwritten = False
        return new

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        seen_overwrite = False
        for value in values:
            is_overwrite, overwrite_value = _get_overwrite(value)
            if is_overwrite:
                if seen_overwrite:
                    from langgraph.errors import (
                        ErrorCode,
                        InvalidUpdateError,
                        create_error_message,
                    )

                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
                self.value = (
                    _copy_value(overwrite_value)
                    if overwrite_value is not None
                    else self.typ()
                )
                self._pending = (
                    [] if overwrite_value is None else [_copy_value(self.value)]
                )
                self._overwritten = True
                seen_overwrite = True
            elif not seen_overwrite:
                base = self.typ() if self.value is MISSING else self.value
                self.value = self.operator(base, value)
                self._pending.append(value)
        return True

    def get(self) -> list[Value]:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        if (
            self.snapshot_every is not None
            and self._steps_since_snapshot >= self.snapshot_every
        ):
            # Emit a full snapshot to cap chain depth at snapshot_every.
            # The saver stores this as a plain (non-diff) blob, so future
            # deltas will chain back to it and traversal depth resets to 1.
            return _copy_value(self.value)
        return DeltaValue(
            delta=self._pending[:],
            prev_checkpoint_id=None if self._overwritten else self._last_checkpoint_id,
        )

    def after_checkpoint(self, version: Any, checkpoint_id: str | None = None) -> None:
        if version != self._base_version:
            if self._base_version is None:
                pass  # First call after from_checkpoint — anchor without counting a step.
            elif self.snapshot_every is not None:
                if self._steps_since_snapshot >= self.snapshot_every:
                    self._steps_since_snapshot = 0
                else:
                    self._steps_since_snapshot += 1
            self._base_version = version
            self._last_checkpoint_id = checkpoint_id
            self._pending = []
            self._overwritten = False
