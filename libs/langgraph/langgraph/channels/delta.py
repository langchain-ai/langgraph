from __future__ import annotations

import collections.abc
from collections.abc import Callable, Sequence
from typing import Any, Generic

from langgraph.checkpoint.base import DeltaChainValue, DeltaValue
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.channels.binop import _get_overwrite, _strip_extras
from langgraph.errors import EmptyChannelError

__all__ = ("DeltaChannel",)


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

    def copy(self) -> Self:
        new = DeltaChannel(self.operator, self.typ, snapshot_every=self.snapshot_every)
        new.key = self.key
        new.value = self.value if self.value is MISSING else self.value.copy()
        new._pending = self._pending[:]
        new._base_version = self._base_version
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
            raise ValueError(
                f"Channel '{self.key}' uses DeltaChannel but the checkpointer "
                "does not support incremental channel storage. "
                "Use InMemorySaver or PostgresSaver, or remove DeltaChannel from your schema."
            )
        else:
            # Backwards compat: plain list from old BinaryOperatorAggregate checkpoint.
            new.value = list(checkpoint)
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
                    list(overwrite_value) if overwrite_value is not None else self.typ()
                )
                self._pending = list(self.value)
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
            return list(self.value)
        return DeltaValue(delta=self._pending[:])

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
            self._pending = []
            self._overwritten = False
