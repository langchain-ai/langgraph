from __future__ import annotations

import collections.abc
from collections.abc import Callable, Sequence
from typing import Any, Generic

from langgraph.checkpoint.base import DiffChainValue, DiffDelta
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.channels.binop import _get_overwrite, _strip_extras
from langgraph.errors import EmptyChannelError

__all__ = ("DiffChannel",)


class DiffChannel(Generic[Value], BaseChannel[list[Value], Value, DiffDelta]):
    """A channel that stores only per-step write deltas in checkpoints.

    Reconstructs the full accumulated list at load time by replaying the
    chain of deltas through the operator. Use with append-style reducers
    (e.g. `add_messages`) on long-running threads to reduce checkpoint
    storage from O(N²) to O(N).

    Requires InMemorySaver or PostgresSaver; SqliteSaver is not supported.

    Usage::

        class State(TypedDict):
            messages: Annotated[list[AnyMessage], DiffChannel(add_messages)]
    """

    __slots__ = (
        "value",
        "operator",
        "rehydrate_every",
        "_pending",
        "_base_version",
        "_overwritten",
        "_steps_since_rehydrate",
    )

    def __init__(
        self,
        operator: Callable[[list[Value], Any], list[Value]],
        typ: type = list,
        *,
        rehydrate_every: int | None = None,
    ) -> None:
        typ = _strip_extras(typ)
        if typ in (
            collections.abc.Sequence,
            collections.abc.MutableSequence,
        ):
            typ = list
        super().__init__(typ)
        self.operator = operator
        self.rehydrate_every = rehydrate_every
        try:
            self.value: list[Value] = typ()
        except Exception:
            self.value = []
        self._pending: list[Any] = []
        self._base_version: str | None = None
        self._overwritten: bool = False
        self._steps_since_rehydrate: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiffChannel):
            return False
        if self.rehydrate_every != other.rehydrate_every:
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
        new = DiffChannel(self.operator, self.typ, rehydrate_every=self.rehydrate_every)
        new.key = self.key
        new.value = self.value[:]
        new._pending = self._pending[:]
        new._base_version = self._base_version
        new._overwritten = self._overwritten
        new._steps_since_rehydrate = self._steps_since_rehydrate
        return new

    def from_checkpoint(self, checkpoint: Any) -> Self:
        new = DiffChannel(self.operator, self.typ, rehydrate_every=self.rehydrate_every)
        new.key = self.key
        if checkpoint is MISSING:
            new.value = []
        elif isinstance(checkpoint, DiffChainValue):
            accumulated: list[Value] = list(checkpoint.base) if checkpoint.base else []
            for step_writes in checkpoint.deltas:
                for write in step_writes:
                    accumulated = new.operator(accumulated, write)
            new.value = accumulated
            # Seed the counter from actual chain depth so rehydration fires at
            # the right time regardless of how many prior invocations there were.
            new._steps_since_rehydrate = len(checkpoint.deltas)
        elif isinstance(checkpoint, DiffDelta):
            raise ValueError(
                "DiffChannel received a raw DiffDelta from the checkpoint saver. "
                "Your saver does not support incremental channel storage. "
                "Use InMemorySaver or PostgresSaver."
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
                    list(overwrite_value) if overwrite_value is not None else []
                )
                self._pending = list(self.value)
                self._overwritten = True
                seen_overwrite = True
            elif not seen_overwrite:
                self.value = self.operator(self.value, value)
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
            self.rehydrate_every is not None
            and self._steps_since_rehydrate >= self.rehydrate_every
        ):
            # Emit a full snapshot to cap chain depth at rehydrate_every.
            # The saver stores this as a plain (non-diff) blob, so future
            # deltas will chain back to it and traversal depth resets to 1.
            return list(self.value)
        return DiffDelta(
            delta=self._pending[:],
            prev_version=None if self._overwritten else self._base_version,
        )

    def after_checkpoint(self, version: Any) -> None:
        if version != self._base_version:
            if self._base_version is None:
                # First call after from_checkpoint — anchor the base version
                # without counting a step (the counter was seeded by from_checkpoint).
                pass
            elif self.rehydrate_every is not None:
                if self._steps_since_rehydrate >= self.rehydrate_every:
                    self._steps_since_rehydrate = 0
                else:
                    self._steps_since_rehydrate += 1
            self._base_version = version
            self._pending = []
            self._overwritten = False
