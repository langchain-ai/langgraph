"""Append-only log channels (transcript log prototype).

!!! warning "Prototype"

    The transcript log is a prototype. Its API and storage may change.

A log channel is append-only and has no reducer: the only write is "append
these entries". Its entries are not stored in checkpoints. A checkpoint's
value for the channel is a marker (`LogStep`) for the step that produced its
version, and the entries live in the saver's log storage, read with
`BaseCheckpointSaver.read_log`.

Two channel types share that storage:

- `Log` holds JSON values. Anything that isn't JSON is rejected.
- `MessagesLog` holds LangChain messages, stored as canonical JSON. Edit ops
  (`RemoveMessage`, `Overwrite`, re-adding an existing id) are rejected; a
  graph that needs a different model context builds it as a projection.

A thread whose field held a plain list before it became a log channel (for
example `Annotated[list, add_messages]` changed to `MessagesLog()`) keeps
working: the stored list becomes the log's first entries, the "seed". The
seed is committed with the first checkpoint saved after the change, and
checkpoints from before the change still read as the list they stored.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Sequence
from typing import Any, NamedTuple

from langgraph.checkpoint.serde.types import LogAppend, LogStep
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import _get_overwrite
from langgraph.errors import EmptyChannelError, InvalidUpdateError

__all__ = ("Log", "LogRef", "MessagesLog")


class LogRef(NamedTuple):
    """What state reads return for a log channel: the log's length at the
    checkpoint. Read the entries with `read_log`."""

    length: int


def _check_json(value: Any, path: str = "entry") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidUpdateError(f"Log {path} is not JSON: {value!r}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _check_json(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for k, item in value.items():
            if not isinstance(k, str):
                raise InvalidUpdateError(f"Log {path} has a non-string key: {k!r}")
            _check_json(item, f"{path}.{k}")
        return
    raise InvalidUpdateError(
        f"Log {path} is not JSON: {type(value).__name__} {value!r}"
    )


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class Log(BaseChannel[list, list, LogStep]):
    """An append-only log of JSON entries.

    Write a list of entries to append them. Nodes read the whole log as a
    list. State reads return a `LogRef` instead of the entries.
    """

    __slots__ = ("entries", "base", "marker", "loaded", "extra", "seed", "seeding")

    def __init__(self, typ: Any = list) -> None:
        super().__init__(typ)
        self.entries: list[Any] = []
        # length covered by the last committed step (the checkpoint's leaf)
        self.base = 0
        self.marker: LogStep | None = None
        # False when built from a checkpoint without reading the log
        self.loaded = True
        # entries appended to a channel that was not loaded
        self.extra = 0
        # True while the entries hold a plain list stored before the field
        # became a log channel, not committed to the log yet
        self.seed = False
        # True from the commit that includes the seed until the loop has
        # stamped that checkpoint's step
        self.seeding = False

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    @property
    def ValueType(self) -> Any:
        return list

    @property
    def UpdateType(self) -> Any:
        return list

    # -- encoding ----------------------------------------------------------

    def encode(self, entry: Any) -> Any:
        """An entry as the JSON value stored for it."""
        _check_json(entry)
        return entry

    def decode(self, data: Any) -> Any:
        """A stored JSON value as the entry nodes see."""
        return data

    def coerce(self, value: Any) -> list[Any]:
        """A write as the list of entries it appends."""
        if not isinstance(value, list):
            raise InvalidUpdateError(
                f"Log channel {self.key!r} takes a list of entries, got "
                f"{type(value).__name__}"
            )
        return value

    def append_value(self, value: Any, exec_: str) -> LogAppend:
        """A write as the `LogAppend` handed to the saver."""
        self._reject_overwrite(value)
        texts = tuple(_dumps(self.encode(e)) for e in self.coerce(value))
        return LogAppend(self.base, texts, exec_)

    # -- channel protocol --------------------------------------------------

    def _new(self) -> Self:
        new = self.__class__.__new__(self.__class__)
        BaseChannel.__init__(new, self.typ, self.key)
        new.entries = []
        new.base = 0
        new.marker = None
        new.loaded = True
        new.extra = 0
        new.seed = False
        new.seeding = False
        return new

    def copy(self) -> Self:
        new = self._new()
        new.entries = list(self.entries)
        new.base = self.base
        new.marker = self.marker
        new.loaded = self.loaded
        new.extra = self.extra
        new.seed = self.seed
        new.seeding = self.seeding
        return new

    def from_checkpoint(self, checkpoint: Any) -> Self:
        """A channel that knows its marker but not its entries.

        State reads build channels this way; `get` then returns a `LogRef`.
        The loop builds log channels with `from_log` instead.

        A plain list is what the field stored before it became a log channel.
        The channel then holds it as its seed: uncommitted entries that the
        next commit writes to the log, ahead of anything appended after.
        """
        new = self._new()
        if isinstance(checkpoint, LogStep):
            new.marker = checkpoint.stripped()
            new.base = checkpoint.end
            new.loaded = False
        elif isinstance(checkpoint, list):
            new.entries = self._seed_entries(checkpoint)
            new.seed = bool(new.entries)
        elif checkpoint is not MISSING and checkpoint is not None:
            raise InvalidUpdateError(
                f"Log channel {self.key!r} can't start from {type(checkpoint).__name__}"
            )
        return new

    def _seed_entries(self, stored: list[Any]) -> list[Any]:
        """The entries a stored plain list becomes."""
        return list(stored)

    def from_log(self, marker: Any, entries: Sequence[Any]) -> Self:
        """A loaded channel: the marker, and the entries read from the log."""
        new = self._new()
        if isinstance(marker, LogStep):
            if len(entries) != marker.end:
                raise ValueError(
                    f"Log channel {self.key!r}: read {len(entries)} entries, "
                    f"the checkpoint has {marker.end}"
                )
            new.marker = marker.stripped()
            new.base = marker.end
        new.entries = [self.decode(e) for e in entries]
        return new

    def get(self) -> Any:
        if not self.loaded:
            return LogRef(self.base + self.extra)
        if not self.entries:
            raise EmptyChannelError()
        return self.entries

    def is_available(self) -> bool:
        return (not self.loaded and self.base + self.extra > 0) or bool(self.entries)

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        appended: list[Any] = []
        for value in values:
            if isinstance(value, LogAppend):
                # a write read back from the saver (pending writes on resume)
                appended.extend(self.decode(json.loads(t)) for t in value.entries)
                continue
            self._reject_overwrite(value)
            for entry in self.coerce(value):
                self.encode(entry)  # validate before anything is applied
                appended.append(entry)
        if not appended:
            return False
        self._check_appends(appended)
        if not self.loaded:
            self.extra += len(appended)
        else:
            self.entries.extend(appended)
        return True

    def _check_appends(self, appended: list[Any]) -> None:
        pass

    def _reject_overwrite(self, value: Any) -> None:
        if _get_overwrite(value)[0]:
            raise InvalidUpdateError(
                f"Log channel {self.key!r} is append-only: Overwrite is not supported"
            )

    def checkpoint(self) -> Any:
        return MISSING if self.marker is None else self.marker

    def commit(self) -> LogStep | None:
        """The marker for a checkpoint about to be saved.

        Entries appended since the last commit become a new `LogStep`
        (carrying them inline; the loop strips them when they already reached
        the saver through `put_writes`), and the channel's base moves to its
        end.
        """
        if self.loaded and len(self.entries) > self.base:
            texts = tuple(_dumps(self.encode(e)) for e in self.entries[self.base :])
            step = LogStep(self.base, len(self.entries) - self.base, "", texts)
            self.base = len(self.entries)
            self.marker = step.stripped()
            self.seeding, self.seed = self.seed, False
            return step
        return self.marker


class MessagesLog(Log):
    """An append-only log of LangChain messages.

    Messages are stored as canonical JSON (`message_to_dict`). Every message
    has an id; writing an id that is already in the log is an error, as are
    `RemoveMessage` and `Overwrite`.
    """

    __slots__ = ()

    def encode(self, entry: Any) -> Any:
        from langchain_core.messages import BaseMessage, RemoveMessage, message_to_dict

        if isinstance(entry, RemoveMessage):
            raise InvalidUpdateError(
                f"MessagesLog {self.key!r} is append-only: RemoveMessage is not "
                "supported. To give the model a different set of messages, build "
                "them when calling the model instead of changing stored history."
            )
        if not isinstance(entry, BaseMessage):
            raise InvalidUpdateError(
                f"MessagesLog {self.key!r} takes messages, got {type(entry).__name__}"
            )
        data = message_to_dict(entry)
        _check_json(data, "message")
        return data

    def decode(self, data: Any) -> Any:
        from langchain_core.messages import messages_from_dict

        return messages_from_dict([data])[0]

    def _seed_entries(self, stored: list[Any]) -> list[Any]:
        # the list `add_messages` stored: already merged, so ids are unique
        # and there are no `RemoveMessage`s
        return self.coerce(stored)

    def coerce(self, value: Any) -> list[Any]:
        from langchain_core.messages import convert_to_messages

        items = value if isinstance(value, list) else [value]
        messages = convert_to_messages(items)
        for m in messages:
            if not m.id:
                m.id = str(uuid.uuid4())
        return messages

    def _check_appends(self, appended: list[Any]) -> None:
        seen = {m.id for m in self.entries} if self.loaded else set()
        for m in appended:
            if m.id in seen:
                raise InvalidUpdateError(
                    f"MessagesLog {self.key!r} already has a message with id "
                    f"{m.id!r}: the log is append-only, so a message can't be "
                    "replaced. Return only new messages from a node, not the "
                    "whole list."
                )
            seen.add(m.id)
