"""Messages in an append-only log.

!!! warning "Prototype"

    `MessagesLog` is a prototype that shows the interface and what it means
    for existing graphs. Its storage here is a stand-in, see below.

`MessagesLog` replaces `Annotated[list, add_messages]` for graphs that only
ever add messages. Nodes still read the list and return new messages. What
changes:

- Stored history is append-only. `RemoveMessage`, `REMOVE_ALL_MESSAGES`,
  replacing a message by id and `Overwrite` are errors. A graph that wants
  the model to see something else (a summary, a window) builds those messages
  when it calls the model, and leaves the history as it is.
- Returning the whole list again is an error (its ids are already there),
  instead of silently duplicating history.
- Messages are stored once, not in every checkpoint, and can be read a page
  at a time without the graph (`read_log`).

Existing graphs: `add_messages` keeps working, and its edit ops are
deprecated. A thread written with `add_messages` keeps its history when the
field changes to `MessagesLog`: the stored list is where the log starts.

Storage stand-in: here `MessagesLog` is a `DeltaChannel` that is never
snapshotted on a schedule and whose reducer only appends, so each message is
saved once, with the step that wrote it. The real storage gives the log its
own lineage, so threads can be pruned and pages read at a cost proportional
to the page, and state reads return a marker (`LogRef(length)`) instead of
the list. Neither changes what graph code writes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple, cast

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.binop import _get_overwrite
from langgraph.channels.delta import DeltaChannel
from langgraph.errors import InvalidUpdateError

__all__ = ("LogPage", "MessagesLog", "read_log")


def _to_messages(items: list[Any]) -> list[AnyMessage]:
    return [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(items)
    ]


def _append(state: list[AnyMessage], writes: Sequence[Any]) -> list[AnyMessage]:
    """Append each write's messages. There is nothing else to do: no fold.

    Batching-invariant, as `DeltaChannel` requires.
    """
    if state and isinstance(state[0], BaseMessage):
        value = list(state)
    else:
        # the list a thread stored while the field was `add_messages`
        value = _to_messages(list(state))
    ids = {m.id for m in value}
    for w in writes:
        for m in _to_messages(w if isinstance(w, list) else [w]):
            if isinstance(m, RemoveMessage):
                raise InvalidUpdateError(
                    "MessagesLog is append-only: RemoveMessage is not supported. "
                    "To give the model a different set of messages, build them "
                    "when calling the model instead of changing stored history."
                )
            if m.id in ids:
                raise InvalidUpdateError(
                    f"MessagesLog already has a message with id {m.id!r}: it is "
                    "append-only, so a message can't be replaced. Return only "
                    "new messages from a node, not the whole list."
                )
            ids.add(m.id)
            value.append(m)
    return value


class MessagesLog(DeltaChannel[list]):
    """Messages in an append-only log.

    Example:
        ```python
        class State(TypedDict):
            messages: Annotated[list[AnyMessage], MessagesLog()]
        ```
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(_append, list)

    def _new(self, typ: Any) -> Self:
        return self.__class__()

    def update(self, values: Sequence[Any]) -> bool:
        if any(_get_overwrite(v)[0] for v in values):
            raise InvalidUpdateError(
                "MessagesLog is append-only: Overwrite is not supported."
            )
        return super().update(values)


class LogPage(NamedTuple):
    """A page of a `MessagesLog`, as returned by `read_log`."""

    entries: list[tuple[int, AnyMessage]]
    """`(position, message)` pairs, in order."""
    length: int
    """Number of messages at the checkpoint."""


def read_log(
    checkpointer: BaseCheckpointSaver,
    config: RunnableConfig,
    channel: str = "messages",
    *,
    after: int | None = None,
    before: int | None = None,
    limit: int | None = None,
) -> LogPage:
    """Read a page of a `MessagesLog` at a checkpoint, without the graph.

    `config` names the thread, and optionally the checkpoint (the latest
    otherwise). `after=p` starts after position `p`, `before=p` stops before
    it. With a `limit`, a read with only `after` takes the first `limit`
    messages of that range; any other read takes the last `limit`, so a chat
    UI pages from the end. Positions never change: the log only grows.

    (In the real design this is a checkpointer method, also reachable as
    `graph.read_log` and through the server.)
    """
    tup = checkpointer.get_tuple(config)
    if tup is None or channel not in tup.checkpoint["channel_versions"]:
        return LogPage([], 0)
    stored = tup.checkpoint["channel_values"].get(channel, MISSING)
    ch = MessagesLog()
    if stored is MISSING:
        history = checkpointer.get_delta_channel_history(
            config=tup.config, channels=[channel]
        )[channel]
        ch = ch.from_checkpoint(history.get("seed", MISSING))
        ch.replay_writes(history["writes"])
    else:
        ch = ch.from_checkpoint(stored)
    messages = ch.get()
    length = len(messages)
    lo = 0 if after is None else max(after + 1, 0)
    hi = length if before is None else min(before, length)
    if limit is not None:
        if after is not None and before is None:
            hi = min(hi, lo + limit)
        else:
            lo = max(lo, hi - limit)
    return LogPage([(i, messages[i]) for i in range(lo, max(lo, hi))], length)
