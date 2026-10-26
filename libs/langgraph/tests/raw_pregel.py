"""Build graphs directly on the Pregel engine, for tests of engine behavior.

LangGraph does not expose this: applications build graphs with `StateGraph` or
`@entrypoint`. Tests use it to set up channels, triggers and writes that those
APIs do not produce on their own.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from langchain_core.runnables.base import Input, Output
from typing_extensions import Self

from langgraph._internal._runnable import (
    Runnable,
    RunnableLike,
    RunnableSeq,
    coerce_to_runnable,
)
from langgraph._internal._timeout import coerce_timeout_policy
from langgraph.pregel import Pregel
from langgraph.pregel._read import DEFAULT_BOUND, PregelNode
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy

__all__ = ("NodeBuilder", "RawPregel")

_WriteValue = Callable[[Input], Output] | Any


class NodeBuilder:
    """Builds a `PregelNode` that reads and writes named channels."""

    __slots__ = (
        "_channels",
        "_triggers",
        "_tags",
        "_metadata",
        "_writes",
        "_bound",
        "_retry_policy",
        "_cache_policy",
        "_timeout",
    )

    _channels: str | list[str]
    _triggers: list[str]
    _tags: list[str]
    _metadata: dict[str, Any]
    _writes: list[ChannelWriteEntry]
    _bound: Runnable
    _retry_policy: list[RetryPolicy]
    _cache_policy: CachePolicy | None
    _timeout: TimeoutPolicy | None

    def __init__(
        self,
    ) -> None:
        self._channels = []
        self._triggers = []
        self._tags = []
        self._metadata = {}
        self._writes = []
        self._bound = DEFAULT_BOUND
        self._retry_policy = []
        self._cache_policy = None
        self._timeout = None

    def subscribe_only(
        self,
        channel: str,
    ) -> Self:
        """Subscribe to a single channel."""
        if not self._channels:
            self._channels = channel
        else:
            raise ValueError(
                "Cannot subscribe to single channels when other channels are already subscribed to"
            )

        self._triggers.append(channel)

        return self

    def subscribe_to(
        self,
        *channels: str,
        read: bool = True,
    ) -> Self:
        """Add channels to subscribe to.

        Node will be invoked when any of these channels are updated, with a dict of the
        channel values as input.

        Args:
            channels: Channel name(s) to subscribe to
            read: If `True`, the channels will be included in the input to the node.
                Otherwise, they will trigger the node without being sent in input.

        Returns:
            Self for chaining
        """
        if isinstance(self._channels, str):
            raise ValueError(
                "Cannot subscribe to channels when subscribed to a single channel"
            )
        if read:
            if not self._channels:
                self._channels = list(channels)
            else:
                self._channels.extend(channels)

        if isinstance(channels, str):
            self._triggers.append(channels)
        else:
            self._triggers.extend(channels)

        return self

    def read_from(
        self,
        *channels: str,
    ) -> Self:
        """Adds the specified channels to read from, without subscribing to them."""
        assert isinstance(self._channels, list), (
            "Cannot read additional channels when subscribed to single channels"
        )
        self._channels.extend(channels)
        return self

    def do(
        self,
        node: RunnableLike,
    ) -> Self:
        """Adds the specified node."""
        if self._bound is not DEFAULT_BOUND:
            self._bound = RunnableSeq(
                self._bound, coerce_to_runnable(node, name=None, trace=True)
            )
        else:
            self._bound = coerce_to_runnable(node, name=None, trace=True)
        return self

    def write_to(
        self,
        *channels: str | ChannelWriteEntry,
        **kwargs: _WriteValue,
    ) -> Self:
        """Add channel writes.

        Args:
            *channels: Channel names to write to.
            **kwargs: Channel name and value mappings.

        Returns:
            Self for chaining
        """
        self._writes.extend(
            ChannelWriteEntry(c) if isinstance(c, str) else c for c in channels
        )
        self._writes.extend(
            ChannelWriteEntry(k, mapper=v)
            if callable(v)
            else ChannelWriteEntry(k, value=v)
            for k, v in kwargs.items()
        )

        return self

    def meta(self, *tags: str, **metadata: Any) -> Self:
        """Add tags or metadata to the node."""
        self._tags.extend(tags)
        self._metadata.update(metadata)
        return self

    def add_retry_policies(self, *policies: RetryPolicy) -> Self:
        """Adds retry policies to the node."""
        self._retry_policy.extend(policies)
        return self

    def add_cache_policy(self, policy: CachePolicy) -> Self:
        """Adds cache policies to the node."""
        self._cache_policy = policy
        return self

    def set_timeout(self, timeout: float | timedelta | TimeoutPolicy | None) -> Self:
        """Set the per-attempt timeout policy for this node."""
        self._timeout = coerce_timeout_policy(timeout)
        return self

    def build(self) -> PregelNode:
        """Builds the node."""
        return PregelNode(
            channels=self._channels,
            triggers=self._triggers,
            tags=self._tags,
            metadata=self._metadata,
            writers=[ChannelWrite(self._writes)],
            bound=self._bound,
            retry_policy=self._retry_policy,
            cache_policy=self._cache_policy,
            timeout=self._timeout,
        )


class RawPregel(Pregel):
    """A `Pregel` built from hand-written nodes and channels.

    Accepts `NodeBuilder` values in `nodes`, and builds them.
    """

    def __init__(self, *, nodes: dict[str, PregelNode | NodeBuilder], **kwargs: Any):
        super().__init__(
            nodes={
                k: v.build() if isinstance(v, NodeBuilder) else v
                for k, v in nodes.items()
            },
            **kwargs,
        )
