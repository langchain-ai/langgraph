from __future__ import annotations

import base64
import re
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from datetime import timedelta
from functools import cached_property
from typing import (
    Any,
)

from langchain_core.runnables import Runnable, RunnableConfig

from langgraph._internal._config import merge_configs
from langgraph._internal._constants import CONF, CONFIG_KEY_READ
from langgraph._internal._runnable import RunnableCallable, RunnableSeq
from langgraph._internal._timeout import coerce_timeout_policy
from langgraph.pregel._utils import find_subgraph_pregel
from langgraph.pregel._write import ChannelWrite
from langgraph.pregel.protocol import PregelProtocol
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy

READ_TYPE = Callable[[str | Sequence[str], bool], Any | dict[str, Any]]
INPUT_CACHE_KEY_TYPE = tuple[Callable[..., Any], tuple[str, ...]]

_DANGEROUS_PATTERNS = [
    re.compile(r"(?i)(ignore\s+(previous|prior|above|all)\s+(instructions?|prompts?|context))"),
    re.compile(r"(?i)(system\s*prompt|you\s+are\s+now|act\s+as\s+(?:a\s+)?(?:different|new|another))"),
    re.compile(r"(?i)(exec\s*\(|eval\s*\(|subprocess|os\.system|shell\s*=\s*True)"),
    re.compile(r"(?i)(rm\s+-rf|del\s+/[sqf]|format\s+c:|mkfs\b|dd\s+if=)"),
    re.compile(r"(?i)(base64\s*decode|atob\s*\(|from_base64)"),
    re.compile(r"(?i)(\bpowershell\b|\bcmd\.exe\b|\b/bin/sh\b|\b/bin/bash\b)"),
    re.compile(r"(?i)(import\s+os|import\s+subprocess|import\s+sys\b)"),
    re.compile(r"(?i)(jailbreak|dan\s+mode|developer\s+mode\s+enabled)"),
]

_LEETSPEAK_PATTERN = re.compile(r"(?i)\b[a-z0-9]*[013457][a-z0-9]*\b")
_EXCESSIVE_LEETSPEAK_THRESHOLD = 5

_MAX_BASE64_CHUNK_LENGTH = 100


def _is_suspicious_base64(text: str) -> bool:
    """Check if text contains suspicious base64-encoded content."""
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
    for match in b64_pattern.finditer(text):
        chunk = match.group(0)
        if len(chunk) > _MAX_BASE64_CHUNK_LENGTH:
            try:
                decoded = base64.b64decode(chunk + "==").decode("utf-8", errors="ignore")
                for pattern in _DANGEROUS_PATTERNS:
                    if pattern.search(decoded):
                        return True
            except Exception:
                pass
    return False


def _contains_leetspeak_commands(text: str) -> bool:
    """Heuristic check for excessive leetspeak that may obfuscate commands."""
    matches = _LEETSPEAK_PATTERN.findall(text)
    leet_chars = re.compile(r"[013457]")
    leet_heavy = [m for m in matches if len(leet_chars.findall(m)) >= 2]
    return len(leet_heavy) >= _EXCESSIVE_LEETSPEAK_THRESHOLD


def _sanitize_and_validate_input(input: Any) -> Any:
    """Sanitize and validate input before passing to the AI model.

    Raises ValueError if the input contains potentially malicious content.
    Returns the input unchanged if it passes validation.
    """
    if input is None:
        return input

    text_to_check: str | None = None

    if isinstance(input, str):
        text_to_check = input
    elif isinstance(input, dict):
        # Check string values in dict inputs
        parts = []
        for v in input.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        for sv in item.values():
                            if isinstance(sv, str):
                                parts.append(sv)
        text_to_check = " ".join(parts) if parts else None
    elif isinstance(input, list):
        parts = []
        for item in input:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        parts.append(v)
        text_to_check = " ".join(parts) if parts else None

    if text_to_check is not None:
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(text_to_check):
                raise ValueError(
                    "Input validation failed: potentially malicious content detected. "
                    "The input contains patterns that may attempt to manipulate the AI model "
                    "or execute malicious commands."
                )

        if _is_suspicious_base64(text_to_check):
            raise ValueError(
                "Input validation failed: suspicious base64-encoded content detected. "
                "The input may contain obfuscated malicious instructions."
            )

        if _contains_leetspeak_commands(text_to_check):
            raise ValueError(
                "Input validation failed: suspicious obfuscated content detected. "
                "The input may contain obfuscated malicious instructions."
            )

    return input


class ChannelRead(RunnableCallable):
    """Implements the logic for reading state from CONFIG_KEY_READ.
    Usable both as a runnable as well as a static method to call imperatively."""

    channel: str | list[str]

    fresh: bool = False

    mapper: Callable[[Any], Any] | None = None

    def __init__(
        self,
        channel: str | list[str],
        *,
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        super().__init__(
            func=self._read,
            afunc=self._aread,
            tags=tags,
            name=None,
            trace=False,
        )
        self.fresh = fresh
        self.mapper = mapper
        self.channel = channel

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        if name:
            pass
        elif isinstance(self.channel, str):
            name = f"ChannelRead<{self.channel}>"
        else:
            name = f"ChannelRead<{','.join(self.channel)}>"
        return super().get_name(suffix, name=name)

    def _read(self, _: Any, config: RunnableConfig) -> Any:
        return self.do_read(
            config, select=self.channel, fresh=self.fresh, mapper=self.mapper
        )

    async def _aread(self, _: Any, config: RunnableConfig) -> Any:
        return self.do_read(
            config, select=self.channel, fresh=self.fresh, mapper=self.mapper
        )

    @staticmethod
    def do_read(
        config: RunnableConfig,
        *,
        select: str | list[str],
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
    ) -> Any:
        try:
            read: READ_TYPE = config[CONF][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                "Not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        if mapper:
            return mapper(read(select, fresh))
        else:
            return read(select, fresh)


DEFAULT_BOUND = RunnableCallable(lambda input: input)


class PregelNode:
    """A node in a Pregel graph. This won't be invoked as a runnable by the graph
    itself, but instead acts as a container for the components necessary to make
    a PregelExecutableTask for a node."""

    channels: str | list[str]
    """The channels that will be passed as input to `bound`.
    If a str, the node will be invoked with its value if it isn't empty.
    If a list, the node will be invoked with a dict of those channels' values."""

    triggers: list[str]
    """If any of these channels is written to, this node will be triggered in
    the next step."""

    mapper: Callable[[Any], Any] | None
    """A function to transform the input before passing it to `bound`."""

    writers: list[Runnable]
    """A list of writers that will be executed after `bound`, responsible for
    taking the output of `bound` and writing it to the appropriate channels."""

    bound: Runnable[Any, Any]
    """The main logic of the node. This will be invoked with the input from 
    `channels`."""

    retry_policy: Sequence[RetryPolicy] | None
    """The retry policies to use when invoking the node."""

    cache_policy: CachePolicy | None
    """The cache policy to use when invoking the node."""

    timeout: TimeoutPolicy | None
    """Timeout policy for a single invocation.

    If exceeded, `NodeTimeoutError` is raised and the retry policy (if any)
    decides whether to retry. Supported only for async nodes.
    """

    tags: Sequence[str] | None
    """Tags to attach to the node for tracing."""

    metadata: Mapping[str, Any] | None
    """Metadata to attach to the node for tracing."""

    is_error_handler: bool
    """Whether this node is registered as an error handler node."""

    error_handler_node: str | None
    """Optional handler node name for failures from this node."""

    subgraphs: Sequence[PregelProtocol]
    """Subgraphs used by the node."""

    def __init__(
        self,
        *,
        channels: str | list[str],
        triggers: Sequence[str],
        mapper: Callable[[Any], Any] | None = None,
        writers: list[Runnable] | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        bound: Runnable[Any, Any] | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        is_error_handler: bool = False,
        error_handler_node: str | None = None,
        subgraphs: Sequence[PregelProtocol] | None = None,
        timeout: float | timedelta | TimeoutPolicy | None = None,
    ) -> None:
        self.channels = channels
        self.triggers = list(triggers)
        self.mapper = mapper
        self.writers = writers or []
        self.bound = bound if bound is not None else DEFAULT_BOUND
        self.cache_policy = cache_policy
        if isinstance(retry_policy, RetryPolicy):
            self.retry_policy = (retry_policy,)
        else:
            self.retry_policy = retry_policy
        self.timeout = coerce_timeout_policy(timeout)
        self.tags = tags
        self.metadata = metadata
        self.is_error_handler = is_error_handler
        self.error_handler_node = error_handler_node
        if subgraphs is not None:
            self.subgraphs = subgraphs
        elif self.bound is not DEFAULT_BOUND:
            try:
                subgraph = find_subgraph_pregel(self.bound)
            except Exception:
                subgraph = None
            if subgraph:
                self.subgraphs = [subgraph]
            else:
                self.subgraphs = []
        else:
            self.subgraphs = []

    def copy(self, update: dict[str, Any]) -> PregelNode:
        attrs = {**self.__dict__, **update}
        # Drop the cached properties
        attrs.pop("flat_writers", None)
        attrs.pop("node", None)
        attrs.pop("input_cache_key", None)
        return PregelNode(**attrs)

    @cached_property
    def flat_writers(self) -> list[Runnable]:
        """Get writers with optimizations applied. Dedupes consecutive ChannelWrites."""
        writers = self.writers.copy()
        while (
            len(writers) > 1
            and isinstance(writers[-1], ChannelWrite)
            and isinstance(writers[-2], ChannelWrite)
        ):
            # we can combine writes if they are consecutive
            # careful to not modify the original writers list or ChannelWrite
            writers[-2] = ChannelWrite(
                writes=writers[-2].writes + writers[-1].writes,
            )
            writers.pop()
        return writers

    @cached_property
    def node(self) -> Runnable[Any, Any] | None:
        """Get a runnable that combines `bound` and `writers`."""
        writers = self.flat_writers
        if self.bound is DEFAULT_BOUND and not writers:
            return None
        elif self.bound is DEFAULT_BOUND and len(writers) == 1:
            return writers[0]
        elif self.bound is DEFAULT_BOUND:
            return RunnableSeq(*writers)
        elif writers:
            return RunnableSeq(self.bound, *writers)
        else:
            return self.bound

    @cached_property
    def input_cache_key(self) -> INPUT_CACHE_KEY_TYPE:
        """Get a cache key for the input to the node.
        This is used to avoid calculating the same input multiple times."""
        return (
            self.mapper,
            tuple(self.channels)
            if isinstance(self.channels, list)
            else (self.channels,),
        )

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Any:
        validated_input = _sanitize_and_validate_input(input)
        self_config: RunnableConfig = {"metadata": self.metadata, "tags": self.tags}
        return self.bound.invoke(
            validated_input,
            merge_configs(self_config, config),
            **kwargs,
        )

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Any:
        validated_input = _sanitize_and_validate_input(input)
        self_config: RunnableConfig = {"metadata": self.metadata, "tags": self.tags}
        return await self.bound.ainvoke(
            validated_input,
            merge_configs(self_config, config),
            **kwargs,
        )

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        validated_input = _sanitize_and_validate_input(input)
        self_config: RunnableConfig = {"metadata": self.metadata, "tags": self.tags}
        yield from self.bound.stream(
            validated_input,
            merge_configs(self_config, config),
            **kwargs,
        )

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        validated_input = _sanitize_and_validate_input(input)
        self_config: RunnableConfig = {"metadata": self.metadata, "tags": self.tags}
        async for item in self.bound.astream(
            validated_input,
            merge_configs(self_config, config),
            **kwargs,
        ):
            yield item