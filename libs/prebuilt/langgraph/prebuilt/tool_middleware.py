"""Composable ``ToolNode`` middleware utilities.

Provides message-level tool-call deduplication and ``wrap_tool_call`` chaining
for applications using :class:`~langgraph.prebuilt.tool_node.ToolNode`.

Key components:

- :func:`build_tool_call_key`: Build a stable deduplication key from tool name
  and arguments.
- :func:`deduplicate_tool_calls`: Remove duplicate tool calls from the trailing
  ``AIMessage`` before tool execution.
- :func:`deduplicate_tool_calls_in_state`: Apply :func:`deduplicate_tool_calls`
  to a graph state mapping.
- :func:`chain_tool_wrappers`: Compose synchronous ``wrap_tool_call`` handlers.
- :func:`chain_async_tool_wrappers`: Compose async ``awrap_tool_call`` handlers.

Example:

    from langgraph.prebuilt import (
        ToolNode,
        chain_tool_wrappers,
        deduplicate_tool_calls_in_state,
    )

    def log_tool(request, execute):
        return execute(request)

    tool_node = ToolNode(tools, wrap_tool_call=chain_tool_wrappers(log_tool))
    state = deduplicate_tool_calls_in_state(state)
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypeAlias

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command
from typing_extensions import TypeIs

from langgraph.prebuilt.tool_node import (
    AsyncToolCallWrapper,
    ToolCallRequest,
    ToolCallWrapper,
)

__all__ = [
    "build_tool_call_key",
    "chain_async_tool_wrappers",
    "chain_tool_wrappers",
    "deduplicate_tool_calls",
    "deduplicate_tool_calls_in_state",
]

ToolCallMapping: TypeAlias = Mapping[str, Any]
ToolCallKeyFn: TypeAlias = Callable[[ToolCallMapping], Any]

ExecuteSync: TypeAlias = Callable[[ToolCallRequest], ToolMessage | Command]
ExecuteAsync: TypeAlias = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]

AnyToolCallWrapper: TypeAlias = ToolCallWrapper | AsyncToolCallWrapper


def build_tool_call_key(
    tool_call: ToolCallMapping,
    *,
    key: ToolCallKeyFn | None = None,
) -> Any:
    """Build a stable deduplication key for a tool call.

    This helper canonicalizes tool name and arguments so equivalent calls with
    different argument ordering compare equal. Custom key functions can override
    the default strategy when domain-specific identity is required.

    Args:
        tool_call: Tool call mapping with ``name`` and ``args`` keys.
        key: Optional custom key function. When omitted, uses
            ``(name, canonical_json(args))``.

    Returns:
        Hashable key identifying the tool invocation.
    """
    if key is not None:
        return key(tool_call)

    name = str(tool_call.get("name", ""))
    args = tool_call.get("args") or {}
    if not isinstance(args, Mapping):
        args = {"value": args}
    args_digest = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    return (name, args_digest)


def deduplicate_tool_calls(
    messages: Sequence[BaseMessage],
    *,
    key: ToolCallKeyFn | None = None,
) -> list[BaseMessage]:
    """Remove duplicate tool calls from the final ``AIMessage``.

    Language models sometimes emit multiple identical tool calls in parallel.
    This utility keeps the first occurrence of each unique call and drops later
    duplicates from the trailing ``AIMessage`` only. Earlier messages are left
    unchanged.

    Args:
        messages: Conversation messages, typically from graph state.
        key: Optional custom key function passed to :func:`build_tool_call_key`.

    Returns:
        A new message list with duplicates removed from the trailing AI message.
        When no deduplication is required, returns a shallow copy of ``messages``.
    """
    if not messages:
        return list(messages)

    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return list(messages)

    seen: set[Any] = set()
    unique_calls: list[dict[str, Any]] = []
    for tool_call in last.tool_calls:
        call_dict = _coerce_tool_call_mapping(tool_call)
        call_key = build_tool_call_key(call_dict, key=key)
        if call_key in seen:
            continue
        seen.add(call_key)
        unique_calls.append(dict(call_dict))

    if len(unique_calls) == len(last.tool_calls):
        return list(messages)

    deduplicated = last.model_copy(update={"tool_calls": unique_calls})
    return [*messages[:-1], deduplicated]


def deduplicate_tool_calls_in_state(
    state: Mapping[str, Any],
    *,
    messages_key: str = "messages",
    key: ToolCallKeyFn | None = None,
) -> dict[str, Any]:
    """Return state with duplicate tool calls removed from the last AI message.

    This is a convenience wrapper around :func:`deduplicate_tool_calls` for
    graph state mappings. Call it immediately before invoking a ``ToolNode`` when
    duplicate parallel tool calls should not be executed.

    Args:
        state: LangGraph state mapping.
        messages_key: State key holding the conversation messages.
        key: Optional custom deduplication key function.

    Returns:
        A shallow copy of ``state``. When the trailing AI message changes, the
        returned mapping contains an updated ``messages`` list; otherwise the
        message objects are unchanged.
    """
    messages = list(state.get(messages_key) or [])
    updated_messages = deduplicate_tool_calls(messages, key=key)
    if _have_same_trailing_tool_calls(messages, updated_messages):
        return dict(state)
    return {**state, messages_key: updated_messages}


def chain_tool_wrappers(*wrappers: ToolCallWrapper) -> ToolCallWrapper:
    """Compose synchronous ``wrap_tool_call`` handlers.

    This utility combines multiple tool interceptors into a single handler
    suitable for ``ToolNode(wrap_tool_call=...)``. Wrappers are applied in
    declaration order: the first wrapper is outermost and sees the request
    first; the last wrapper sits closest to tool execution.

    Args:
        *wrappers: One or more sync tool interceptors.

    Returns:
        Combined interceptor that runs each wrapper in order before executing
        the tool.

    Raises:
        ValueError: If no wrappers are provided.
    """
    if not wrappers:
        msg = "chain_tool_wrappers requires at least one wrapper"
        raise ValueError(msg)

    if len(wrappers) == 1:
        return wrappers[0]

    def chained(
        request: ToolCallRequest, execute: ExecuteSync
    ) -> ToolMessage | Command:
        return _invoke_wrapper_chain(0, request, execute, wrappers)

    return chained


def chain_async_tool_wrappers(
    *wrappers: AnyToolCallWrapper,
) -> AsyncToolCallWrapper:
    """Compose async-compatible ``awrap_tool_call`` handlers.

    This utility combines multiple tool interceptors into a single async handler
    suitable for ``ToolNode(awrap_tool_call=...)``. Sync wrappers are supported
    via a thread-pool bridge so they do not block the event loop when calling
    async tool execution.

    Args:
        *wrappers: One or more sync or async tool interceptors.

    Returns:
        Combined async interceptor that runs each wrapper in order before
        executing the tool.

    Raises:
        ValueError: If no wrappers are provided.
    """
    if not wrappers:
        msg = "chain_async_tool_wrappers requires at least one wrapper"
        raise ValueError(msg)

    if len(wrappers) == 1:
        wrapper = wrappers[0]
        if _is_async_tool_wrapper(wrapper):
            return wrapper
        return _adapt_sync_wrapper_to_async(wrapper)

    async def chained(
        request: ToolCallRequest, execute: ExecuteAsync
    ) -> ToolMessage | Command:
        return await _invoke_async_wrapper_chain(0, request, execute, wrappers)

    return chained


def _is_async_tool_wrapper(wrapper: AnyToolCallWrapper) -> TypeIs[AsyncToolCallWrapper]:
    """Return whether ``wrapper`` is an async tool interceptor."""
    return inspect.iscoroutinefunction(wrapper)


def _have_same_trailing_tool_calls(
    before: Sequence[BaseMessage],
    after: Sequence[BaseMessage],
) -> bool:
    """Return whether two message lists have identical trailing tool calls.

    Args:
        before: Message list before deduplication.
        after: Message list after deduplication.

    Returns:
        ``True`` when both lists have the same length and the trailing AI
        message contains the same tool calls.
    """
    if len(before) != len(after):
        return False
    if not before:
        return True
    last_before = before[-1]
    last_after = after[-1]
    if not isinstance(last_before, AIMessage) or not isinstance(last_after, AIMessage):
        return last_before == last_after
    return last_before.tool_calls == last_after.tool_calls


def _coerce_tool_call_mapping(tool_call: Any) -> dict[str, Any]:
    """Normalize a tool call object to a plain dict.

    Args:
        tool_call: Tool call dict, ``ToolCall`` model, or compatible object.

    Returns:
        Tool call mapping with ``name``, ``args``, ``id``, and ``type`` keys.
    """
    if isinstance(tool_call, Mapping):
        return {
            "name": tool_call.get("name", ""),
            "args": dict(tool_call.get("args") or {}),
            "id": tool_call.get("id", ""),
            "type": tool_call.get("type", "tool_call"),
        }
    if hasattr(tool_call, "model_dump"):
        dumped = tool_call.model_dump()
        return {
            "name": dumped.get("name", ""),
            "args": dict(dumped.get("args") or {}),
            "id": dumped.get("id", ""),
            "type": dumped.get("type", "tool_call"),
        }
    return {
        "name": getattr(tool_call, "name", ""),
        "args": dict(getattr(tool_call, "args", {}) or {}),
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "tool_call"),
    }


def _invoke_wrapper_chain(
    index: int,
    request: ToolCallRequest,
    execute: ExecuteSync,
    wrappers: Sequence[ToolCallWrapper],
) -> ToolMessage | Command:
    """Invoke a sync wrapper chain recursively.

    Args:
        index: Current wrapper index in ``wrappers``.
        request: Tool execution request.
        execute: Final sync execute callable passed to the innermost wrapper.
        wrappers: Ordered sync wrappers to apply.

    Returns:
        Tool execution result from the composed wrapper chain.
    """
    if index >= len(wrappers):
        return execute(request)

    wrapper = wrappers[index]

    def next_execute(next_request: ToolCallRequest) -> ToolMessage | Command:
        return _invoke_wrapper_chain(index + 1, next_request, execute, wrappers)

    return wrapper(request, next_execute)


async def _invoke_async_wrapper_chain(
    index: int,
    request: ToolCallRequest,
    execute: ExecuteAsync,
    wrappers: Sequence[AnyToolCallWrapper],
) -> ToolMessage | Command:
    """Invoke an async wrapper chain recursively.

    Args:
        index: Current wrapper index in ``wrappers``.
        request: Tool execution request.
        execute: Final async execute callable passed to the innermost wrapper.
        wrappers: Ordered sync or async wrappers to apply.

    Returns:
        Tool execution result from the composed wrapper chain.
    """
    if index >= len(wrappers):
        return await execute(request)

    wrapper = wrappers[index]

    async def next_execute(next_request: ToolCallRequest) -> ToolMessage | Command:
        return await _invoke_async_wrapper_chain(
            index + 1, next_request, execute, wrappers
        )

    if _is_async_tool_wrapper(wrapper):
        return await wrapper(request, next_execute)

    loop = asyncio.get_running_loop()
    return await asyncio.to_thread(
        wrapper,
        request,
        _make_sync_execute_bridge(next_execute, loop),
    )


def _make_sync_execute_bridge(
    async_execute: ExecuteAsync,
    loop: asyncio.AbstractEventLoop,
) -> ExecuteSync:
    """Bridge async tool execution into a sync callable.

    Args:
        async_execute: Async execute callable from the wrapper chain.
        loop: Event loop used to schedule ``async_execute``.

    Returns:
        Sync callable that blocks until ``async_execute`` completes.
    """

    def sync_execute(request: ToolCallRequest) -> ToolMessage | Command:
        future = asyncio.run_coroutine_threadsafe(async_execute(request), loop)
        return future.result()

    return sync_execute


def _adapt_sync_wrapper_to_async(wrapper: ToolCallWrapper) -> AsyncToolCallWrapper:
    """Adapt a sync wrapper for ``ToolNode(awrap_tool_call=...)``.

    Args:
        wrapper: Sync tool interceptor.

    Returns:
        Async wrapper that runs ``wrapper`` in a worker thread.
    """

    async def async_wrapper(
        request: ToolCallRequest,
        execute: ExecuteAsync,
    ) -> ToolMessage | Command:
        loop = asyncio.get_running_loop()
        return await asyncio.to_thread(
            wrapper,
            request,
            _make_sync_execute_bridge(execute, loop),
        )

    return async_wrapper
