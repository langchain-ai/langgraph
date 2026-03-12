from __future__ import annotations

import asyncio
import contextvars
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Generic, TypeVar, cast

from langgraph.types import Command, Send

StateT = TypeVar("StateT")

_CURRENT_RUN: contextvars.ContextVar[_GraphEngineRun | None] = contextvars.ContextVar(
    "langgraph_advanced_graph_run", default=None
)


@dataclass(frozen=True)
class _ChannelSpec:
    typ: Any
    maxsize: int


@dataclass(frozen=True)
class ChannelCondition:
    channel: str
    n: int = 1


@dataclass(frozen=True)
class TimerCondition:
    seconds: float


@dataclass(frozen=True)
class AnyOfCondition:
    conditions: tuple[WaitCondition, ...]


WaitCondition = ChannelCondition | TimerCondition | AnyOfCondition


class AdvancedStateGraph(Generic[StateT]):
    """Experimental in-memory graph engine with async channels."""

    def __init__(self, state_schema: type[StateT]) -> None:
        self.state_schema = state_schema
        self._nodes: dict[str, Callable[..., Any]] = {}
        self._channels: dict[str, _ChannelSpec] = {}
        self._entry_point: str | None = None
        self._finish_point: str | None = None

    def add_node(
        self,
        name_or_node: str | Callable[..., Any],
        node: Callable[..., Any] | None = None,
    ) -> str:
        if node is None:
            if not callable(name_or_node):
                raise TypeError("add_node() expects a callable when name is omitted")
            node_name = _infer_node_name(name_or_node)
            node_fn = name_or_node
        else:
            if not isinstance(name_or_node, str):
                raise TypeError("add_node() expects a string node name")
            node_name = name_or_node
            node_fn = node

        if node_name in self._nodes:
            raise ValueError(f"Node `{node_name}` already exists")
        self._nodes[node_name] = node_fn
        return node_name

    def node(self, node: Callable[..., Any]) -> str:
        """Register a node using the function name as node id."""
        return self.add_node(node)

    def add_async_channel(
        self, name: str, typ: Any, maxsize: int | None = None
    ) -> None:
        if name in self._channels:
            raise ValueError(f"Channel `{name}` already exists")
        self._channels[name] = _ChannelSpec(typ=typ, maxsize=maxsize or 0)

    def set_entry_point(self, name_or_node: str | Callable[..., Any]) -> None:
        self._entry_point = self._resolve_node_name(name_or_node)

    def set_finish_point(self, name_or_node: str | Callable[..., Any]) -> None:
        self._finish_point = self._resolve_node_name(name_or_node)

    def add_entry_node(self, node: Callable[..., Any]) -> str:
        node_name = self.add_node(node)
        self.set_entry_point(node_name)
        return node_name

    def add_finish_node(self, node: Callable[..., Any]) -> str:
        node_name = self.add_node(node)
        self.set_finish_point(node_name)
        return node_name

    def _resolve_node_name(self, name_or_node: str | Callable[..., Any]) -> str:
        if isinstance(name_or_node, str):
            return name_or_node
        node_name = _infer_node_name(name_or_node)
        if node_name not in self._nodes:
            self._nodes[node_name] = name_or_node
        return node_name

    def compile(self) -> CompiledGraphEngine[StateT]:
        if self._entry_point is None:
            raise ValueError("Entry point is not set")
        if self._finish_point is None:
            raise ValueError("Finish point is not set")
        if self._entry_point not in self._nodes:
            raise ValueError(f"Entry point node `{self._entry_point}` does not exist")
        if self._finish_point not in self._nodes:
            raise ValueError(f"Finish point node `{self._finish_point}` does not exist")
        return CompiledGraphEngine(
            nodes=dict(self._nodes),
            channels=dict(self._channels),
            entry_point=self._entry_point,
            finish_point=self._finish_point,
        )


class CompiledGraphEngine(Generic[StateT]):
    """Executable runtime for `AdvancedStateGraph`."""

    def __init__(
        self,
        *,
        nodes: dict[str, Callable[..., Any]],
        channels: dict[str, _ChannelSpec],
        entry_point: str,
        finish_point: str,
    ) -> None:
        self._nodes = nodes
        self._channels = channels
        self._entry_point = entry_point
        self._finish_point = finish_point
        self._active_run: _GraphEngineRun | None = None
        self._run_lock = asyncio.Lock()

    async def ainvoke(self, initial_state: StateT) -> StateT:
        async with self._run_lock:
            if self._active_run is not None:
                raise RuntimeError("Graph engine already has an active run")
            run = _GraphEngineRun(
                nodes=self._nodes,
                channel_specs=self._channels,
                entry_point=self._entry_point,
                finish_point=self._finish_point,
            )
            self._active_run = run
        try:
            return await run.run(initial_state)
        finally:
            async with self._run_lock:
                if self._active_run is run:
                    self._active_run = None

    async def apublish_to_channel(self, channel: str, value: Any) -> None:
        run = self._active_run
        if run is None:
            raise RuntimeError("No active graph run to publish to")
        await run.publish(channel, value)


class _GraphEngineRun:
    def __init__(
        self,
        *,
        nodes: dict[str, Callable[..., Any]],
        channel_specs: dict[str, _ChannelSpec],
        entry_point: str,
        finish_point: str,
    ) -> None:
        self._nodes = nodes
        self._entry_point = entry_point
        self._finish_point = finish_point
        self._channels: dict[str, asyncio.Queue[Any]] = {
            name: asyncio.Queue(maxsize=spec.maxsize)
            for name, spec in channel_specs.items()
        }
        self._tasks: set[asyncio.Task[list[Send]]] = set()
        self._finished = False
        self._state: Any = None

    async def run(self, initial_state: StateT) -> StateT:
        self._state = initial_state
        self._schedule(Send(self._entry_point, initial_state))
        try:
            while self._tasks and not self._finished:
                done, _ = await asyncio.wait(
                    self._tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    self._tasks.remove(task)
                    exc = task.exception()
                    if exc is not None:
                        await self._cancel_all_tasks()
                        raise exc
                    sends = task.result()
                    for send in sends:
                        self._schedule(send)
            if self._finished:
                await self._cancel_all_tasks()
            return cast(StateT, self._state)
        finally:
            await self._cancel_all_tasks()

    async def publish(self, channel: str, value: Any) -> None:
        queue = self._get_channel(channel)
        await queue.put(value)

    def publish_nowait(self, channel: str, value: Any) -> None:
        queue = self._get_channel(channel)
        queue.put_nowait(value)

    async def wait_for(self, target: str | WaitCondition, n: int = 1) -> Any:
        if isinstance(target, str):
            return await self._wait_for_channel_values(target, n=n)
        if isinstance(target, ChannelCondition):
            value = await self._wait_for_channel_values(target.channel, n=target.n)
            return {
                "condition": "channel",
                "channel": target.channel,
                "value": value,
            }
        if isinstance(target, TimerCondition):
            await asyncio.sleep(target.seconds)
            return {"condition": "timer", "seconds": target.seconds}
        if isinstance(target, AnyOfCondition):
            return await self._wait_for_any_of(target)
        raise ValueError(f"Unsupported wait condition type: {type(target)!r}")

    async def _wait_for_channel_values(self, channel: str, n: int) -> Any:
        if n < 1:
            raise ValueError("wait_for count `n` must be >= 1")
        queue = self._get_channel(channel)
        if n == 1:
            return await queue.get()
        values: list[Any] = []
        for _ in range(n):
            values.append(await queue.get())
        return values

    async def _wait_for_any_of(self, condition: AnyOfCondition) -> Any:
        if not condition.conditions:
            raise ValueError("any_of() requires at least one condition")

        tasks = [
            asyncio.create_task(self.wait_for(inner_condition, n=1))
            for inner_condition in condition.conditions
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        first = done.pop()
        return first.result()

    def _get_channel(self, channel: str) -> asyncio.Queue[Any]:
        if channel not in self._channels:
            raise ValueError(f"Unknown channel `{channel}`")
        return self._channels[channel]

    def _schedule(self, send: Send) -> None:
        if self._finished:
            return
        task: asyncio.Task[list[Send]] = asyncio.create_task(self._execute_send(send))
        self._tasks.add(task)

    async def _cancel_all_tasks(self) -> None:
        if not self._tasks:
            return
        to_cancel = list(self._tasks)
        for task in to_cancel:
            task.cancel()
        await asyncio.gather(*to_cancel, return_exceptions=True)
        self._tasks.clear()

    async def _execute_send(self, send: Send) -> list[Send]:
        node_name = _resolve_target_name(send.node)
        if node_name not in self._nodes:
            raise ValueError(f"Unknown node `{node_name}`")
        node = self._nodes[node_name]

        token = _CURRENT_RUN.set(self)
        try:
            result = node(send.arg)
            if inspect.isawaitable(result):
                result = await result
        finally:
            _CURRENT_RUN.reset(token)

        if isinstance(result, Command):
            self._apply_update(result.update)
            next_sends = _normalize_goto(result.goto, default_arg=self._state)
        else:
            self._apply_update(result)
            next_sends = _normalize_result_to_sends(result, default_arg=self._state)

        if node_name == self._finish_point:
            self._finished = True
            return []
        return next_sends

    def _apply_update(self, update: Any) -> None:
        if update is None:
            return
        if isinstance(update, Mapping):
            if isinstance(self._state, Mapping):
                # Keep semantics simple: in-place update for mapping-like state.
                cast(dict[str, Any], self._state).update(update)
            return
        if isinstance(update, Sequence) and not isinstance(update, (str, bytes)):
            pairs = list(update)
            if all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                for item in pairs
            ):
                if isinstance(self._state, Mapping):
                    cast(dict[str, Any], self._state).update(
                        cast(dict[str, Any], pairs)
                    )
                return


def _normalize_result_to_sends(result: Any, *, default_arg: Any) -> list[Send]:
    if result is None:
        return []
    if isinstance(result, Send):
        return [result]
    if callable(result):
        return [Send(_infer_node_name(result), default_arg)]
    if isinstance(result, str):
        return [Send(result, default_arg)]
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        sends: list[Send] = []
        for item in result:
            if isinstance(item, Send):
                sends.append(item)
            elif callable(item):
                sends.append(Send(_infer_node_name(item), default_arg))
            elif isinstance(item, str):
                sends.append(Send(item, default_arg))
        return sends
    return []


def _normalize_goto(goto: Any, *, default_arg: Any) -> list[Send]:
    if not goto:
        return []
    if isinstance(goto, Send):
        return [goto]
    if callable(goto):
        return [Send(_infer_node_name(goto), default_arg)]
    if isinstance(goto, str):
        return [Send(goto, default_arg)]
    if isinstance(goto, Sequence):
        sends: list[Send] = []
        for item in goto:
            if isinstance(item, Send):
                sends.append(item)
            elif callable(item):
                sends.append(Send(_infer_node_name(item), default_arg))
            elif isinstance(item, str):
                sends.append(Send(item, default_arg))
        return sends
    return []


async def wait_for(target: str | WaitCondition, n: int = 1) -> Any:
    run = _CURRENT_RUN.get()
    if run is None:
        raise RuntimeError("wait_for() can only be used inside advanced_graph nodes")
    return await run.wait_for(target, n=n)


def publish_to_channel(channel: str, value: Any) -> None:
    run = _CURRENT_RUN.get()
    if run is None:
        raise RuntimeError(
            "publish_to_channel() can only be used inside advanced_graph nodes"
        )
    run.publish_nowait(channel, value)


def channel_condition(channel: str, n: int = 1) -> ChannelCondition:
    if n < 1:
        raise ValueError("channel_condition `n` must be >= 1")
    return ChannelCondition(channel=channel, n=n)


def timer_condition(
    timeout: float | timedelta | None = None,
    *,
    seconds: float | None = None,
    minutes: float | None = None,
) -> TimerCondition:
    if timeout is not None and (seconds is not None or minutes is not None):
        raise ValueError(
            "Provide either `timeout` or named `seconds`/`minutes`, not both"
        )

    if isinstance(timeout, timedelta):
        resolved_seconds = timeout.total_seconds()
    elif isinstance(timeout, (int, float)):
        resolved_seconds = float(timeout)
    else:
        resolved_seconds = 0.0
        if seconds is not None:
            resolved_seconds += float(seconds)
        if minutes is not None:
            resolved_seconds += float(minutes) * 60.0

    if resolved_seconds <= 0:
        raise ValueError("timer_condition must be greater than 0 seconds")
    return TimerCondition(seconds=resolved_seconds)


def any_of(*conditions: WaitCondition) -> AnyOfCondition:
    if not conditions:
        raise ValueError("any_of() requires at least one condition")
    return AnyOfCondition(conditions=tuple(conditions))


def _infer_node_name(node: Callable[..., Any]) -> str:
    node_name = getattr(node, "__name__", "")
    if not node_name or node_name == "<lambda>":
        raise ValueError("Cannot infer node name from anonymous callable")
    return node_name


def _resolve_target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    if callable(target):
        return _infer_node_name(target)
    raise ValueError(f"Unsupported node target type: {type(target)!r}")
