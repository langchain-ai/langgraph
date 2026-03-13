from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Generic, TypeVar, cast

from langgraph_rust_core import PyRustEngine  # type: ignore[import-untyped]

from langgraph.types import Command, Send

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class _ChannelSpec:
    typ: Any


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


WaitCondition = ChannelCondition | TimerCondition


class AdvancedStateGraph(Generic[StateT]):
    """Experimental in-memory graph engine with async channels."""

    def __init__(self, state_schema: type[StateT]) -> None:
        self.state_schema = state_schema
        self._nodes: dict[str, Callable[..., Any]] = {}
        self._async_channels: dict[str, _ChannelSpec] = {}
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

    def add_async_channel(self, name: str, typ: Any) -> None:
        if name in self._async_channels:
            raise ValueError(f"Channel `{name}` already exists")
        self._async_channels[name] = _ChannelSpec(typ=typ)

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
            async_channels=dict(self._async_channels),
            entry_point=self._entry_point,
            finish_point=self._finish_point,
        )


class CompiledGraphEngine(Generic[StateT]):
    """Executable runtime for `AdvancedStateGraph`."""

    def __init__(
        self,
        *,
        nodes: dict[str, Callable[..., Any]],
        async_channels: dict[str, _ChannelSpec],
        entry_point: str,
        finish_point: str,
    ) -> None:
        self._nodes = nodes
        self._async_channels = async_channels
        self._entry_point = entry_point
        self._finish_point = finish_point

    async def ainvoke(self, initial_state: StateT) -> StateT:
        handler = await self.astart(initial_state)
        return await handler

    async def astart(self, initial_state: StateT) -> GraphRunHandler[StateT]:
        run = _GraphEngineRun(
            nodes=self._nodes,
            async_channel_specs=self._async_channels,
            entry_point=self._entry_point,
            finish_point=self._finish_point,
        )
        task = asyncio.create_task(run.run(initial_state))
        return GraphRunHandler(run=run, task=task)


class Context:
    """Per-run context injected into advanced graph nodes."""

    def __init__(self, run: _GraphEngineRun) -> None:
        self._run = run

    async def wait_for(self, target: WaitCondition | AnyOfCondition) -> Any:
        return await self._run.wait_for(target)

    def publish_to_channel(self, channel: str, value: Any) -> None:
        self._run.publish_nowait(channel, value)

    async def apublish_to_channel(self, channel: str, value: Any) -> None:
        await self._run.publish(channel, value)


class GraphRunHandler(Generic[StateT]):
    """Handle for an active in-memory run."""

    def __init__(self, *, run: _GraphEngineRun, task: asyncio.Task[StateT]) -> None:
        self._run = run
        self._task = task

    async def apublish_to_channel(self, channel: str, value: Any) -> None:
        if self._task.done():
            raise RuntimeError("Run has already completed")
        await self._run.publish(channel, value)

    async def aresult(self) -> StateT:
        return await self._task

    def __await__(self) -> Any:
        return self._task.__await__()


class _GraphEngineRun:
    def __init__(
        self,
        *,
        nodes: dict[str, Callable[..., Any]],
        async_channel_specs: dict[str, _ChannelSpec],
        entry_point: str,
        finish_point: str,
    ) -> None:
        self._nodes = nodes
        self._entry_point = entry_point
        self._finish_point = finish_point
        self._rust_engine = PyRustEngine()
        for name in async_channel_specs:
            self._rust_engine.add_async_channel(name)
        self._tasks: set[asyncio.Task[list[Send]]] = set()
        self._finished = False
        self._state: Any = None
        self.context = Context(self)

    async def run(self, initial_state: StateT) -> StateT:
        result_obj = await asyncio.to_thread(
            self._rust_engine.run_graph_py,
            self._entry_point,
            self._finish_point,
            initial_state,
            self._execute_node_for_rust,
        )
        self._state = result_obj
        return cast(StateT, self._state)

    async def publish(self, channel: str, value: Any) -> None:
        await asyncio.to_thread(self._publish_sync, channel, value)

    def publish_nowait(self, channel: str, value: Any) -> None:
        self._publish_sync(channel, value)

    async def wait_for(self, target: WaitCondition | AnyOfCondition) -> Any:
        if isinstance(target, ChannelCondition):
            value = await self._wait_for_channel_values(target.channel, n=target.n)
            return {
                "condition": "channel",
                "channel": target.channel,
                "value": value,
            }
        if isinstance(target, TimerCondition):
            return await asyncio.to_thread(self._rust_engine.wait_timer, target.seconds)
        if isinstance(target, AnyOfCondition):
            return await self._wait_for_any_of(target)
        raise ValueError(f"Unsupported wait condition type: {type(target)!r}")

    async def _wait_for_channel_values(self, channel: str, n: int) -> Any:
        if n < 1:
            raise ValueError("wait_for count `n` must be >= 1")
        event = await asyncio.to_thread(self._rust_engine.wait_channel, channel, n)
        return event["value"]

    async def _wait_for_any_of(self, condition: AnyOfCondition) -> Any:
        if not condition.conditions:
            raise ValueError("any_of() requires at least one condition")
        payload = {
            "conditions": [_condition_to_rust(cond) for cond in condition.conditions]
        }
        return await asyncio.to_thread(self._rust_engine.wait_any_of_obj, payload)

    def _publish_sync(self, channel: str, value: Any) -> None:
        self._rust_engine.publish_obj(channel, value)

    def _execute_node_for_rust(self, node_name: str, arg: Any, state: Any) -> dict[str, Any]:

        if node_name not in self._nodes:
            raise ValueError(f"Unknown node `{node_name}`")
        node = self._nodes[node_name]
        result = _invoke_node(node, self.context, arg)
        if inspect.isawaitable(result):
            result = asyncio.run(cast(Coroutine[Any, Any, Any], result))

        if isinstance(result, Command):
            update = result.update
            sends = _normalize_goto(result.goto, default_arg=state)
        else:
            update = result
            sends = _normalize_result_to_sends(result, default_arg=state)

        if update is None and isinstance(arg, dict):
            # Preserve in-place state mutations for prototype nodes like wait_node.
            update = arg

        return {
            "update": update,
            "sends": [
                {"node": _resolve_target_name(send.node), "arg": send.arg}
                for send in sends
            ],
        }

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


def _condition_to_rust(condition: WaitCondition) -> dict[str, Any]:
    if isinstance(condition, ChannelCondition):
        return {"kind": "channel", "channel": condition.channel, "n": condition.n}
    if isinstance(condition, TimerCondition):
        return {"kind": "timer", "seconds": condition.seconds}
    raise TypeError(f"Unsupported condition type: {type(condition)!r}")


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


def _invoke_node(node: Callable[..., Any], ctx: Context, state: Any) -> Any:
    try:
        params = list(inspect.signature(node).parameters.values())
    except (TypeError, ValueError):
        params = []

    if len(params) >= 2:
        return node(ctx, state)
    if len(params) == 1:
        return node(state)
    return node()
