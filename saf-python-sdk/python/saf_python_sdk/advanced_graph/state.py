from __future__ import annotations

import atexit
import asyncio
import inspect
import os
import threading
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Generic, TypeVar, cast

from saf_python_sdk.langgraph_rust_core import PyRustEngine  # type: ignore[import-untyped]

from saf_python_sdk.types import Command, Send

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class _ChannelSpec:
    typ: Any


@dataclass(frozen=True)
class ChannelCondition:
    channel: str
    min: int = 1
    max: int = 0


@dataclass(frozen=True)
class TimerCondition:
    seconds: float


@dataclass(frozen=True)
class AnyOfCondition:
    conditions: tuple[WaitCondition, ...]


WaitCondition = ChannelCondition | TimerCondition

_EXECUTOR_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None


def _advanced_graph_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            worker_count = int(os.getenv("LANGGRAPH_ADVANCED_GRAPH_PY_THREADS", "256"))
            worker_count = max(worker_count, 1)
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="saf-advanced-py",
            )
            atexit.register(_shutdown_advanced_graph_executor)
        return _EXECUTOR


def _shutdown_advanced_graph_executor() -> None:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=False, cancel_futures=False)
            _EXECUTOR = None


class WaitRequested(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("wait requested")
        self.payload = payload


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

    def add_entry_node(self, node: Callable[..., Any]) -> str:
        node_name = self.add_node(node)
        self._entry_point = self._resolve_node_name(node_name)
        return node_name

    def add_finish_node(self, node: Callable[..., Any]) -> str:
        node_name = self.add_node(node)
        self._finish_point = self._resolve_node_name(node_name)
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
        if self._entry_point not in self._nodes:
            raise ValueError(f"Entry point node `{self._entry_point}` does not exist")
        if self._finish_point is not None and self._finish_point not in self._nodes:
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
        finish_point: str | None,
    ) -> None:
        self._nodes = nodes
        self._async_channels = async_channels
        self._entry_point = entry_point
        self._finish_point = finish_point

    async def ainvoke(self, initial_state: StateT) -> StateT:
        handler = await self.astart(initial_state)
        return await handler

    async def astart(
        self, initial_state: StateT, *, stream_mode: str | None = None
    ) -> GraphRunHandler[StateT]:
        run = _GraphEngineRun(
            nodes=self._nodes,
            async_channel_specs=self._async_channels,
            entry_point=self._entry_point,
            finish_point=self._finish_point,
            stream_mode=stream_mode,
        )
        task = asyncio.create_task(run.run(initial_state))
        return GraphRunHandler(run=run, task=task)


class Context:
    """Per-run context injected into advanced graph nodes."""

    def __init__(self, run: _GraphEngineRun) -> None:
        self._run = run

    async def wait_for(self, target: WaitCondition | AnyOfCondition) -> Any:
        resumed = self._run._consume_resume_event(target)
        if resumed is not None:
            return resumed
        raise WaitRequested(_target_to_suspend_payload(target))

    def publish_to_channel(self, channel: str, value: Any) -> None:
        self._run.publish_nowait(channel, value)

    async def apublish_to_channel(self, channel: str, value: Any) -> None:
        await self._run.publish(channel, value)

    def send_custom_stream_event(self, value: Any) -> None:
        self._run.send_custom_stream_event(value)


class GraphRunHandler(Generic[StateT]):
    """Handle for an active in-memory run."""

    def __init__(self, *, run: _GraphEngineRun, task: asyncio.Task[StateT]) -> None:
        self._run = run
        self._task = task

    async def apublish_to_channel(self, channel: str, value: Any) -> None:
        if self._task.done():
            raise RuntimeError("Run has already completed")
        await self._run.publish(channel, value)

    async def receive_stream(self) -> Any | None:
        # Use a separate thread pool from graph execution to avoid deadlock
        # when LANGGRAPH_ADVANCED_GRAPH_PY_THREADS is configured to 1.
        return await asyncio.to_thread(self._run.receive_stream_sync)

    def close_stream(self) -> None:
        self._run.close_stream_sync()

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
        finish_point: str | None,
        stream_mode: str | None,
    ) -> None:
        self._nodes = nodes
        self._entry_point = entry_point
        self._finish_point = finish_point
        self._stream_mode = stream_mode
        self._rust_engine = PyRustEngine()
        for name in async_channel_specs:
            self._rust_engine.add_async_channel(name)
        self._stream_ready = threading.Event()
        if self._stream_mode is None:
            self._stream_ready.set()
        self._tasks: set[asyncio.Task[list[Send]]] = set()
        self._finished = False
        self._state: Any = None
        self._local = threading.local()
        self.context = Context(self)

    async def run(self, initial_state: StateT) -> StateT:
        finish_point = self._finish_point or ""
        if self._stream_mode is not None:
            try:
                self._rust_engine.start_stream(self._stream_mode)
            finally:
                self._stream_ready.set()
        loop = asyncio.get_running_loop()
        try:
            result_obj = await loop.run_in_executor(
                _advanced_graph_executor(),
                self._rust_engine.run_graph_py,
                self._entry_point,
                finish_point,
                initial_state,
                self._execute_node_for_rust,
                None,
            )
        finally:
            self._stream_ready.set()
        self._state = result_obj
        return cast(StateT, self._state)

    async def publish(self, channel: str, value: Any) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _advanced_graph_executor(),
            self._publish_sync,
            channel,
            value,
        )

    def publish_nowait(self, channel: str, value: Any) -> None:
        self._publish_sync(channel, value)

    async def wait_for(self, target: WaitCondition | AnyOfCondition) -> Any:
        if isinstance(target, ChannelCondition):
            value = await self._wait_for_channel_values(
                target.channel, min=target.min, max=target.max
            )
            return {
                "condition": "channel",
                "channel": target.channel,
                "value": value,
            }
        if isinstance(target, TimerCondition):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _advanced_graph_executor(),
                self._rust_engine.wait_timer,
                target.seconds,
            )
        if isinstance(target, AnyOfCondition):
            return await self._wait_for_any_of(target)
        raise ValueError(f"Unsupported wait condition type: {type(target)!r}")

    async def _wait_for_channel_values(
        self, channel: str, min: int, max: int = 0
    ) -> Any:
        if min < 1:
            raise ValueError("wait_for count `min` must be >= 1")
        if max < 0:
            raise ValueError("wait_for max count `max` must be >= 0")
        if max != 0 and max < min:
            raise ValueError("wait_for max count `max` must be 0 or >= min")
        loop = asyncio.get_running_loop()
        event = await loop.run_in_executor(
            _advanced_graph_executor(),
            self._rust_engine.wait_channel,
            channel,
            min,
            max,
        )
        return event["value"]

    async def _wait_for_any_of(self, condition: AnyOfCondition) -> Any:
        if not condition.conditions:
            raise ValueError("any_of() requires at least one condition")
        payload = {
            "conditions": [_condition_to_rust(cond) for cond in condition.conditions]
        }
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _advanced_graph_executor(),
            self._rust_engine.wait_any_of_obj,
            payload,
        )

    def _publish_sync(self, channel: str, value: Any) -> None:
        self._rust_engine.publish_obj(channel, value)

    def send_custom_stream_event(self, value: Any) -> None:
        self._rust_engine.send_custom_stream_event_obj(value)

    def receive_stream_sync(self) -> Any | None:
        self._stream_ready.wait()
        return self._rust_engine.receive_stream_obj()

    def close_stream_sync(self) -> None:
        self._rust_engine.close_stream()

    def _execute_node_for_rust(
        self, node_name: str, node_input: Any, state: Any
    ) -> dict[str, Any]:
        node_input, resume_event = _unwrap_resume_input(node_input)
        self._set_resume_event(resume_event)
        if node_name not in self._nodes:
            raise ValueError(f"Unknown node `{node_name}`")
        node = self._nodes[node_name]
        try:
            result = _invoke_node(node, self.context, node_input, state)
            if inspect.isawaitable(result):
                result = self._run_awaitable_in_worker(
                    cast(Coroutine[Any, Any, Any], result)
                )
        except WaitRequested as suspend:
            return {"suspend": suspend.payload}
        finally:
            self._set_resume_event(None)

        if isinstance(result, Command):
            update = result.update
            sends = _normalize_goto(result.goto, default_input=node_input)
        else:
            update = result
            sends = _normalize_result_to_sends(result, default_input=node_input)

        return {
            "update": update,
            "sends": [
                {"node": _resolve_target_name(send.node), "arg": send.arg}
                for send in sends
            ],
        }

    def _set_resume_event(self, event: dict[str, Any] | None) -> None:
        self._local.resume_event = event

    def _consume_resume_event(self, target: WaitCondition | AnyOfCondition) -> Any | None:
        event = cast(dict[str, Any] | None, getattr(self._local, "resume_event", None))
        if event is None:
            return None
        self._local.resume_event = None
        return event

    def _run_awaitable_in_worker(self, awaitable: Coroutine[Any, Any, Any]) -> Any:
        # Create and close a dedicated loop per execution to avoid
        # interpreter-shutdown warnings from lingering thread-local loops.
        return asyncio.run(awaitable)


def _normalize_result_to_sends(result: Any, *, default_input: Any) -> list[Send]:
    if result is None:
        return []
    if isinstance(result, Send):
        return [result]
    if callable(result):
        return [Send(_infer_node_name(result), default_input)]
    if isinstance(result, str):
        return [Send(result, default_input)]
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        sends: list[Send] = []
        for item in result:
            if isinstance(item, Send):
                sends.append(item)
            elif callable(item):
                sends.append(Send(_infer_node_name(item), default_input))
            elif isinstance(item, str):
                sends.append(Send(item, default_input))
        return sends
    return []


def _normalize_goto(goto: Any, *, default_input: Any) -> list[Send]:
    if not goto:
        return []
    if isinstance(goto, Send):
        return [goto]
    if callable(goto):
        return [Send(_infer_node_name(goto), default_input)]
    if isinstance(goto, str):
        return [Send(goto, default_input)]
    if isinstance(goto, Sequence):
        sends: list[Send] = []
        for item in goto:
            if isinstance(item, Send):
                sends.append(item)
            elif callable(item):
                sends.append(Send(_infer_node_name(item), default_input))
            elif isinstance(item, str):
                sends.append(Send(item, default_input))
        return sends
    return []


def channel_condition(channel: str, min: int = 1, max: int = 0) -> ChannelCondition:
    if min < 1:
        raise ValueError("channel_condition `min` must be >= 1")
    if max < 0:
        raise ValueError("channel_condition `max` must be >= 0")
    if max != 0 and max < min:
        raise ValueError("channel_condition `max` must be 0 or >= min")
    return ChannelCondition(channel=channel, min=min, max=max)


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
        return {
            "kind": "channel",
            "channel": condition.channel,
            "min": condition.min,
            "max": condition.max,
        }
    if isinstance(condition, TimerCondition):
        return {"kind": "timer", "seconds": condition.seconds}
    raise TypeError(f"Unsupported condition type: {type(condition)!r}")


def _target_to_suspend_payload(target: WaitCondition | AnyOfCondition) -> dict[str, Any]:
    if isinstance(target, AnyOfCondition):
        return {
            "kind": "any_of",
            "any_of": {
                "conditions": [_condition_to_rust(cond) for cond in target.conditions]
            },
        }
    return {"kind": "condition", "condition": _condition_to_rust(target)}


def _unwrap_resume_input(node_input: Any) -> tuple[Any, dict[str, Any] | None]:
    if not isinstance(node_input, dict):
        return node_input, None
    if "__lg_resume_arg__" not in node_input or "__lg_resume_event__" not in node_input:
        return node_input, None
    resume_arg = node_input["__lg_resume_arg__"]
    resume_event = node_input["__lg_resume_event__"]
    if isinstance(resume_event, dict):
        return resume_arg, resume_event
    return resume_arg, None


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


def _invoke_node(node: Callable[..., Any], ctx: Context, node_input: Any, state: Any) -> Any:
    try:
        params = list(inspect.signature(node).parameters.values())
    except (TypeError, ValueError):
        params = []

    if not params:
        return node()

    names = [param.name.lower() for param in params]
    has_ctx = [("ctx" in name or "context" in name) for name in names]
    has_state = [("state" in name) for name in names]
    has_input = [("input" in name) for name in names]

    kwargs: dict[str, Any] = {}
    unresolved = False
    for idx, param in enumerate(params):
        if has_ctx[idx]:
            kwargs[param.name] = ctx
        elif has_state[idx]:
            kwargs[param.name] = state
        elif has_input[idx]:
            kwargs[param.name] = node_input
        else:
            unresolved = True

    if kwargs and not unresolved:
        return node(**kwargs)

    if len(params) == 1:
        if has_ctx[0]:
            return node(ctx)
        if has_state[0]:
            return node(state)
        return node(node_input)

    if len(params) == 2:
        if has_ctx[0] and has_state[1]:
            return node(ctx, state)
        if has_ctx[0] and has_input[1]:
            return node(ctx, node_input)
        if has_input[0] and has_state[1]:
            return node(node_input, state)
        if has_state[0] and has_input[1]:
            return node(state, node_input)
        if has_state[0]:
            return node(state, node_input)
        if has_state[1]:
            return node(node_input, state)
        if has_ctx[0]:
            return node(ctx, node_input)
        return node(node_input, state)

    return node(ctx, node_input, state)

