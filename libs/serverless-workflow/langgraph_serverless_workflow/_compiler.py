from __future__ import annotations

import json
import keyword
import operator
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any, Callable, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from serverlessworkflow.sdk.workflow import Workflow

__all__ = (
    "ExpressionEvaluator",
    "SimpleExpressionEvaluator",
    "ServerlessWorkflowCompiler",
)


@runtime_checkable
class ExpressionEvaluator(Protocol):
    """Evaluates Serverless Workflow condition expressions against state data.

    Serverless Workflow defaults to `jq` as its expression language. Plug in a
    real jq-backed evaluator (e.g. the `jq` python package) via
    `ServerlessWorkflowCompiler.register_evaluator` when you need the full
    language; the bundled `SimpleExpressionEvaluator` covers the common
    `switch`-state patterns used in most workflows.
    """

    def evaluate(self, expression: str, data: dict) -> bool: ...

    def resolve(self, expression: str, data: dict) -> Any: ...


class SimpleExpressionEvaluator:
    """A small jq-subset evaluator used when the `jq` package is unavailable.

    Supports the patterns most commonly found in Serverless Workflow specs:
    path access (`.a.b`, `.[0]`, `["key"]`), the `length` filter, the pipe
    operator (`a | b`), and the comparison operators `==`, `!=`, `>`, `>=`,
    `<`, `<=`. `evaluate` reduces the result to a boolean; `resolve` returns
    the underlying value. Anything that cannot be parsed is reduced to the
    truthiness of the resolved value.
    """

    _COMPARE_OPS = ("==", "!=", ">=", "<=", ">", "<")

    def evaluate(self, expression: str, data: dict) -> bool:
        expr = (expression or "").strip()
        for op in self._COMPARE_OPS:
            idx = expr.find(op)
            if idx > 0:
                left = expr[:idx].strip()
                right = expr[idx + len(op) :].strip()
                return self._compare(
                    self.resolve(left, data), op, self._parse_literal(right)
                )
        return bool(self.resolve(expr, data))

    def resolve(self, expression: str, data: dict) -> Any:
        expr = (expression or "").strip()
        if expr.startswith("(") and expr.endswith(")"):
            expr = expr[1:-1].strip()
        if "|" in expr:
            value: Any = data
            for part in (p.strip() for p in expr.split("|")):
                value = self._apply_filter(part, value)
            return value
        return self._apply_filter(expr, data)

    @staticmethod
    def _parse_literal(token: str) -> Any:
        token = token.strip()
        low = token.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("null", "none"):
            return None
        if (token.startswith('"') and token.endswith('"')) or (
            token.startswith("'") and token.endswith("'")
        ):
            return token[1:-1]
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return token

    def _apply_filter(self, expr: str, value: Any) -> Any:
        expr = expr.strip()
        if expr in ("", "."):
            return value
        if expr == "length":
            try:
                return len(value)
            except TypeError:
                return 0
        return self._navigate(expr, value)

    def _navigate(self, expr: str, value: Any) -> Any:
        if not expr:
            return value
        pos = 1 if expr.startswith(".") else 0
        cur: Any = value
        while pos < len(expr):
            ch = expr[pos]
            if ch == ".":
                pos += 1
                continue
            if ch == "[":
                close = expr.find("]", pos)
                if close == -1:
                    return cur
                cur = self._index(cur, expr[pos + 1 : close].strip())
                pos = close + 1
            elif ch in ("'", '"'):
                close = expr.find(ch, pos + 1)
                if close == -1:
                    return cur
                cur = self._index(cur, expr[pos + 1 : close])
                pos = close + 1
            else:
                end = pos
                while end < len(expr) and expr[end] not in ".[":
                    end += 1
                cur = self._index(cur, expr[pos:end])
                pos = end
        return cur

    @staticmethod
    def _index(cur: Any, key: str) -> Any:
        if cur is None:
            return None
        try:
            idx = int(key)
            if isinstance(cur, (list, tuple, str)):
                return cur[idx] if -len(cur) <= idx < len(cur) else None
        except ValueError:
            pass
        if isinstance(cur, dict):
            return cur.get(key)
        return getattr(cur, key, None)

    @staticmethod
    def _compare(left: Any, op: str, right: Any) -> bool:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        try:
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
        except TypeError:
            return False
        return False


def _default_evaluator() -> ExpressionEvaluator:
    try:
        import jq  # type: ignore[import-untyped]
    except ImportError:
        return SimpleExpressionEvaluator()

    class JqEvaluator:
        def evaluate(self, expression: str, data: dict) -> bool:
            return bool(jq.first(expression, data))

        def resolve(self, expression: str, data: dict) -> Any:
            return jq.first(expression, data)

    return JqEvaluator()


class _GraphState(TypedDict):
    # Each node returns a dict that is incrementally merged into the shared
    # `data` channel via dict union, mirroring Serverless Workflow's state data
    # model.
    data: Annotated[dict, operator.or_]


_DURATION_RE = re.compile(
    r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?"
    r"(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
)


def _parse_duration(value: Any) -> float:
    """Parse an ISO-8601 duration (e.g. `PT5S`, `PT1M30S`) into seconds.

    A bare number is treated as seconds. Unparseable values yield 0.
    """
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        pass
    match = _DURATION_RE.match(text)
    if not match:
        return 0.0
    years, months, weeks, days, hours, mins, secs = match.groups()
    total = 0.0
    if years:
        total += int(years) * 365 * 24 * 3600
    if months:
        total += int(months) * 30 * 24 * 3600
    if weeks:
        total += int(weeks) * 7 * 24 * 3600
    if days:
        total += int(days) * 24 * 3600
    if hours:
        total += int(hours) * 3600
    if mins:
        total += int(mins) * 60
    if secs:
        total += float(secs)
    return total


def _collection_key(path: str | None) -> str:
    """Turn a jq-ish collection path (`.results`, `results`) into a state key."""
    if not path:
        return "results"
    key = path.strip().lstrip(".")
    if "." in key:
        key = key.split(".")[-1]
    return key or "results"


def _transition_target(transition: Any) -> str | None:
    """Resolve a `transition` field (str | Transition | None) to a state name."""
    if transition is None:
        return None
    if isinstance(transition, str):
        return transition
    return getattr(transition, "nextState", None)


def _is_end(end: Any) -> bool:
    """Resolve an `end` field (bool | End | None) to a terminate decision."""
    if end is None:
        return False
    if isinstance(end, bool):
        return end
    # An explicit `End` object means the workflow terminates here.
    return True


def _coerce_data(data: Any) -> dict:
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return {"value": data}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": data}


def _function_ref(func_ref: Any) -> tuple[str, dict]:
    """Normalize a `functionRef` (str | dict | FunctionRef) into (refName, arguments)."""
    if isinstance(func_ref, str):
        return func_ref, {}
    if isinstance(func_ref, dict):
        return func_ref.get("refName") or "", func_ref.get("arguments") or {}
    return (
        getattr(func_ref, "refName", None) or "",
        getattr(func_ref, "arguments", None) or {},
    )


def _action_field(action: Any, name: str, default: Any = None) -> Any:
    """Read a field from an action that may be a hydrated SDK object or a raw dict.

    The `serverlessworkflow` SDK hydrates operation/foreach/parallel/branch
    actions into `Action` objects, but leaves callback `action` and event
    `onEvents[].actions` as plain dicts. This helper papers over the
    difference so the compiler can treat both uniformly.
    """
    if isinstance(action, dict):
        return action.get(name, default)
    return getattr(action, name, default)


def _callable_kwargs(data: dict) -> dict:
    """Forward only state keys that are valid, non-keyword Python identifiers.

    Workflow state data is arbitrary JSON and may contain keys like
    ``"my-key"`` or ``"123"`` that cannot be unpacked as keyword arguments;
    those are dropped rather than crashing ``func(**kwargs)``. Functions that
    need such keys should declare explicit ``arguments`` in the DSL.
    """
    return {
        k: v
        for k, v in data.items()
        if isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k)
    }


class ServerlessWorkflowCompiler:
    """Compile a CNCF Serverless Workflow DSL document into a LangGraph `StateGraph`.

    The compiler translates every DSL `states` entry into a LangGraph node and
    the DSL transitions / `switch` conditions into LangGraph edges (including
    conditional edges). Function callables are supplied locally through
    `register_function` and are looked up by the DSL `functionRef.refName`.
    External event consumption (for `event` and `callback` states) and the
    sleep behavior are pluggable via `register_event_handler`,
    `register_callback_handler` and `register_sleeper`.

    Example:
        ```python
        compiler = ServerlessWorkflowCompiler()
        compiler.register_function("greet", lambda **kw: {"msg": f"hi {kw['who']}"})
        graph = compiler.compile(dsl_yaml_string)
        graph.invoke({"data": {"who": "world"}})
        ```
    """

    def __init__(
        self,
        expression_evaluator: ExpressionEvaluator | None = None,
        max_parallel_workers: int | None = None,
    ) -> None:
        self.function_registry: dict[str, Callable[..., Any]] = {}
        self.event_handlers: dict[str, Callable[..., Any]] = {}
        self.callback_handlers: dict[str, Callable[..., Any]] = {}
        self.expression_evaluator: ExpressionEvaluator = (
            expression_evaluator or _default_evaluator()
        )
        self._fallback_resolver = SimpleExpressionEvaluator()
        self.sleeper: Callable[[float], None] = time.sleep
        self.max_parallel_workers = max_parallel_workers or (os.cpu_count() or 4)

    # -- registration ------------------------------------------------------

    def register_function(self, name: str, func: Callable[..., Any]) -> "ServerlessWorkflowCompiler":
        """Register a callable for a DSL `functionRef.refName`."""
        if not callable(func):
            raise TypeError(f"function for '{name}' must be callable")
        self.function_registry[name] = func
        return self

    def register_evaluator(self, evaluator: ExpressionEvaluator) -> "ServerlessWorkflowCompiler":
        """Override the expression evaluator used for conditions and value resolution."""
        self.expression_evaluator = evaluator
        return self

    def register_event_handler(
        self, event_ref: str, handler: Callable[..., Any]
    ) -> "ServerlessWorkflowCompiler":
        """Register a callable that resolves the payload for an `event` state trigger."""
        if not callable(handler):
            raise TypeError(f"event handler for '{event_ref}' must be callable")
        self.event_handlers[event_ref] = handler
        return self

    def register_callback_handler(
        self, result_event_ref: str, handler: Callable[..., Any]
    ) -> "ServerlessWorkflowCompiler":
        """Register a callable that resolves the result payload for a `callback` state."""
        if not callable(handler):
            raise TypeError(f"callback handler for '{result_event_ref}' must be callable")
        self.callback_handlers[result_event_ref] = handler
        return self

    def register_sleeper(self, sleeper: Callable[[float], None]) -> "ServerlessWorkflowCompiler":
        """Override the sleep function (seconds) used by `sleep` states and action sleeps."""
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")
        self.sleeper = sleeper
        return self

    # -- compilation -------------------------------------------------------

    def compile(self, dsl_str: str) -> CompiledStateGraph:
        """Parse `dsl_str` (YAML/JSON) and return a compiled LangGraph graph."""
        workflow = Workflow.from_source(dsl_str)
        states = workflow.states or []
        states_map: dict[str, Any] = {}
        for state in states:
            name = getattr(state, "name", None)
            if not name:
                raise ValueError("Every workflow state must declare a name")
            states_map[name] = state

        builder = StateGraph(_GraphState)

        # 1. Register one LangGraph node per DSL state.
        for name, state in states_map.items():
            builder.add_node(name, self._build_node_function(state))

        # 2. Translate transitions / switches into edges.
        for name, state in states_map.items():
            self._add_edges(builder, name, state)

        # 3. Wire the entry point.
        builder.add_edge(START, self._resolve_start(workflow, states_map))

        return builder.compile()

    # -- entry point -------------------------------------------------------

    def _resolve_start(self, workflow: Workflow, states_map: dict[str, Any]) -> str:
        start = workflow.start
        if isinstance(start, str) and start:
            return start
        if start is not None:
            name = getattr(start, "stateName", None) or getattr(start, "name", None)
            if name:
                return name
        if states_map:
            # Serverless Workflow allows omitting `start`; the first state is
            # the implicit entry point.
            return next(iter(states_map))
        raise ValueError("Workflow declares no states and no start state")

    # -- nodes -------------------------------------------------------------

    def _build_node_function(self, state: Any) -> Callable[[dict], dict]:
        stype = getattr(state, "type", None)
        if stype == "inject":
            return self._inject_node(state)
        if stype == "operation":
            return self._operation_node(state)
        if stype == "sleep":
            return self._sleep_node(state)
        if stype == "event":
            return self._event_node(state)
        if stype == "callback":
            return self._callback_node(state)
        if stype == "foreach":
            return self._foreach_node(state)
        if stype == "parallel":
            return self._parallel_node(state)
        if stype == "switch":
            # switch states don't transform data; routing happens via the
            # conditional edge built in `_add_edges`.
            return self._passthrough_node()
        name = getattr(state, "name", "<unnamed>")
        raise ValueError(
            f"Unsupported state type {stype!r} for state {name!r}. "
            f"Supported types: inject, operation, switch, sleep, event, "
            f"callback, foreach, parallel."
        )

    def _passthrough_node(self) -> Callable[[dict], dict]:
        # States that only route (e.g. `switch`) pass the data through
        # unchanged; their behavior lives in the edges, not the node body.
        def node(state_data: dict) -> dict:
            return {"data": dict(state_data.get("data", {}))}

        return node

    def _inject_node(self, state: Any) -> Callable[[dict], dict]:
        injected = _coerce_data(getattr(state, "data", None))

        def node(state_data: dict) -> dict:
            # Returning only the injected delta lets the graph reducer merge it
            # into the running state data.
            return {"data": injected}

        return node

    def _operation_node(self, state: Any) -> Callable[[dict], dict]:
        actions = getattr(state, "actions", []) or []

        def node(state_data: dict) -> dict:
            current, _ = self._run_actions(actions, state_data.get("data", {}))
            return {"data": current}

        return node

    def _sleep_node(self, state: Any) -> Callable[[dict], dict]:
        duration = getattr(state, "duration", None)
        sleeper = self.sleeper

        def node(state_data: dict) -> dict:
            if duration:
                secs = _parse_duration(duration)
                if secs > 0:
                    sleeper(secs)
            return {"data": dict(state_data.get("data", {}))}

        return node

    def _event_node(self, state: Any) -> Callable[[dict], dict]:
        on_events_list = getattr(state, "onEvents", []) or []
        handlers = self.event_handlers

        def node(state_data: dict) -> dict:
            current = dict(state_data.get("data", {}))
            for on_events in on_events_list:
                event_refs = getattr(on_events, "eventRefs", []) or []
                # Resolve the event payload from the first registered handler.
                for ref in event_refs:
                    handler = handlers.get(ref)
                    if handler is not None:
                        payload = handler(current)
                        if isinstance(payload, dict):
                            current.update(payload)
                        break
                actions = getattr(on_events, "actions", []) or []
                current, _ = self._run_actions(actions, current)
            return {"data": current}

        return node

    def _callback_node(self, state: Any) -> Callable[[dict], dict]:
        action = getattr(state, "action", None)
        event_ref = getattr(state, "eventRef", None)
        handlers = self.callback_handlers

        def node(state_data: dict) -> dict:
            current = dict(state_data.get("data", {}))
            if action is not None:
                output = self._invoke_action(action, current)
                if isinstance(output, dict):
                    current.update(output)
            if event_ref:
                handler = handlers.get(event_ref)
                if handler is not None:
                    payload = handler(current)
                    if isinstance(payload, dict):
                        current.update(payload)
            return {"data": current}

        return node

    def _foreach_node(self, state: Any) -> Callable[[dict], dict]:
        input_collection = getattr(state, "inputCollection", None)
        output_collection = getattr(state, "outputCollection", None)
        iteration_param = getattr(state, "iterationParam", None)
        actions = getattr(state, "actions", []) or []
        mode = getattr(state, "mode", "parallel") or "parallel"
        pool = self.max_parallel_workers

        def node(state_data: dict) -> dict:
            current = dict(state_data.get("data", {}))
            items = self._resolve_value(input_collection, current) if input_collection else []
            if not isinstance(items, (list, tuple)):
                items = []

            def run_one(item: Any) -> dict:
                ctx = dict(current)
                if iteration_param:
                    ctx[iteration_param] = item
                _, delta = self._run_actions(actions, ctx)
                return delta

            if mode == "sequential" or len(items) <= 1:
                results = [run_one(item) for item in items]
            else:
                with ThreadPoolExecutor(max_workers=pool) as ex:
                    results = list(ex.map(run_one, items))

            key = _collection_key(output_collection) if output_collection else (
                iteration_param or "results"
            )
            current[key] = results
            return {"data": current}

        return node

    def _parallel_node(self, state: Any) -> Callable[[dict], dict]:
        branches = getattr(state, "branches", []) or []
        completion_type = getattr(state, "completionType", "allOf") or "allOf"
        num_completed = getattr(state, "numCompleted", None)
        pool = self.max_parallel_workers

        def node(state_data: dict) -> dict:
            current = dict(state_data.get("data", {}))
            if not branches:
                return {"data": current}

            def run_branch(branch: Any) -> dict:
                actions = getattr(branch, "actions", []) or []
                _, delta = self._run_actions(actions, current)
                return delta

            # Don't use a `with` block here: its `__exit__` calls
            # shutdown(wait=True), which would block until every branch
            # finishes and defeat the spec's atLeast/numCompleted semantics.
            ex = ThreadPoolExecutor(max_workers=pool)
            try:
                futures = [ex.submit(run_branch, branch) for branch in branches]
                if completion_type == "atLeast" and num_completed:
                    wanted = max(1, min(int(num_completed), len(futures)))
                    deltas: list[dict] = []
                    for fut in as_completed(futures):
                        deltas.append(fut.result())
                        if len(deltas) >= wanted:
                            break
                    # Abandon the remaining branches instead of waiting for them.
                    for fut in futures:
                        fut.cancel()
                else:
                    deltas = [future.result() for future in futures]
            finally:
                ex.shutdown(wait=False)

            for delta in deltas:
                if isinstance(delta, dict):
                    current.update(delta)
            return {"data": current}

        return node

    # -- action execution -------------------------------------------------

    def _run_actions(self, actions: Any, base: dict) -> tuple[dict, dict]:
        """Run a sequence of actions, returning (merged_state, produced_delta)."""
        current = dict(base)
        delta: dict[str, Any] = {}
        for action in actions or []:
            self._action_sleep(action, before=True)
            condition = _action_field(action, "condition", None)
            if condition and not self.expression_evaluator.evaluate(condition, current):
                self._action_sleep(action, after=True)
                continue
            output = self._invoke_action(action, current)
            if isinstance(output, dict):
                current.update(output)
                delta.update(output)
            self._action_sleep(action, after=True)
        return current, delta

    def _action_sleep(self, action: Any, *, before: bool = False, after: bool = False) -> None:
        if not (before or after):
            return
        sleep = _action_field(action, "sleep", None)
        if sleep is None:
            return
        if isinstance(sleep, dict):
            duration = sleep.get("before" if before else "after", None)
        else:
            duration = getattr(sleep, "before" if before else "after", None)
        if duration:
            self.sleeper(_parse_duration(duration))

    def _invoke_action(self, action: Any, current: dict) -> Any:
        func_ref = _action_field(action, "functionRef", None)
        if func_ref is None:
            return None
        ref_name, arguments = _function_ref(func_ref)
        func = self.function_registry.get(ref_name)
        if func is None:
            raise KeyError(
                f"No function registered for functionRef '{ref_name}'. "
                f"Call register_function('{ref_name}', ...) before compiling."
            )
        # Explicit DSL arguments are forwarded as-is; otherwise the running
        # state data is forwarded as keyword arguments so simple functions can
        # read straight from the workflow state. Only identifier-safe keys are
        # forwarded so non-identifier JSON keys (e.g. "my-key") don't crash
        # func(**kwargs); see `_callable_kwargs`.
        kwargs = dict(arguments) if arguments else _callable_kwargs(current)
        return func(**kwargs)

    def _resolve_value(self, expression: str | None, data: dict) -> Any:
        """Resolve a jq-ish expression to a value using the active evaluator."""
        if not expression:
            return None
        resolve = getattr(self.expression_evaluator, "resolve", None)
        if callable(resolve):
            try:
                return resolve(expression, data)
            except Exception:
                pass
        return self._fallback_resolver.resolve(expression, data)

    # -- edges -------------------------------------------------------------

    def _add_edges(self, builder: StateGraph, name: str, state: Any) -> None:
        stype = getattr(state, "type", None)
        if stype == "switch":
            routing_fn, path_map = self._build_conditional_edge(state)
            builder.add_conditional_edges(name, routing_fn, path_map)
            return

        if _is_end(getattr(state, "end", None)):
            builder.add_edge(name, END)
            return

        nxt = _transition_target(getattr(state, "transition", None))
        if nxt:
            builder.add_edge(name, nxt)
        # A state with neither `end` nor `transition` is malformed in the DSL;
        # leave it unconnected and let LangGraph surface the error on compile.

    def _build_conditional_edge(
        self, state: Any
    ) -> tuple[Callable[[dict], str], dict[str, str]]:
        """Compile a `switch` state into a LangGraph conditional edge."""
        conditions = getattr(state, "dataConditions", []) or []
        path_map: dict[str, str] = {}
        ordered: list[tuple[str, str | None]] = []

        for idx, cond in enumerate(conditions):
            key = f"cond_{idx}"
            expr = getattr(cond, "condition", None)
            target = _transition_target(getattr(cond, "transition", None))
            if target:
                path_map[key] = target
            elif _is_end(getattr(cond, "end", None)):
                path_map[key] = END
            else:
                path_map[key] = END
            ordered.append((key, expr))

        default_key = "default"
        default_cond = getattr(state, "defaultCondition", None)
        default_target = _transition_target(getattr(default_cond, "transition", None))
        if default_target:
            path_map[default_key] = default_target
        else:
            path_map[default_key] = END

        evaluator = self.expression_evaluator

        def route_fn(state_data: dict) -> str:
            data = state_data.get("data", {})
            for key, expr in ordered:
                if not expr:
                    continue
                try:
                    if evaluator.evaluate(expr, data):
                        return key
                except Exception:
                    continue
            return default_key

        return route_fn, path_map
