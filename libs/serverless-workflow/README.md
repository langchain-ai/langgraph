# langgraph-serverless-workflow

Compile a CNCF [Serverless Workflow](https://github.com/serverlessworkflow/specification) DSL document into a LangGraph `StateGraph` object.

The compiler translates the DSL `states` into LangGraph nodes and the DSL transitions / `switch` conditions into LangGraph edges (including conditional edges). Function callables are supplied locally through `register_function` and are looked up by the DSL `functionRef.refName`.

## Install

```bash
pip install langgraph-serverless-workflow
```

## Quick start

```python
from langgraph_serverless_workflow import ServerlessWorkflowCompiler

dsl = """
id: greetflow
version: '1.0'
specVersion: '0.8'
start: InjectState
states:
  - name: InjectState
    type: inject
    data:
      who: world
    transition: GreetState
  - name: GreetState
    type: operation
    actions:
      - functionRef:
          refName: greet
    end: true
"""

compiler = ServerlessWorkflowCompiler()
compiler.register_function("greet", lambda **kw: {"msg": f"hi {kw['who']}"})
graph = compiler.compile(dsl)
print(graph.invoke({"data": {}}))
# {'data': {'who': 'world', 'msg': 'hi world'}}
```

## Supported state types

- `inject` - merges its `data` into the workflow state.
- `operation` - runs each registered `functionRef` action sequentially, merging results.
- `switch` (data-based) - compiles to a LangGraph conditional edge. `dataConditions` are evaluated against the state data; the matching branch (or `defaultCondition`) is followed.
- `sleep` - pauses for an ISO-8601 `duration` (e.g. `PT2S`). Override the sleeper with `register_sleeper`.
- `event` - consumes a trigger event via a handler registered with `register_event_handler`, then runs the matching `onEvents` actions.
- `callback` - invokes the callback `action`, then resolves a result event via a handler registered with `register_callback_handler`.
- `foreach` - iterates over `inputCollection`, runs `actions` per item (sequentially or in parallel), and writes the results to `outputCollection`.
- `parallel` - runs every branch concurrently with a thread pool and merges the produced deltas.

## Limitations

The compiler targets the commonly used subset of the Serverless Workflow DSL:

- `subflow` and any other unsupported state type raise a `ValueError` at compile time rather than executing silently.
- `foreach` `outputCollection` is resolved to a single top-level state key (the last path segment). Nested destination paths such as `.a.b.results` are flattened to `results`.
- `event` states resolve the trigger payload from the first registered handler matching the first `eventRefs` entry; multiple event references are not consumed.
- Function callables are invoked with the running state data as keyword arguments, so only state keys that are valid Python identifiers are forwarded. Keys such as `"my-key"` are dropped; declare explicit `arguments` in the DSL when a function needs them.

## Expression evaluation

Serverless Workflow defaults to `jq` as its expression language. When the `jq` python package is installed it is used automatically; otherwise the bundled `SimpleExpressionEvaluator` covers the common `switch`-state patterns: path access (`.a.b`, `.[0]`, `["key"]`), the `length` filter, and `==`, `!=`, `>`, `>=`, `<`, `<=` comparisons. Plug in a full evaluator with:

```python
compiler.register_evaluator(MyEvaluator())
```

An evaluator needs both an `evaluate(expression: str, data: dict) -> bool` method and a `resolve(expression: str, data: dict) -> Any` method.
## Development

```bash
make format   # run code formatters
make lint     # run the linter
make test     # execute the test suite
```

To run a particular test file: `TEST=path/to/test.py make test`.
