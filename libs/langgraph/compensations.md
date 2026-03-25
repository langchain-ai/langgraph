# Compensations Proposal

## Summary

Make compensations a task-level feature, not a tool-level feature.

```python
@task(compensation=refund_charge)
def charge(order: Order) -> ChargeReceipt:
    ...

graph.invoke(Command(compensate=True), config)
# or: graph.compensate(config)
```

## Why Task-Level

- `@task` is already the durable execution unit with stable input/output, retry, cache, and checkpoint semantics.
- `StateGraph` nodes are broader state transformers, so node-level compensation is less precise.
- `ToolNode` executes tool calls inside a node today, so tool-level compensation would require separate persistence semantics.

## Runtime Shape

- `task()` carries an optional `CompensationSpec`.
- `Call` and `PregelExecutableTask` carry that spec through execution.
- On successful task commit, the runner appends a hidden compensation registration write.
- `Command(compensate=True)` triggers an internal unwinder.
- The unwinder executes compensation handlers in LIFO order as hidden tasks, using normal retry, interrupt, and checkpoint behavior.

## Persistence

- Persist compensation state in hidden checkpointed channels, not user state and not checkpoint metadata.
- No new saver API or database schema should be required.
- Older checkpoints should treat missing compensation state as an empty stack.
- Persist handler identifiers and serializable context, not raw callables.

## Scope

- Start with task-level support only.
- Node-level support can be sugar over the same mechanism later.
- Tool support should delegate to the same core mechanism rather than define separate behavior.
- Do not auto-compensate on every error by default; make unwind explicit.
