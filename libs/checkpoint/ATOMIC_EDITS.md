# Proposal: revision-checked store edits

**Prototype, not a distributed-store rollout.** This branch adds public Python
`BaseStore.compare_and_swap` / `acompare_and_swap` methods and optional
`Item.revision` metadata. Only `InMemoryStore` implements them. Production
adapters, JS, remote protocols, server support, and StoreBackend integration are
not implemented here.

## Why a runtime primitive

`get` followed by `put` is not atomic: two agents can read one document and
unconditionally replace each other's edits. A process-local mutex cannot protect
other processes writing to the same database. Existing `put` remains an
unconditional whole-value replacement; this proposal does not silently change
its semantics or make existing StoreBackend edits safe automatically.

One conditional write primitive is necessary for correctness, not an alternative
convenience spelling of `put`. It is deliberately separate from `PutOp` and
`batch`: current batching can coalesce multiple puts to one key, whereas each
conditional attempt must receive its own independent success/failure result.
There is no callback API, patch API, or get-then-put fallback.

## Contract

```python
store.compare_and_swap(
    namespace,
    key,
    value,
    expected_revision=revision,
    index=None,
)
```

The async spelling is `await store.acompare_and_swap(...)`, with the same
arguments and semantics.

- `value` is a complete replacement mapping; `None` requests deletion.
- `expected_revision` is required. `None` requires that the key be absent **at
  commit time**; a string must equal the current item's opaque revision.
  Deleting an absent key with `expected_revision=None` succeeds as a no-op.
  An absence check does not assert that the key has never previously existed.
- `True` means the condition matched and the write committed. `False` means a
  conflict; value, revision, timestamps, and index are unchanged by that attempt.
  Other errors propagate, rather than looking like contention. Embedding work
  may run even for a losing attempt; embedding failure raises instead of returning
  `False`. Network failures in future adapters may leave the outcome unknown.
- Every committed replacement, including an identical-value ordinary `put`,
  mints a fresh opaque revision. Revisions must not be reused after delete/recreate
  or A → B → A; timestamps and content hashes alone do not meet this contract.
  Treat tokens as exact strings, not ordered counters or authorization credentials.
- `Item` and `SearchItem` expose optional `revision`; it is also serialized by
  `dict()` when present. Legacy constructors and serialized items without a
  revision remain valid. `None` on an **existing** item means no usable revision,
  never permission to overwrite it as if it were absent. Snapshots and submitted
  JSON values must not alias internal storage. Editing a snapshot's content or
  metadata must not mutate the stored item or its revision.
- Base implementations raise `NotImplementedError` without reading or writing;
  adapters opt in by implementing each sync/async method. A non-`None` revision
  alone does not promise that both interfaces are supported. Unsupported adapters
  must never emulate this with separate `get` and `put` calls.
- Index selection matches `put`. A successful replacement replaces all old
  vectors; `index=False` removes them. A failed condition must publish neither
  replacement vectors nor content. Embeddings can be calculated before the
  storage transaction, but the revision is checked **after** that work, at commit.
- This prototype has no conditional TTL argument or TTL implementation. A future
  TTL-capable adapter must specify expiry/refresh semantics and treat expired
  entries as absent atomically; a background sweeper must not resurrect data.
- Atomicity is per key, not a multi-key transaction. A later unconditional put or
  delete may still overwrite a successful conditional edit. All cooperating
  editors must use conditional writes to avoid lost edits between them.

## Read, replace, retry

A StoreBackend can recompute a pure text replacement from a fresh snapshot on
conflict. Do not reuse an already-modified candidate or repeat non-idempotent
external side effects in the retry loop. This example intentionally refuses an
ambiguous/missing text match rather than overwriting a concurrent semantic edit.

```python
from langgraph.store.base import BaseStore


def replace_text(
    store: BaseStore,
    namespace: tuple[str, ...],
    key: str,
    old: str,
    new: str,
) -> None:
    for _ in range(8):
        snapshot = store.get(namespace, key)
        if snapshot is None:
            raise FileNotFoundError(key)
        if snapshot.revision is None:
            raise NotImplementedError("This store does not expose write revisions")
        text = snapshot.value["content"]
        if not old or text.count(old) != 1:
            raise ValueError("Replacement must match exactly once in the latest text")
        candidate = {**snapshot.value, "content": text.replace(old, new, 1)}
        if store.compare_and_swap(
            namespace, key, candidate, expected_revision=snapshot.revision
        ):
            return
    raise RuntimeError("Concurrent edits exhausted the retry budget")
```

For async use, await `aget` and `acompare_and_swap` at the same points. Bounded
backoff/cancellation can be applied by the caller; the store never retries a
transformation it cannot inspect. To create without overwriting, use
`expected_revision=None`. To delete a read snapshot, pass `value=None` and its
non-`None` revision. A conflict requires a fresh read and a fresh decision.

## Reference implementation and production requirements

`InMemoryStore` uses one threading lock for snapshots and all commit paths,
including ordinary puts/deletes and direct batches. Inputs/outputs are copied;
content, fresh revision, and staged vectors are published together. Embedding
functions run outside the lock, including across awaits. A suspended conditional
writer rechecks its token when it resumes. A suspended **unconditional** writer
can commit after a conditional writer; its content and vectors must agree with
that later commit. Errors/cancellation during embedding publish nothing.

That lock coordinates only callers sharing **one InMemoryStore instance in one
process**, including sync threads and async tasks. Separate instances/processes
have separate data. This is a working reference of the contract, not distributed
atomicity. Direct mutation of private `_data` / `_vectors` bypasses the API and is
outside the contract. Copying snapshots adds memory/CPU cost.

Before this can make production StoreBackend edits safe:

1. Supporting adapters need durable revision metadata and a schema/backfill
   strategy. Ordinary writes, conditional writes, deletes, TTL expiry, and any
   other writer must participate. Old deployed writers must not bypass revision
   changes; mixed-version deployments require explicit rollout coordination.
2. Implement matching/replacement in the storage engine, not in an application
   mutex: for example, SQL `UPDATE ... WHERE revision = expected RETURNING ...`,
   conditional `DELETE`, and a unique-key-guarded insert for the absent case.
   Update content/revision/vector rows in the same transaction. Failed conditions
   must not touch index rows or TTLs. Use unique write tokens (or durable generations
   that survive deletion), not a per-process counter that can reset.
3. Preserve one response per attempt, independent of ordinary batch deduplication.
   Define ambiguous network outcomes and use adapter conformance tests, including
   multiple independent connections/processes. An application lock test is not
   evidence of database correctness.
4. Land equivalent JS runtime semantics and revision serialization, including
   sync/async differences where applicable. Remote clients/servers must round-trip
   revisions and conflicts without falling back to unconditional writes. No JS
   implementation is included in this Python-only proposal branch.
5. Switch StoreBackend's actual read-modify-write path to this primitive and
   bounded re-read/recompute retries. Unsupported adapters should produce an
   explicit unsupported-operation error, never degrade to unsafe edits.
