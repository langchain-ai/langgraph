# RemoteGraph v3 streaming support — design

**Date:** 2026-05-21
**Status:** Approved (pending user review of this file)
**Owner:** Nick Hollon
**Scope:** `libs/langgraph/langgraph/pregel/remote.py` and a new private module `libs/langgraph/langgraph/pregel/_remote_run_stream.py`.

## Goal

Add `stream_events(version="v3")` and `astream_events(version="v3")` support to `RemoteGraph` so callers can use the same v3 streaming surface they get from local `CompiledStateGraph`. Without this, code polymorphic over `Graph | RemoteGraph` cannot adopt v3 streaming for the remote case — `RemoteGraph.astream_events` currently raises `NotImplementedError` for every version, and `RemoteGraph.stream_events` is the inherited `Runnable.stream_events` which does not accept `version="v3"`.

## Approach

Thin adapter module wrapping `langgraph_sdk._async.stream.AsyncThreadStream` and `langgraph_sdk._sync.stream.SyncThreadStream`. Two new classes — `_RemoteGraphRunStream` (sync) and `_AsyncRemoteGraphRunStream` (async) — duck-type-conform to the local `GraphRunStream` / `AsyncGraphRunStream` public surface but delegate every operation to the SDK thread stream.

Rejected alternatives:

- **Subclass `GraphRunStream` / `AsyncGraphRunStream`** and re-bind their internal `StreamMux` to SDK events. Would give true monotonic-stamp ordering parity, but couples `RemoteGraph` to internals of the local v3 stream machinery (which is still marked experimental), and contradicts the "transformers run server-side" model — local mux assumes client-side transformers.
- **Return the SDK thread stream directly.** Smallest code, but breaks the `PregelProtocol` contract: local `stream_events(version="v3")` returns a `GraphRunStream`, so a `RemoteGraph` returning something differently shaped breaks polymorphism.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Full sync + async parity (with one exception, see #6) | Match local v3 surface so callers don't have to special-case `RemoteGraph`. |
| 2 | `interleave()` cross-channel ordering is **best-effort by client receive order**, documented | Adding monotonic server-side stamps to v3 SDK events is out of scope for this PR. |
| 3 | `control=`, `transformers=`, `interrupt_before=`, `interrupt_after=` all raise `NotImplementedError` at dispatch | Server / v3 SDK does not yet plumb these kwargs through. Hard-reject surfaces the limitation; easy to lift later. |
| 4 | PR scope: v3 only. v1/v2 `astream_events` keeps raising `NotImplementedError` | Smallest focused PR. v1/v2 gap is a separate ticket. |
| 5 | Branch off main; not runnable until v3 SDK surface lands on main | Paper exercise / parallel-track design; rebase before merge. |
| 6 | Sync `interleave()` raises `NotImplementedError` (points caller at `astream_events`) | Real sync interleave requires drainer threads with ~30-40 LOC of cleanup edge cases. Sync RemoteGraph callers almost always do simple iteration; easy to upgrade later. |
| 7 | Unknown `**kwargs` on the v3 dispatch raise `NotImplementedError` listing supported kwargs | Catches mistakes loudly; matches the explicit-reject policy of #3. |

## Module layout

```
libs/langgraph/langgraph/pregel/
├── remote.py                       # add stream_events override + replace astream_events v3 branch
└── _remote_run_stream.py           # NEW — private adapter classes
```

No changes to `libs/langgraph/langgraph/stream/run_stream.py` or to `libs/langgraph/langgraph/pregel/main.py`. The adapters are duck-typed; nothing inherits from local v3 classes.

Imports at module top of `_remote_run_stream.py`. If the v3 SDK surface (`langgraph_sdk._async.stream`, `langgraph_sdk._sync.stream`) isn't installed on the running `langgraph_sdk` version, the module fails to import and `RemoteGraph.stream_events(version="v3")` raises the same `ImportError`. No feature detection or soft-fail.

Dependency direction is unchanged from today: `langgraph.pregel` → `langgraph_sdk`.

## Adapter class surface

Both classes take the same constructor kwargs; differences are entry/exit and iteration protocol.

```python
class _RemoteGraphRunStream:
    """Sync adapter conforming to GraphRunStream's public surface."""

    def __init__(
        self,
        *,
        sync_client: SyncLangGraphClient,
        sdk_thread: SyncThreadStream,       # unentered
        input: Any,
        config: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None: ...

    def __enter__(self) -> "_RemoteGraphRunStream": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...

    @property
    def output(self) -> dict[str, Any] | None: ...       # blocks until terminal
    @property
    def interrupted(self) -> bool: ...
    @property
    def interrupts(self) -> list[Any]: ...

    def abort(self) -> None: ...                         # idempotent, best-effort
    def __iter__(self) -> Iterator[Any]: ...             # cached fresh subscription
    def interleave(self, *names: str) -> Iterator[tuple[str, Any]]:
        raise NotImplementedError(
            "sync interleave() is not supported on RemoteGraph; "
            "use astream_events(version='v3') for cross-channel iteration."
        )


class _AsyncRemoteGraphRunStream:
    """Async adapter conforming to AsyncGraphRunStream's public surface."""

    def __init__(
        self,
        *,
        client: LangGraphClient,
        sdk_thread: AsyncThreadStream,
        input: Any,
        config: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None: ...

    async def __aenter__(self) -> "_AsyncRemoteGraphRunStream": ...
    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    @property
    def output(self): ...                                # SDK's _OutputAwaitable; await for value
    @property
    async def interrupted(self) -> bool: ...
    @property
    async def interrupts(self) -> list[Any]: ...

    async def abort(self) -> None: ...
    def __aiter__(self) -> AsyncIterator[Any]: ...       # cached fresh subscription
    def interleave(self, *names: str) -> AsyncIterator[tuple[str, Any]]: ...  # drainer-task pattern
```

### Lifecycle invariants

1. `__enter__` / `__aenter__` opens the SDK CM, then calls `sdk.run.start(...)`. The returned `{"run_id": ...}` is stored on `self._run_id`.
2. If `run.start` raises, the SDK CM is exited with the captured `sys.exc_info()` before the error propagates. The adapter is never observed in a half-entered state.
3. `_closed` flag set by `__exit__` and `abort()` makes both idempotent.
4. `__iter__` / `__aiter__` cache the first iterator returned by `self._sdk.events`. `AsyncThreadStream.events` opens a *fresh* subscription on every access; without caching, two iteration passes would each produce duplicate events.

### `abort()` contract

- Best-effort. Inner failures (`runs.cancel` raising, `sdk.close()` raising) are swallowed and logged at debug level.
- Calls `client.runs.cancel(thread_id, run_id, wait=False)`. `wait=False` so `abort()` returns immediately rather than blocking on server acknowledgement.
- If `_run_id` is `None` (called before `__enter__` completed), no cancel is sent — just close the transport.
- Idempotent: second call is a no-op via `_closed` flag.
- If the v3 SDK exposes a thread-scoped `sdk.run.cancel()` (cleaner than the separate `client.runs.cancel` roundtrip), prefer that — verify when implementing.

### `interleave()` async implementation

Drainer-task pattern: one task per projection name drains into a shared `asyncio.Queue` tagged with the name; consumer yields from the queue as items arrive.

```python
async def interleave(self, *names: str) -> AsyncIterator[tuple[str, Any]]:
    if not names:
        return
    sources = {n: self._resolve_projection(n) for n in names}
    queue: asyncio.Queue[tuple[str, Any] | _Sentinel] = asyncio.Queue()

    async def _drain(n: str, src: AsyncIterator[Any]) -> None:
        try:
            async for item in src:
                await queue.put((n, item))
        finally:
            await queue.put(_SENTINEL)

    drainers = [asyncio.create_task(_drain(n, src)) for n, src in sources.items()]
    pending = len(drainers)
    try:
        while pending > 0:
            item = await queue.get()
            if item is _SENTINEL:
                pending -= 1
                continue
            yield item
    finally:
        for t in drainers:
            if not t.done():
                t.cancel()
        await asyncio.gather(*drainers, return_exceptions=True)
```

`_resolve_projection(name)` maps builtins (`messages`, `tool_calls`, `values`, `subgraphs`) to direct SDK attributes; everything else goes through `self._sdk.extensions[name]`. Cleanup `finally` block must cancel and await every drainer even if the consumer breaks early; `gather(return_exceptions=True)` swallows the `CancelledError`s.

**Behavioral note:** The SDK's `_ExtensionsProjection.__getitem__` returns a lazy projection that yields nothing if the server never emitted the named channel. So unknown names yield nothing rather than raising — a difference from local `GraphRunStream.interleave` which raises `KeyError`. Documented as a remote-vs-local difference.

## `RemoteGraph` dispatchers

```python
# remote.py — sync side (NEW override; currently inherited from Runnable).
def stream_events(
    self,
    input: Any,
    config: RunnableConfig | None = None,
    *,
    version: Literal["v1", "v2", "v3"] = "v2",
    interrupt_before: All | Sequence[str] | None = None,
    interrupt_after: All | Sequence[str] | None = None,
    control: RunControl | None = None,
    transformers: Sequence[Any] | None = None,
    headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> Iterator[StreamEvent] | "_RemoteGraphRunStream":
    if version != "v3":
        return super().stream_events(input, config, version=version, **kwargs)
    self._reject_v3_unsupported(
        control=control,
        transformers=transformers,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        extra_kwargs=kwargs,
    )
    sync_client = self._validate_sync_client()
    sanitized = self._sanitize_config(merge_configs(self.config, config))
    thread_id = sanitized.get("configurable", {}).pop("thread_id", None)
    merged_headers = (
        _merge_tracing_headers(headers) if self.distributed_tracing else headers
    )
    sdk_thread = sync_client.threads.stream(
        thread_id=thread_id,
        assistant_id=self.assistant_id,
        headers=merged_headers,
    )
    return _RemoteGraphRunStream(
        sync_client=sync_client,
        sdk_thread=sdk_thread,
        input=_translate_command_input(input),
        config=sanitized,
        metadata=kwargs.get("metadata"),
    )


# remote.py — async side (REPLACES NotImplementedError body for v3 only).
def astream_events(self, input, config=None, *, version="v2", ..., **kwargs):
    if version != "v3":
        raise NotImplementedError(
            f"RemoteGraph.astream_events(version={version!r}) is not implemented; "
            "use stream() for v1/v2 streaming or version='v3'."
        )
    # ... same dispatch as sync, returns _AsyncRemoteGraphRunStream
```

### `_reject_v3_unsupported` helper

```python
_V3_SUPPORTED_KWARGS = frozenset({"metadata", "headers"})

def _reject_v3_unsupported(
    self, *, control, transformers, interrupt_before, interrupt_after, extra_kwargs
) -> None:
    for name, value in (
        ("control", control),
        ("transformers", transformers),
        ("interrupt_before", interrupt_before),
        ("interrupt_after", interrupt_after),
    ):
        if value:
            raise NotImplementedError(
                f"RemoteGraph.stream_events(version='v3') does not support `{name}=`."
            )
    unknown = set(extra_kwargs) - _V3_SUPPORTED_KWARGS
    if unknown:
        raise NotImplementedError(
            f"RemoteGraph.stream_events(version='v3') does not support "
            f"the following kwargs: {sorted(unknown)!r}. "
            f"Supported: {sorted(_V3_SUPPORTED_KWARGS)!r}."
        )
```

### `_translate_command_input` helper

Mirror of the existing v2 translation at remote.py:763-767:

```python
def _translate_command_input(input: Any) -> Any:
    if isinstance(input, Command):
        return asdict(input)
    return input
```

### `thread_id` minting

Popped from `sanitized["configurable"]` if present. If absent, passed as `None` to `client.threads.stream`, which mints `uuid.uuid4()` client-side. Server never echoes `thread_id` back in v3 — client owns it. Memory `project_v3_thread_id_assignment` documents the broader rationale.

### Header / tracing parity

Identical to v2 (remote.py:782-784): `_merge_tracing_headers(headers)` if `self.distributed_tracing`, else pass headers through unchanged. `client.threads.stream(headers=...)` stores them on the stream and reuses on every command + transport open for the session.

## Error handling

| Source | Policy |
|---|---|
| Caller passes unsupported kwarg | `NotImplementedError` raised by `_reject_v3_unsupported` at dispatch time, before any wire traffic |
| SDK transport / HTTP error | Propagates unchanged (no wrapping in `RemoteException`); matches v2 stream policy of letting SDK errors propagate |
| Run errors (terminal) | Property reads (`output`, `interrupted`, `interrupts`) raise the SDK-native error type; iteration ends |
| Interrupts | Observable state via `interrupted` / `interrupts` properties. Iteration yields events normally; does NOT raise `GraphInterrupt` |
| `run.start` raises inside `__enter__` | SDK CM exited with `sys.exc_info()`; original error re-raised; adapter never visible to caller |
| `__exit__` / `__aexit__` | Delegates to SDK CM; lets exceptions propagate with default Python chaining |
| `abort()` inner failures | Swallowed, logged at debug. Both `runs.cancel` and `sdk.close()` wrapped in `try/except` |

### Behavioral differences vs v2 `RemoteGraph.stream` (protocol-level, not remote-specific)

| Concern | v2 RemoteGraph.stream | v3 (local & remote both) |
|---|---|---|
| Interrupts | Iteration raises `GraphInterrupt` | `interrupted` / `interrupts` properties; iteration yields events normally |
| `Command.PARENT` | Iteration raises `ParentCommand` | Not supported in v3; no wire pathway |
| Generic errors | Iteration raises `RemoteException(data)` | Property reads raise actual error type |

These are v3 protocol contracts, not remote-vs-local divergences. Polymorphism between local `CompiledStateGraph.stream_events(version="v3")` and `RemoteGraph.stream_events(version="v3")` is preserved.

`Command.PARENT` confirmed absent from v3: zero references in `langgraph/stream/`, `langgraph/pregel/main.py` (v3 paths), or the v3 SDK streaming modules.

## Testing strategy

### Unit tests — `libs/langgraph/tests/test_remote_graph_v3.py`

Mocked `LangGraphClient` / `SyncLangGraphClient`; no network, no docker. Run via `make test` in `libs/langgraph/`.

1. `stream_events(version="v3")` constructs `sync_client.threads.stream(thread_id=..., assistant_id=..., headers=...)` with sanitized args.
2. `thread_id` extraction: present-in-configurable case forwarded; absent case passes `None`.
3. `distributed_tracing=True` merges tracing headers into the `threads.stream(headers=...)` call.
4. Each unsupported kwarg raises `NotImplementedError` before any SDK call: `control=`, `transformers=`, `interrupt_before=`, `interrupt_after=`, unknown `context=`.
5. `v1`/`v2`: sync `stream_events` delegates to `super().stream_events`; async raises `NotImplementedError` with corrected message.
6. `_RemoteGraphRunStream` lifecycle:
   - `__enter__` calls `sdk.__enter__()` then `sdk.run.start(...)`, stores `run_id`.
   - `__enter__` with `run.start` raising → SDK CM exited cleanly, error propagates.
   - `__exit__` calls `sdk.__exit__` with right `exc_info`.
   - `abort()` calls `client.runs.cancel(thread_id, run_id, wait=False)` then `sdk.close()`.
   - `abort()` before `__enter__` → no `runs.cancel`, close is no-op.
   - `abort()` twice → second is no-op.
   - `abort()` with `runs.cancel` raising → swallowed, `sdk.close()` still runs.
7. `_AsyncRemoteGraphRunStream` lifecycle: mirror of (6) with async variants.
8. `interleave(*names)` drainer-task pattern: arrival-order delivery of `(name, item)` tuples; early `break` cancels drainers without leaks.
9. Sync `interleave` raises `NotImplementedError` with the documented message.
10. `Command` input translation: `run.start(input=...)` receives the dict form.
11. `__iter__` / `__aiter__` caching: two iteration passes share the same subscription.
12. `interrupted` / `interrupts` pass through from the SDK without re-raising.

### Integration tests — `libs/sdk-py/tests/integration/test_remote_graph_v3.py`

Behind the existing `-m integration` marker. Uses the docker stack at `libs/sdk-py/integration/`.

13. `RemoteGraph(assistant_id="agent", url="http://localhost:2024")` against the `agent` graph in `libs/sdk-py/integration/graph/streaming_graph.py`.
14. Async happy path: `async with graph.astream_events(input, version="v3") as stream:` → iterate `stream.messages` → await `stream.output` → assert `not stream.interrupted`.
15. Async interrupt path: against `tools_agent`, run interrupts → `stream.interrupted is True`, `len(stream.interrupts) >= 1`. (Resume flow uses a separate `astream_events` call with `Command(resume=...)`; not exercised in this PR's adapter.)
16. Sync happy path: mirror of (14).
17. `abort()` mid-run: cancel the run, assert via subsequent `runs.get(run_id)` that status is `cancelled` / `interrupted`.

### CI

Extend `.github/workflows/_sdk_integration_test.yml` path filter to also trigger on `libs/langgraph/langgraph/pregel/remote.py` and `libs/langgraph/langgraph/pregel/_remote_run_stream.py` changes.

## Out of scope (follow-ups)

- v1 / v2 `astream_events` real implementation (replaces remaining `NotImplementedError`).
- Server-side support for `interrupt_before` / `interrupt_after` / `control` on v3 runs. Once available, lift the `NotImplementedError` in `_reject_v3_unsupported`.
- Monotonic server-side event stamps to give `interleave()` strict arrival ordering matching local v3.
- Sync `interleave()` via drainer threads.
- `context` kwarg passthrough if `client.threads.stream` or `run.start` adds support.
- Resume-after-interrupt convenience on the adapter (currently callers must do a fresh `astream_events(input=Command(resume=...))`).

## Risks

| Risk | Mitigation |
|---|---|
| Branch is not runnable until v3 SDK lands on main | Document; rebase before merge. Spec is decoupled from order-of-landing. |
| `AsyncThreadStream.output` is `_OutputAwaitable` vs local `AsyncGraphRunStream.output` being `async def` property | Returning the SDK awaitable from a plain `@property` works because callers do `await stream.output` either way. Unit test verifies. |
| `runs.cancel` signature could differ from assumed `(thread_id, run_id, *, wait=False)` | Verify when implementing; if v3 SDK exposes `sdk.run.cancel()`, prefer that. |
| SDK `events` property opens fresh subscription per access | Adapter caches first iterator; unit test #11 verifies no duplicate events on repeat iteration. |
| `Command.PARENT` was the only v2-stream exception not preserved in v3 | Verified absent from v3 protocol entirely; documented as protocol-level limitation, not adapter limitation. |
