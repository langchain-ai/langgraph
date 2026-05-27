# RemoteGraph v3 streaming support — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `stream_events(version="v3")` / `astream_events(version="v3")` to `RemoteGraph` by wrapping `langgraph_sdk` v3 thread-stream surface with two duck-typed adapter classes.

**Architecture:** Two private adapter classes in a new module (`_remote_run_stream.py`) conform to the local `GraphRunStream` / `AsyncGraphRunStream` public surface but delegate every operation to `langgraph_sdk._async.stream.AsyncThreadStream` / `langgraph_sdk._sync.stream.SyncThreadStream`. `RemoteGraph` grows a sync `stream_events` override and replaces the `astream_events` `NotImplementedError` body for v3 only. Unsupported kwargs (`control`, `transformers`, `interrupt_before`, `interrupt_after`, unknown `**kwargs`) raise `NotImplementedError` at dispatch.

**Tech Stack:** Python 3.10+, `langgraph` (libs/langgraph), `langgraph-sdk` 0.4.0 (libs/sdk-py editable), `pytest` with `unittest.mock`, `make test` / `make lint` / `make format`. Spec: `docs/superpowers/specs/2026-05-21-remote-graph-v3-streaming-design.md`.

**Branch:** Off `main` (v3 SDK has now landed). Constraint bump in `libs/langgraph/pyproject.toml` (`langgraph-sdk<0.4.0` → `<0.5.0`) is deferred to a separate post-publication PR per user decision; tests resolve against the editable workspace SDK.

---

## File Structure

**Create:**
- `libs/langgraph/langgraph/pregel/_remote_run_stream.py` — adapter classes (`_RemoteGraphRunStream`, `_AsyncRemoteGraphRunStream`) and small helpers (`_SENTINEL`, `_resolve_projection`).
- `libs/langgraph/tests/test_remote_graph_v3.py` — unit tests with mocked SDK clients.
- `libs/sdk-py/tests/integration/test_remote_graph_v3.py` — integration tests against the docker stack at `libs/sdk-py/integration/`.

**Modify:**
- `libs/langgraph/langgraph/pregel/remote.py` — add `stream_events` override, replace `astream_events` body for v3 dispatch, add `_reject_v3_unsupported` and `_translate_command_input` helpers.
- `.github/workflows/_sdk_integration_test.yml` — extend path filter to trigger on changes to `libs/langgraph/langgraph/pregel/remote.py` and `libs/langgraph/langgraph/pregel/_remote_run_stream.py`.

---

## Task 1: Adapter module skeleton + import check

**Files:**
- Create: `libs/langgraph/langgraph/pregel/_remote_run_stream.py`

- [ ] **Step 1: Write the skeleton**

```python
# libs/langgraph/langgraph/pregel/_remote_run_stream.py
from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk._sync.stream import SyncThreadStream
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient

logger = logging.getLogger(__name__)

_SENTINEL: Any = object()


class _RemoteGraphRunStream:
    """Sync adapter: SyncThreadStream -> GraphRunStream surface."""

    def __init__(
        self,
        *,
        sync_client: SyncLangGraphClient,
        sdk_thread: SyncThreadStream,
        input: Any,
        config: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        self._client = sync_client
        self._sdk = sdk_thread
        self._start_kwargs: dict[str, Any] = {
            "input": input,
            "config": config,
            "metadata": metadata,
        }
        self._run_id: str | None = None
        self._closed = False
        self._events_iter: Iterator[Any] | None = None


class _AsyncRemoteGraphRunStream:
    """Async adapter: AsyncThreadStream -> AsyncGraphRunStream surface."""

    def __init__(
        self,
        *,
        client: LangGraphClient,
        sdk_thread: AsyncThreadStream,
        input: Any,
        config: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        self._client = client
        self._sdk = sdk_thread
        self._start_kwargs: dict[str, Any] = {
            "input": input,
            "config": config,
            "metadata": metadata,
        }
        self._run_id: str | None = None
        self._closed = False
        self._events_aiter: AsyncIterator[Any] | None = None
```

- [ ] **Step 2: Verify imports resolve**

Run from repo root: `cd libs/langgraph && python -c "from langgraph.pregel._remote_run_stream import _RemoteGraphRunStream, _AsyncRemoteGraphRunStream; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add libs/langgraph/langgraph/pregel/_remote_run_stream.py
git commit -m "Add RemoteGraph v3 adapter module skeleton"
```

---

## Task 2: Sync adapter lifecycle (`__enter__` / `__exit__`)

**Files:**
- Modify: `libs/langgraph/langgraph/pregel/_remote_run_stream.py`
- Create: `libs/langgraph/tests/test_remote_graph_v3.py`

- [ ] **Step 1: Write failing tests**

```python
# libs/langgraph/tests/test_remote_graph_v3.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langgraph.pregel._remote_run_stream import _RemoteGraphRunStream


def _make_sync_adapter(*, run_start_returns=None, run_start_raises=None):
    sync_client = MagicMock()
    sdk_thread = MagicMock()
    sdk_thread.thread_id = "thread-abc"
    sdk_thread.__enter__ = MagicMock(return_value=sdk_thread)
    sdk_thread.__exit__ = MagicMock(return_value=None)
    sdk_thread.run = MagicMock()
    if run_start_raises is not None:
        sdk_thread.run.start = MagicMock(side_effect=run_start_raises)
    else:
        sdk_thread.run.start = MagicMock(
            return_value=run_start_returns or {"run_id": "run-xyz"}
        )
    adapter = _RemoteGraphRunStream(
        sync_client=sync_client,
        sdk_thread=sdk_thread,
        input={"x": 1},
        config={"configurable": {}},
        metadata=None,
    )
    return adapter, sync_client, sdk_thread


def test_enter_calls_sdk_enter_then_run_start_and_captures_run_id():
    adapter, _, sdk_thread = _make_sync_adapter()
    with adapter as stream:
        assert stream is adapter
        sdk_thread.__enter__.assert_called_once()
        sdk_thread.run.start.assert_called_once_with(
            input={"x": 1}, config={"configurable": {}}, metadata=None
        )
        assert adapter._run_id == "run-xyz"


def test_exit_delegates_to_sdk_exit_with_exc_info():
    adapter, _, sdk_thread = _make_sync_adapter()
    with adapter:
        pass
    sdk_thread.__exit__.assert_called_once_with(None, None, None)


def test_enter_unwinds_sdk_cm_when_run_start_raises():
    adapter, _, sdk_thread = _make_sync_adapter(
        run_start_raises=RuntimeError("start boom")
    )
    with pytest.raises(RuntimeError, match="start boom"):
        with adapter:
            pytest.fail("body should not run")
    sdk_thread.__enter__.assert_called_once()
    sdk_thread.__exit__.assert_called_once()
    exc_info = sdk_thread.__exit__.call_args.args
    assert exc_info[0] is RuntimeError
    assert isinstance(exc_info[1], RuntimeError)
    assert adapter._run_id is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `libs/langgraph/`: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 3 failures with `AttributeError` (no `__enter__` / `__exit__` on adapter).

- [ ] **Step 3: Implement `__enter__` and `__exit__`**

Append to `_RemoteGraphRunStream` in `_remote_run_stream.py`:

```python
    def __enter__(self) -> _RemoteGraphRunStream:
        if self._closed:
            raise RuntimeError("_RemoteGraphRunStream already closed")
        self._sdk.__enter__()
        try:
            result = self._sdk.run.start(**self._start_kwargs)
        except BaseException:
            self._sdk.__exit__(*sys.exc_info())
            raise
        self._run_id = result["run_id"]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._closed:
            return
        self._closed = True
        self._sdk.__exit__(exc_type, exc, tb)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/langgraph/langgraph/pregel/_remote_run_stream.py libs/langgraph/tests/test_remote_graph_v3.py
git commit -m "Add sync RemoteGraph v3 adapter lifecycle"
```

---

## Task 3: Sync adapter properties + `__iter__` caching + `abort()` + sync `interleave` reject

**Files:**
- Modify: `libs/langgraph/langgraph/pregel/_remote_run_stream.py`
- Modify: `libs/langgraph/tests/test_remote_graph_v3.py`

- [ ] **Step 1: Write failing tests**

Append to `test_remote_graph_v3.py`:

```python
def test_output_interrupted_interrupts_passthrough():
    adapter, _, sdk_thread = _make_sync_adapter()
    sdk_thread.output = {"foo": 1}
    sdk_thread.interrupted = True
    sdk_thread.interrupts = [{"interrupt_id": "i1", "namespace": [], "value": "v"}]
    with adapter as stream:
        assert stream.output == {"foo": 1}
        assert stream.interrupted is True
        assert stream.interrupts == [
            {"interrupt_id": "i1", "namespace": [], "value": "v"}
        ]


def test_iter_caches_first_subscription():
    adapter, _, sdk_thread = _make_sync_adapter()
    fake_events = [object(), object(), object()]
    sdk_thread.events = iter(fake_events)
    with adapter as stream:
        first = iter(stream)
        second = iter(stream)
        assert first is second
        assert list(first) == fake_events


def test_abort_cancels_run_and_closes_sdk():
    adapter, sync_client, sdk_thread = _make_sync_adapter()
    with adapter as stream:
        stream.abort()
        sync_client.runs.cancel.assert_called_once_with(
            "thread-abc", "run-xyz", wait=False
        )
        sdk_thread.close.assert_called_once()


def test_abort_before_enter_skips_cancel_but_closes_sdk():
    adapter, sync_client, sdk_thread = _make_sync_adapter()
    adapter.abort()
    sync_client.runs.cancel.assert_not_called()
    sdk_thread.close.assert_called_once()


def test_abort_is_idempotent():
    adapter, sync_client, sdk_thread = _make_sync_adapter()
    with adapter as stream:
        stream.abort()
        stream.abort()
        assert sync_client.runs.cancel.call_count == 1
        assert sdk_thread.close.call_count == 1


def test_abort_swallows_cancel_failure_and_still_closes():
    adapter, sync_client, sdk_thread = _make_sync_adapter()
    sync_client.runs.cancel.side_effect = RuntimeError("cancel boom")
    with adapter as stream:
        stream.abort()
        sdk_thread.close.assert_called_once()


def test_sync_interleave_raises_not_implemented():
    adapter, _, _ = _make_sync_adapter()
    with adapter as stream:
        with pytest.raises(NotImplementedError, match="sync interleave"):
            list(stream.interleave("messages"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 7 failures (missing attributes/methods).

- [ ] **Step 3: Implement properties, `__iter__`, `abort`, `interleave`**

Append to `_RemoteGraphRunStream`:

```python
    @property
    def output(self) -> Any:
        return self._sdk.output

    @property
    def interrupted(self) -> bool:
        return self._sdk.interrupted

    @property
    def interrupts(self) -> list[Any]:
        return list(self._sdk.interrupts)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._run_id is not None:
            try:
                self._client.runs.cancel(
                    self._sdk.thread_id, self._run_id, wait=False
                )
            except Exception:
                logger.debug("abort: runs.cancel failed", exc_info=True)
        try:
            self._sdk.close()
        except Exception:
            logger.debug("abort: sdk.close failed", exc_info=True)

    def __iter__(self) -> Iterator[Any]:
        if self._events_iter is None:
            self._events_iter = iter(self._sdk.events)
        return self._events_iter

    def interleave(self, *names: str) -> Iterator[tuple[str, Any]]:
        raise NotImplementedError(
            "sync interleave() is not supported on RemoteGraph; "
            "use astream_events(version='v3') for cross-channel iteration."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 10 passed (3 from Task 2 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add libs/langgraph/langgraph/pregel/_remote_run_stream.py libs/langgraph/tests/test_remote_graph_v3.py
git commit -m "Add sync RemoteGraph v3 adapter accessors, abort, iter caching"
```

---

## Task 4: Async adapter lifecycle (`__aenter__` / `__aexit__`)

**Files:**
- Modify: `libs/langgraph/langgraph/pregel/_remote_run_stream.py`
- Modify: `libs/langgraph/tests/test_remote_graph_v3.py`

- [ ] **Step 1: Write failing tests**

Append to `test_remote_graph_v3.py`:

```python
from unittest.mock import AsyncMock

from langgraph.pregel._remote_run_stream import _AsyncRemoteGraphRunStream


def _make_async_adapter(*, run_start_returns=None, run_start_raises=None):
    client = MagicMock()
    client.runs.cancel = AsyncMock()
    sdk_thread = MagicMock()
    sdk_thread.thread_id = "thread-abc"
    sdk_thread.__aenter__ = AsyncMock(return_value=sdk_thread)
    sdk_thread.__aexit__ = AsyncMock(return_value=None)
    sdk_thread.close = AsyncMock()
    sdk_thread.run = MagicMock()
    if run_start_raises is not None:
        sdk_thread.run.start = AsyncMock(side_effect=run_start_raises)
    else:
        sdk_thread.run.start = AsyncMock(
            return_value=run_start_returns or {"run_id": "run-xyz"}
        )
    adapter = _AsyncRemoteGraphRunStream(
        client=client,
        sdk_thread=sdk_thread,
        input={"x": 1},
        config={"configurable": {}},
        metadata=None,
    )
    return adapter, client, sdk_thread


@pytest.mark.anyio
async def test_aenter_calls_sdk_aenter_then_run_start_and_captures_run_id():
    adapter, _, sdk_thread = _make_async_adapter()
    async with adapter as stream:
        assert stream is adapter
        sdk_thread.__aenter__.assert_awaited_once()
        sdk_thread.run.start.assert_awaited_once_with(
            input={"x": 1}, config={"configurable": {}}, metadata=None
        )
        assert adapter._run_id == "run-xyz"


@pytest.mark.anyio
async def test_aexit_delegates_to_sdk_aexit():
    adapter, _, sdk_thread = _make_async_adapter()
    async with adapter:
        pass
    sdk_thread.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.anyio
async def test_aenter_unwinds_sdk_cm_when_run_start_raises():
    adapter, _, sdk_thread = _make_async_adapter(
        run_start_raises=RuntimeError("start boom")
    )
    with pytest.raises(RuntimeError, match="start boom"):
        async with adapter:
            pytest.fail("body should not run")
    sdk_thread.__aenter__.assert_awaited_once()
    sdk_thread.__aexit__.assert_awaited_once()
    assert adapter._run_id is None
```

If `pytest.mark.anyio` isn't already configured in `libs/langgraph/tests/conftest.py`, switch to `pytest.mark.asyncio` — check existing async tests in `libs/langgraph/tests/` to see which marker is in use. Run `grep -l "pytest.mark" libs/langgraph/tests/*.py | head -3` and look at how they mark async tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 3 failures (missing `__aenter__`/`__aexit__`).

- [ ] **Step 3: Implement `__aenter__` and `__aexit__`**

Append to `_AsyncRemoteGraphRunStream`:

```python
    async def __aenter__(self) -> _AsyncRemoteGraphRunStream:
        if self._closed:
            raise RuntimeError("_AsyncRemoteGraphRunStream already closed")
        await self._sdk.__aenter__()
        try:
            result = await self._sdk.run.start(**self._start_kwargs)
        except BaseException:
            await self._sdk.__aexit__(*sys.exc_info())
            raise
        self._run_id = result["run_id"]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._closed:
            return
        self._closed = True
        await self._sdk.__aexit__(exc_type, exc, tb)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/langgraph/langgraph/pregel/_remote_run_stream.py libs/langgraph/tests/test_remote_graph_v3.py
git commit -m "Add async RemoteGraph v3 adapter lifecycle"
```

---

## Task 5: Async adapter properties + `__aiter__` caching + `abort()`

**Files:**
- Modify: `libs/langgraph/langgraph/pregel/_remote_run_stream.py`
- Modify: `libs/langgraph/tests/test_remote_graph_v3.py`

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.anyio
async def test_async_output_interrupted_interrupts_passthrough():
    adapter, _, sdk_thread = _make_async_adapter()

    async def _fake_output_awaitable():
        return {"foo": 1}

    sdk_thread.output = _fake_output_awaitable()
    sdk_thread.interrupted = True
    sdk_thread.interrupts = [{"interrupt_id": "i1", "namespace": [], "value": "v"}]
    async with adapter as stream:
        assert await stream.output == {"foo": 1}
        assert await stream.interrupted is True
        assert await stream.interrupts == [
            {"interrupt_id": "i1", "namespace": [], "value": "v"}
        ]


@pytest.mark.anyio
async def test_aiter_caches_first_subscription():
    adapter, _, sdk_thread = _make_async_adapter()

    class _FakeAsyncEvents:
        def __init__(self, items):
            self._items = list(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    sdk_thread.events = _FakeAsyncEvents([object(), object()])
    async with adapter as stream:
        first = stream.__aiter__()
        second = stream.__aiter__()
        assert first is second


@pytest.mark.anyio
async def test_async_abort_cancels_run_and_closes_sdk():
    adapter, client, sdk_thread = _make_async_adapter()
    async with adapter as stream:
        await stream.abort()
        client.runs.cancel.assert_awaited_once_with(
            "thread-abc", "run-xyz", wait=False
        )
        sdk_thread.close.assert_awaited_once()


@pytest.mark.anyio
async def test_async_abort_before_aenter_skips_cancel():
    adapter, client, sdk_thread = _make_async_adapter()
    await adapter.abort()
    client.runs.cancel.assert_not_awaited()
    sdk_thread.close.assert_awaited_once()


@pytest.mark.anyio
async def test_async_abort_swallows_cancel_failure():
    adapter, client, sdk_thread = _make_async_adapter()
    client.runs.cancel.side_effect = RuntimeError("cancel boom")
    async with adapter as stream:
        await stream.abort()
        sdk_thread.close.assert_awaited_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 5 failures.

- [ ] **Step 3: Implement async accessors and `abort`**

Append to `_AsyncRemoteGraphRunStream`:

```python
    @property
    def output(self) -> Any:
        return self._sdk.output

    @property
    async def interrupted(self) -> bool:
        return self._sdk.interrupted

    @property
    async def interrupts(self) -> list[Any]:
        return list(self._sdk.interrupts)

    async def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._run_id is not None:
            try:
                await self._client.runs.cancel(
                    self._sdk.thread_id, self._run_id, wait=False
                )
            except Exception:
                logger.debug("abort: runs.cancel failed", exc_info=True)
        try:
            await self._sdk.close()
        except Exception:
            logger.debug("abort: sdk.close failed", exc_info=True)

    def __aiter__(self) -> AsyncIterator[Any]:
        if self._events_aiter is None:
            self._events_aiter = self._sdk.events.__aiter__()
        return self._events_aiter
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/langgraph/langgraph/pregel/_remote_run_stream.py libs/langgraph/tests/test_remote_graph_v3.py
git commit -m "Add async RemoteGraph v3 adapter accessors, abort, aiter caching"
```

---

## Task 6: Async `interleave()` drainer-task implementation

**Files:**
- Modify: `libs/langgraph/langgraph/pregel/_remote_run_stream.py`
- Modify: `libs/langgraph/tests/test_remote_graph_v3.py`

- [ ] **Step 1: Write failing tests**

```python
class _FakeAsyncIter:
    def __init__(self, items, *, sleeps=None):
        self._items = list(items)
        self._sleeps = list(sleeps) if sleeps else [0.0] * len(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        delay = self._sleeps.pop(0)
        if delay:
            await asyncio.sleep(delay)
        return self._items.pop(0)


@pytest.mark.anyio
async def test_interleave_yields_tagged_tuples():
    adapter, _, sdk_thread = _make_async_adapter()
    sdk_thread.messages = _FakeAsyncIter(["m1", "m2"])
    sdk_thread.values = _FakeAsyncIter([{"v": 1}])
    async with adapter as stream:
        got = []
        async for name, item in stream.interleave("messages", "values"):
            got.append((name, item))
    names = [n for n, _ in got]
    assert sorted(names) == ["messages", "messages", "values"]


@pytest.mark.anyio
async def test_interleave_routes_unknown_through_extensions():
    adapter, _, sdk_thread = _make_async_adapter()
    sdk_thread.extensions = {"custom": _FakeAsyncIter(["c1", "c2"])}
    async with adapter as stream:
        got = [
            (name, item)
            async for name, item in stream.interleave("custom")
        ]
    assert got == [("custom", "c1"), ("custom", "c2")]


@pytest.mark.anyio
async def test_interleave_no_names_yields_nothing():
    adapter, _, _ = _make_async_adapter()
    async with adapter as stream:
        got = [
            (name, item)
            async for name, item in stream.interleave()
        ]
    assert got == []


@pytest.mark.anyio
async def test_interleave_cancels_drainers_on_early_break():
    adapter, _, sdk_thread = _make_async_adapter()
    sdk_thread.messages = _FakeAsyncIter(
        ["m1", "m2", "m3", "m4"], sleeps=[0.0, 0.05, 0.05, 0.05]
    )
    sdk_thread.values = _FakeAsyncIter(
        [{"v": 1}, {"v": 2}], sleeps=[0.0, 0.05]
    )
    async with adapter as stream:
        async for _name, _item in stream.interleave("messages", "values"):
            break
    # Reaching here without hanging or leaked tasks is the success condition.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 4 failures (no `interleave` on async adapter).

- [ ] **Step 3: Implement `interleave` and `_resolve_projection`**

Append to `_AsyncRemoteGraphRunStream`:

```python
    def _resolve_projection(self, name: str) -> AsyncIterator[Any]:
        builtin = {
            "messages": self._sdk.messages,
            "tool_calls": self._sdk.tool_calls,
            "values": self._sdk.values,
            "subgraphs": self._sdk.subgraphs,
        }
        source = builtin[name] if name in builtin else self._sdk.extensions[name]
        return source.__aiter__()

    async def interleave(
        self, *names: str
    ) -> AsyncIterator[tuple[str, Any]]:
        if not names:
            return
        sources = {n: self._resolve_projection(n) for n in names}
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def _drain(channel: str, src: AsyncIterator[Any]) -> None:
            try:
                async for item in src:
                    await queue.put((channel, item))
            finally:
                await queue.put(_SENTINEL)

        drainers = [
            asyncio.create_task(_drain(n, src)) for n, src in sources.items()
        ]
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

Note: `interleave` is `async def` and uses `yield`, making it an async generator. The signature in the docstring of `AsyncGraphRunStream` may say it returns an `AsyncIterator`, which an async generator satisfies.

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/langgraph/langgraph/pregel/_remote_run_stream.py libs/langgraph/tests/test_remote_graph_v3.py
git commit -m "Add async interleave drainer-task implementation"
```

---

## Task 7: `_reject_v3_unsupported` and `_translate_command_input` helpers on `RemoteGraph`

**Files:**
- Modify: `libs/langgraph/langgraph/pregel/remote.py`
- Modify: `libs/langgraph/tests/test_remote_graph_v3.py`

- [ ] **Step 1: Write failing tests**

Append to `test_remote_graph_v3.py`:

```python
from dataclasses import asdict

from langgraph.pregel.remote import (
    RemoteGraph,
    _translate_command_input,
    _V3_SUPPORTED_KWARGS,
)
from langgraph.types import Command


def _make_remote_graph() -> RemoteGraph:
    sync_client = MagicMock()
    async_client = MagicMock()
    rg = RemoteGraph(
        "agent",
        client=async_client,
        sync_client=sync_client,
    )
    return rg


def test_reject_v3_unsupported_passes_when_all_clear():
    rg = _make_remote_graph()
    rg._reject_v3_unsupported(
        control=None,
        transformers=None,
        interrupt_before=None,
        interrupt_after=None,
        extra_kwargs={},
    )


@pytest.mark.parametrize(
    "kwarg_name,kwarg_value",
    [
        ("control", object()),
        ("transformers", [object()]),
        ("interrupt_before", ["node_a"]),
        ("interrupt_after", ["node_b"]),
    ],
)
def test_reject_v3_unsupported_raises_per_kwarg(kwarg_name, kwarg_value):
    rg = _make_remote_graph()
    kwargs = dict(
        control=None,
        transformers=None,
        interrupt_before=None,
        interrupt_after=None,
        extra_kwargs={},
    )
    kwargs[kwarg_name] = kwarg_value
    with pytest.raises(NotImplementedError, match=f"`{kwarg_name}=`"):
        rg._reject_v3_unsupported(**kwargs)


def test_reject_v3_unsupported_raises_on_unknown_extra_kwarg():
    rg = _make_remote_graph()
    with pytest.raises(NotImplementedError, match="context"):
        rg._reject_v3_unsupported(
            control=None,
            transformers=None,
            interrupt_before=None,
            interrupt_after=None,
            extra_kwargs={"context": {}},
        )


def test_reject_v3_unsupported_allows_metadata_and_headers():
    rg = _make_remote_graph()
    rg._reject_v3_unsupported(
        control=None,
        transformers=None,
        interrupt_before=None,
        interrupt_after=None,
        extra_kwargs={"metadata": {"a": 1}, "headers": {"X": "y"}},
    )


def test_translate_command_input_converts_command_to_dict():
    cmd = Command(update={"a": 1})
    assert _translate_command_input(cmd) == asdict(cmd)


def test_translate_command_input_passes_through_non_command():
    assert _translate_command_input({"a": 1}) == {"a": 1}
    assert _translate_command_input(None) is None


def test_v3_supported_kwargs_known_set():
    assert _V3_SUPPORTED_KWARGS == frozenset({"metadata", "headers"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: failures with `ImportError` for `_translate_command_input` / `_V3_SUPPORTED_KWARGS` and `AttributeError` for `_reject_v3_unsupported`.

- [ ] **Step 3: Add helpers to `remote.py`**

In `libs/langgraph/langgraph/pregel/remote.py`, add module-level constant near the existing `_CONF_DROPLIST` (around line 74):

```python
_V3_SUPPORTED_KWARGS = frozenset({"metadata", "headers"})
```

Add module-level helper near `_sanitize_config_value` (around line 84):

```python
def _translate_command_input(input: Any) -> Any:
    """Translate a local Command into the SDK Command dict shape, else passthrough."""
    if isinstance(input, Command):
        return asdict(input)
    return input
```

`Command` and `asdict` are already imported at the top of `remote.py` (line 5 imports `asdict`; `Command` is imported via `langgraph.types`).

Add method on `RemoteGraph` (near other private helpers like `_validate_client`):

```python
    def _reject_v3_unsupported(
        self,
        *,
        control: Any,
        transformers: Any,
        interrupt_before: Any,
        interrupt_after: Any,
        extra_kwargs: dict[str, Any],
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 32 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/langgraph/langgraph/pregel/remote.py libs/langgraph/tests/test_remote_graph_v3.py
git commit -m "Add RemoteGraph v3 kwarg rejection and Command input translation helpers"
```

---

## Task 8: `RemoteGraph.stream_events(version="v3")` sync dispatch

**Files:**
- Modify: `libs/langgraph/langgraph/pregel/remote.py`
- Modify: `libs/langgraph/tests/test_remote_graph_v3.py`

- [ ] **Step 1: Write failing tests**

```python
def test_stream_events_v3_constructs_sdk_thread_with_sanitized_args():
    sync_client = MagicMock()
    sdk_thread = MagicMock()
    sync_client.threads.stream.return_value = sdk_thread
    rg = RemoteGraph(
        "agent",
        client=MagicMock(),
        sync_client=sync_client,
    )
    result = rg.stream_events(
        {"input_key": 1},
        config={"configurable": {"thread_id": "t1", "user": "u"}},
        version="v3",
    )
    assert isinstance(result, _RemoteGraphRunStream)
    sync_client.threads.stream.assert_called_once()
    call = sync_client.threads.stream.call_args
    assert call.kwargs["thread_id"] == "t1"
    assert call.kwargs["assistant_id"] == "agent"
    assert call.kwargs["headers"] is None


def test_stream_events_v3_passes_none_thread_id_when_absent():
    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=sync_client)
    rg.stream_events({"x": 1}, version="v3")
    call = sync_client.threads.stream.call_args
    assert call.kwargs["thread_id"] is None


def test_stream_events_v3_rejects_unsupported_kwargs_before_sdk_call():
    sync_client = MagicMock()
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=sync_client)
    with pytest.raises(NotImplementedError, match="control"):
        rg.stream_events({"x": 1}, version="v3", control=object())
    sync_client.threads.stream.assert_not_called()


def test_stream_events_v3_translates_command_input():
    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=sync_client)
    adapter = rg.stream_events(Command(update={"a": 1}), version="v3")
    assert adapter._start_kwargs["input"] == {
        "update": {"a": 1},
        "resume": None,
        "goto": (),
        "graph": None,
    }


def test_stream_events_v3_strips_checkpoint_keys_from_configurable():
    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=sync_client)
    adapter = rg.stream_events(
        {"x": 1},
        config={
            "configurable": {
                "thread_id": "t1",
                "checkpoint_id": "c1",
                "checkpoint_ns": "ns",
                "user": "u",
            }
        },
        version="v3",
    )
    sent_config = adapter._start_kwargs["config"]
    assert "checkpoint_id" not in sent_config["configurable"]
    assert "checkpoint_ns" not in sent_config["configurable"]
    assert sent_config["configurable"]["user"] == "u"


def test_stream_events_v3_merges_tracing_headers_when_distributed_tracing(
    monkeypatch,
):
    from langgraph.pregel import remote as remote_mod

    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph(
        "agent",
        client=MagicMock(),
        sync_client=sync_client,
        distributed_tracing=True,
    )
    captured = {}

    def fake_merge(headers):
        captured["arg"] = headers
        return {"x-ls-trace": "1", **(headers or {})}

    monkeypatch.setattr(remote_mod, "_merge_tracing_headers", fake_merge)
    rg.stream_events({"x": 1}, version="v3", headers={"X-Custom": "y"})
    assert captured["arg"] == {"X-Custom": "y"}
    sent_headers = sync_client.threads.stream.call_args.kwargs["headers"]
    assert sent_headers["x-ls-trace"] == "1"
    assert sent_headers["X-Custom"] == "y"


def test_stream_events_v3_passes_headers_unchanged_without_tracing():
    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph(
        "agent",
        client=MagicMock(),
        sync_client=sync_client,
        distributed_tracing=False,
    )
    rg.stream_events({"x": 1}, version="v3", headers={"X-Custom": "y"})
    sent_headers = sync_client.threads.stream.call_args.kwargs["headers"]
    assert sent_headers == {"X-Custom": "y"}


def test_stream_events_non_v3_delegates_to_super():
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=MagicMock())
    # super().stream_events on Runnable raises for unsupported version="v2"
    # behavior, but at minimum should NOT route through our v3 path.
    sync_client_attr = rg.sync_client
    try:
        rg.stream_events({"x": 1}, version="v2")
    except Exception:
        pass
    sync_client_attr.threads.stream.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: failures (no override; current `stream_events` is inherited from `Runnable`).

- [ ] **Step 3: Add `stream_events` override on `RemoteGraph`**

Add to `RemoteGraph` in `remote.py` (just before `astream_events` at line 999). Also extend the top-of-file imports to include `_RemoteGraphRunStream` and `_AsyncRemoteGraphRunStream`:

```python
from langgraph.pregel._remote_run_stream import (
    _AsyncRemoteGraphRunStream,
    _RemoteGraphRunStream,
)
```

And add the method body:

```python
    def stream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        control: Any = None,
        transformers: Sequence[Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream events from this remote graph.

        For `version="v3"`, returns a `_RemoteGraphRunStream` whose surface
        matches the local `GraphRunStream`. For other versions, delegates to
        `Runnable.stream_events`.
        """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 40 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/langgraph/langgraph/pregel/remote.py libs/langgraph/tests/test_remote_graph_v3.py
git commit -m "Add RemoteGraph.stream_events v3 dispatch"
```

---

## Task 9: `RemoteGraph.astream_events(version="v3")` async dispatch

**Files:**
- Modify: `libs/langgraph/langgraph/pregel/remote.py`
- Modify: `libs/langgraph/tests/test_remote_graph_v3.py`

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.anyio
async def test_astream_events_v3_constructs_sdk_thread():
    client = MagicMock()
    sdk_thread = MagicMock()
    client.threads.stream.return_value = sdk_thread
    rg = RemoteGraph("agent", client=client, sync_client=MagicMock())
    result = rg.astream_events(
        {"x": 1},
        config={"configurable": {"thread_id": "t1"}},
        version="v3",
    )
    assert isinstance(result, _AsyncRemoteGraphRunStream)
    call = client.threads.stream.call_args
    assert call.kwargs["thread_id"] == "t1"
    assert call.kwargs["assistant_id"] == "agent"


@pytest.mark.anyio
async def test_astream_events_v3_rejects_unsupported_kwargs():
    client = MagicMock()
    rg = RemoteGraph("agent", client=client, sync_client=MagicMock())
    with pytest.raises(NotImplementedError, match="transformers"):
        rg.astream_events({"x": 1}, version="v3", transformers=[object()])
    client.threads.stream.assert_not_called()


@pytest.mark.anyio
async def test_astream_events_non_v3_raises_not_implemented():
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=MagicMock())
    with pytest.raises(NotImplementedError, match="not implemented"):
        # astream_events is a coroutine-ish; the body raises immediately.
        result = rg.astream_events({"x": 1}, version="v2")
        if hasattr(result, "__aiter__"):
            async for _ in result:
                pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 3 failures (current `astream_events` raises for all versions including v3).

- [ ] **Step 3: Replace `astream_events` body**

Replace the existing `astream_events` method on `RemoteGraph` (currently at remote.py:999-1013) with:

```python
    def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        control: Any = None,
        transformers: Sequence[Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Async-stream events from this remote graph.

        For `version="v3"`, returns an `_AsyncRemoteGraphRunStream`. For
        `version="v1"`/`"v2"`, raises NotImplementedError (use `astream`).
        """
        if version != "v3":
            raise NotImplementedError(
                f"RemoteGraph.astream_events(version={version!r}) is not "
                "implemented; use astream() for v1/v2 streaming or "
                "version='v3'."
            )
        self._reject_v3_unsupported(
            control=control,
            transformers=transformers,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            extra_kwargs=kwargs,
        )
        client = self._validate_client()
        sanitized = self._sanitize_config(merge_configs(self.config, config))
        thread_id = sanitized.get("configurable", {}).pop("thread_id", None)
        merged_headers = (
            _merge_tracing_headers(headers) if self.distributed_tracing else headers
        )
        sdk_thread = client.threads.stream(
            thread_id=thread_id,
            assistant_id=self.assistant_id,
            headers=merged_headers,
        )
        return _AsyncRemoteGraphRunStream(
            client=client,
            sdk_thread=sdk_thread,
            input=_translate_command_input(input),
            config=sanitized,
            metadata=kwargs.get("metadata"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST=tests/test_remote_graph_v3.py make test`
Expected: 43 passed.

- [ ] **Step 5: Run full langgraph unit tests to check for regressions**

Run from `libs/langgraph/`: `make test`
Expected: all existing tests pass; no regressions from the new `stream_events` override or `astream_events` rewrite.

- [ ] **Step 6: Commit**

```bash
git add libs/langgraph/langgraph/pregel/remote.py libs/langgraph/tests/test_remote_graph_v3.py
git commit -m "Add RemoteGraph.astream_events v3 dispatch"
```

---

## Task 10: Lint + format pass

**Files:**
- Touched in earlier tasks; no new files.

- [ ] **Step 1: Run formatter**

From `libs/langgraph/`: `make format`
Expected: no diff, or minor whitespace normalization.

- [ ] **Step 2: Run linter**

From `libs/langgraph/`: `make lint`
Expected: clean. If complaints, fix and re-run.

- [ ] **Step 3: Commit any formatter/linter changes**

```bash
git status
# if anything changed:
git add libs/langgraph/
git commit -m "Lint and format pass for RemoteGraph v3 streaming"
```

---

## Task 11: Integration tests against docker stack

**Files:**
- Create: `libs/sdk-py/tests/integration/test_remote_graph_v3.py`

These tests require the docker stack at `libs/sdk-py/integration/` running with `LANGSMITH_API_KEY` exported. They run behind the `-m integration` marker (excluded from default `make test`).

- [ ] **Step 1: Verify docker stack runs**

From `libs/sdk-py/integration/`:

```bash
source ~/.zshrc
docker compose up -d
```

Expected: postgres + redis + langgraph-api containers healthy. Check `docker compose ps`.

Then verify reachability: `curl -s http://localhost:2024/ok` → `{"ok":true}` or similar.

If the stack isn't running, follow the existing instructions in `libs/sdk-py/integration/README.md` or PR #7884's setup notes.

- [ ] **Step 2: Write the integration test file**

```python
# libs/sdk-py/tests/integration/test_remote_graph_v3.py
from __future__ import annotations

import asyncio

import pytest

from langgraph.pregel.remote import RemoteGraph

pytestmark = pytest.mark.integration

URL = "http://localhost:2024"


@pytest.fixture
def remote_agent() -> RemoteGraph:
    return RemoteGraph("agent", url=URL)


@pytest.fixture
def remote_tools_agent() -> RemoteGraph:
    return RemoteGraph("tools_agent", url=URL)


@pytest.mark.anyio
async def test_async_happy_path_yields_messages_and_terminates(remote_agent):
    async with remote_agent.astream_events(
        {"messages": [{"role": "user", "content": "hi"}]},
        version="v3",
    ) as stream:
        messages_seen = []
        async for chunk in stream.messages:
            messages_seen.append(chunk)
            if len(messages_seen) >= 1:
                break
        output = await stream.output
        assert await stream.interrupted is False
        assert output is not None


@pytest.mark.anyio
async def test_async_interrupt_path_surfaces_interrupts(remote_tools_agent):
    async with remote_tools_agent.astream_events(
        {"messages": [{"role": "user", "content": "interrupt please"}]},
        version="v3",
    ) as stream:
        async for _ in stream.events:
            if stream.interrupted:
                break
        assert await stream.interrupted is True
        interrupts = await stream.interrupts
        assert len(interrupts) >= 1


def test_sync_happy_path_yields_output(remote_agent):
    with remote_agent.stream_events(
        {"messages": [{"role": "user", "content": "hi"}]},
        version="v3",
    ) as stream:
        for _ in stream:
            pass
        assert stream.output is not None
        assert stream.interrupted is False


@pytest.mark.anyio
async def test_abort_mid_run_cancels_server_side(remote_agent):
    async with remote_agent.astream_events(
        {"messages": [{"role": "user", "content": "hi"}]},
        version="v3",
    ) as stream:
        # Pull the first event then abort.
        async for _ in stream.events:
            break
        await stream.abort()
    # Adapter is closed; if abort worked, no exceptions on exit.
```

These tests are structural — they verify the adapter is wired to real wire traffic without asserting exact event contents (which depend on the test graph's behavior in `libs/sdk-py/integration/graph/`).

- [ ] **Step 3: Run integration tests**

From `libs/sdk-py/`:

```bash
source ~/.zshrc
pytest tests/integration/test_remote_graph_v3.py -m integration -v
```

Expected: 4 passed. If failures, inspect the docker container logs (`docker compose logs api`) and adjust the test's input / projection assumptions to match `libs/sdk-py/integration/graph/streaming_graph.py` and `tools_agent.py`.

- [ ] **Step 4: Commit**

```bash
git add libs/sdk-py/tests/integration/test_remote_graph_v3.py
git commit -m "Add integration tests for RemoteGraph v3 streaming"
```

---

## Task 12: CI workflow path filter update

**Files:**
- Modify: `.github/workflows/_sdk_integration_test.yml`

- [ ] **Step 1: Read the current workflow**

Run: `cat .github/workflows/_sdk_integration_test.yml | head -30`

Locate the `on.pull_request.paths:` block (or equivalent). It currently lists `libs/sdk-py/**`.

- [ ] **Step 2: Extend the path filter**

Add two more entries to the paths list:

```yaml
on:
  pull_request:
    paths:
      - "libs/sdk-py/**"
      - "libs/langgraph/langgraph/pregel/remote.py"
      - "libs/langgraph/langgraph/pregel/_remote_run_stream.py"
```

Use the Edit tool to add exactly those two lines after the existing `libs/sdk-py/**` entry. Preserve existing indentation.

- [ ] **Step 3: Validate YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/_sdk_integration_test.yml'))" && echo "ok"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/_sdk_integration_test.yml
git commit -m "Extend SDK integration CI to trigger on RemoteGraph v3 changes"
```

---

## Task 13: Final verification

- [ ] **Step 1: Run all langgraph unit tests**

From `libs/langgraph/`: `make test`
Expected: all pass.

- [ ] **Step 2: Run lint + format checks one more time**

From `libs/langgraph/`: `make format && make lint`
Expected: clean, no diffs.

- [ ] **Step 3: Verify `make test` works in `libs/sdk-py/` too**

From `libs/sdk-py/`: `make test`
Expected: default suite passes; integration tests are deselected by default (they require `-m integration`).

- [ ] **Step 4: Manual smoke test**

From any working directory with the docker stack up:

```bash
source ~/.zshrc
python -c "
import asyncio
from langgraph.pregel.remote import RemoteGraph

async def main():
    rg = RemoteGraph('agent', url='http://localhost:2024')
    async with rg.astream_events(
        {'messages': [{'role': 'user', 'content': 'hi'}]},
        version='v3',
    ) as stream:
        async for chunk in stream.messages:
            print('msg:', type(chunk).__name__)
            break
        print('done. interrupted=', await stream.interrupted)

asyncio.run(main())
"
```

Expected: prints one `msg: ...` line, then `done. interrupted= False`. Confirms end-to-end wiring works.

- [ ] **Step 5: No commit required for verification.**

---

## Out of scope (future PRs)

- Bump `libs/langgraph/pyproject.toml` constraint `langgraph-sdk>=0.3.0,<0.4.0` → `>=0.4.0,<0.5.0`. Deferred until `langgraph-sdk 0.4.0` is published to PyPI; before then the editable workspace dep handles dev resolution.
- Real implementation of `astream_events(version="v1"|"v2")` (currently still raises `NotImplementedError`).
- Server-side support for `interrupt_before`, `interrupt_after`, `control=` on v3 runs.
- Monotonic server-side event stamps so `interleave()` has strict arrival ordering across channels.
- Sync `interleave()` via drainer threads.
- Resume-after-interrupt convenience on the adapter (callers currently do a fresh `astream_events(input=Command(resume=...))`).
