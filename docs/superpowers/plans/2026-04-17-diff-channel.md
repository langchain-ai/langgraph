# DiffChannel: Incremental Checkpoint Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `DiffChannel`, a new opt-in channel type that stores only per-step write deltas instead of the full accumulated list, reducing checkpoint storage from O(N²) to O(N) for append-style reducers like `add_messages`.

**Architecture:** A `DiffChannel` wraps a binary operator (e.g. `add_messages`) and returns a `DiffDelta` from `checkpoint()` instead of the full accumulated list. Savers detect the `"diff"` type tag, follow the `prev_version` chain via dict lookups (InMemorySaver) or a SQL range query (PostgresSaver), assemble a `DiffChainValue`, and pass it to `from_checkpoint` which replays deltas through the operator to reconstruct the full list. A new `after_checkpoint(version)` hook on `BaseChannel` allows `DiffChannel` to track its chain pointer without changing the saver public interface.

**Tech Stack:** Python 3.11+, ormsgpack, langchain-core messages, pytest/anyio, psycopg (Postgres tests)

**Branch:** `diff-channel-incremental-checkpointing`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `libs/checkpoint/langgraph/checkpoint/base/__init__.py` | Modify | Add `DiffDelta`, `DiffChainValue` dataclasses |
| `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py` | Modify | Add `"diff"` in `dumps_typed` + `loads_typed` |
| `libs/checkpoint/langgraph/checkpoint/memory/__init__.py` | Modify | Chain traversal in `_load_blobs` |
| `libs/checkpoint/tests/test_jsonplus.py` | Modify | Serde round-trip tests |
| `libs/checkpoint/tests/test_memory.py` | Modify | InMemorySaver diff-chain tests |
| `libs/langgraph/langgraph/channels/base.py` | Modify | Add no-op `after_checkpoint` |
| `libs/langgraph/langgraph/channels/diff.py` | Create | `DiffChannel` implementation |
| `libs/langgraph/langgraph/channels/__init__.py` | Modify | Export `DiffChannel` |
| `libs/langgraph/tests/test_channels.py` | Modify | `DiffChannel` unit tests |
| `libs/langgraph/tests/test_pregel.py` | Modify | End-to-end graph integration tests |
| `libs/langgraph/langgraph/pregel/_checkpoint.py` | Modify | Call `after_checkpoint` in `channels_from_checkpoint` |
| `libs/langgraph/langgraph/pregel/_loop.py` | Modify | Call `after_checkpoint` after `create_checkpoint` |
| `libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py` | Modify | Range-query chain reconstruction in `_load_blobs` |
| `libs/checkpoint-postgres/tests/test_postgres.py` | Modify | Postgres diff-chain integration tests |

---

## Task 1: Add Protocol Types `DiffDelta` and `DiffChainValue`

**Files:**
- Modify: `libs/checkpoint/langgraph/checkpoint/base/__init__.py`

These are the shared contract types between `DiffChannel` and savers. `DiffDelta` is what `channel.checkpoint()` returns; `DiffChainValue` is what savers pass to `channel.from_checkpoint()`.

- [ ] **Step 1: Open `libs/checkpoint/langgraph/checkpoint/base/__init__.py` and locate the `PendingWrite` line (currently around line 30)**

```python
PendingWrite = tuple[str, str, Any]
```

- [ ] **Step 2: Add `dataclasses` import and the two new types immediately after `PendingWrite`**

Add to the imports at the top of the file (after the existing imports block):
```python
import dataclasses
```

Then directly after `PendingWrite = tuple[str, str, Any]`:
```python
@dataclasses.dataclass
class DiffDelta:
    """Returned by DiffChannel.checkpoint(). Represents one step's writes."""

    delta: list[Any]
    prev_version: str | None  # version of previous diff blob; None = chain root


@dataclasses.dataclass
class DiffChainValue:
    """Passed to DiffChannel.from_checkpoint(). Assembled by saver _load_blobs()."""

    base: list[Any] | None  # starting accumulated value; None = start from empty
    deltas: list[list[Any]]  # per-step write-sets, ordered oldest → newest
```

- [ ] **Step 3: Add `DiffDelta` and `DiffChainValue` to the module's `__all__` (if one exists) or confirm they are importable**

Run:
```bash
cd /Users/sydney_runkle/oss/langgraph && python -c "from langgraph.checkpoint.base import DiffDelta, DiffChainValue; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add libs/checkpoint/langgraph/checkpoint/base/__init__.py
git commit -m "feat(checkpoint): add DiffDelta and DiffChainValue protocol types"
```

---

## Task 2: Extend Serde for `"diff"` Type Tag

**Files:**
- Modify: `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py`
- Modify: `libs/checkpoint/tests/test_jsonplus.py`

The serde must serialize `DiffDelta` as `("diff", bytes)` and deserialize `("diff", bytes)` back to `{"d": [...], "p": version_or_none}`. Savers check the `"diff"` type tag and call `serde.loads_typed` to decode — no direct `ormsgpack` import needed in savers.

- [ ] **Step 1: Write the failing serde tests in `libs/checkpoint/tests/test_jsonplus.py`**

Add at the end of the file:
```python
def test_diff_delta_serde_round_trip() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import DiffDelta

    serde = JsonPlusSerializer()
    prev = "00000000000000000000000000000001.1234567890123456"
    delta = DiffDelta(
        delta=[HumanMessage(content="hello", id="msg-1")],
        prev_version=prev,
    )
    type_tag, blob = serde.dumps_typed(delta)
    assert type_tag == "diff"

    result = serde.loads_typed(("diff", blob))
    assert isinstance(result, dict)
    assert result["p"] == prev
    assert len(result["d"]) == 1
    assert result["d"][0].content == "hello"


def test_diff_delta_serde_root_blob() -> None:
    from langgraph.checkpoint.base import DiffDelta

    serde = JsonPlusSerializer()
    delta = DiffDelta(delta=[], prev_version=None)
    type_tag, blob = serde.dumps_typed(delta)
    assert type_tag == "diff"

    result = serde.loads_typed(("diff", blob))
    assert result["p"] is None
    assert result["d"] == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd libs/checkpoint && TEST=tests/test_jsonplus.py::test_diff_delta_serde_round_trip make test
```
Expected: `FAILED` — `NotImplementedError: Unknown serialization type: diff` (or similar)

- [ ] **Step 3: Add the `"diff"` branch to `dumps_typed` in `jsonplus.py`**

In `JsonPlusSerializer.dumps_typed`, locate this block (around line 235):
```python
def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
    if obj is None:
        return "null", EMPTY_BYTES
    elif isinstance(obj, bytes):
        return "bytes", obj
    elif isinstance(obj, bytearray):
        return "bytearray", obj
    else:
        try:
            return "msgpack", _msgpack_enc(obj)
```

Add the `DiffDelta` branch **before** the `else` block:
```python
def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
    if obj is None:
        return "null", EMPTY_BYTES
    elif isinstance(obj, bytes):
        return "bytes", obj
    elif isinstance(obj, bytearray):
        return "bytearray", obj
    elif isinstance(obj, DiffDelta):
        return "diff", _msgpack_enc({"d": obj.delta, "p": obj.prev_version})
    else:
        try:
            return "msgpack", _msgpack_enc(obj)
```

Add the import of `DiffDelta` at the top of `jsonplus.py` (alongside existing imports):
```python
from langgraph.checkpoint.base import DiffDelta
```

- [ ] **Step 4: Add the `"diff"` branch to `loads_typed` in `jsonplus.py`**

In `JsonPlusSerializer.loads_typed`, locate the dispatch (around line 250):
```python
def loads_typed(self, data: tuple[str, bytes]) -> Any:
    type_, data_ = data
    if type_ == "null":
        return None
    elif type_ == "bytes":
        return data_
    elif type_ == "bytearray":
        return bytearray(data_)
    elif type_ == "json":
        return json.loads(data_, object_hook=self._reviver)
    elif type_ == "msgpack":
        return ormsgpack.unpackb(
            data_, ext_hook=self._unpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
        )
    elif self.pickle_fallback and type_ == "pickle":
        return pickle.loads(data_)
    else:
        raise NotImplementedError(f"Unknown serialization type: {type_}")
```

Add the `"diff"` branch **before** the `else` raise:
```python
    elif type_ == "diff":
        return ormsgpack.unpackb(
            data_, ext_hook=self._unpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
        )
    elif self.pickle_fallback and type_ == "pickle":
        return pickle.loads(data_)
    else:
        raise NotImplementedError(f"Unknown serialization type: {type_}")
```

- [ ] **Step 5: Run the serde tests to confirm they pass**

```bash
cd libs/checkpoint && TEST=tests/test_jsonplus.py::test_diff_delta_serde_round_trip\ tests/test_jsonplus.py::test_diff_delta_serde_root_blob make test
```
Expected: both `PASSED`

- [ ] **Step 6: Run the full serde test suite to check for regressions**

```bash
cd libs/checkpoint && TEST=tests/test_jsonplus.py make test
```
Expected: all existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py libs/checkpoint/tests/test_jsonplus.py
git commit -m "feat(checkpoint/serde): serialize DiffDelta as 'diff' type tag"
```

---

## Task 3: Add `after_checkpoint` Hook to `BaseChannel`

**Files:**
- Modify: `libs/langgraph/langgraph/channels/base.py`

This is a no-op default method. All existing channels inherit it silently. `DiffChannel` will override it to advance `_base_version` and clear `_pending`.

- [ ] **Step 1: Open `libs/langgraph/langgraph/channels/base.py` and locate the `finish` method (currently the last method, around line 112)**

```python
def finish(self) -> bool:
    ...
    return False
```

- [ ] **Step 2: Add `after_checkpoint` after `finish`**

```python
    def after_checkpoint(self, version: Any) -> None:
        """Called after checkpoint() with the assigned version, and after
        from_checkpoint() with the current channel version.

        No-op by default. Override in channels that track their own version
        for incremental checkpointing (e.g. DiffChannel).
        """
        pass
```

- [ ] **Step 3: Verify the method is accessible on existing channel types**

```bash
cd libs/langgraph && python -c "
from langgraph.channels.last_value import LastValue
from langgraph.channels.binop import BinaryOperatorAggregate
import operator
ch = LastValue(int).from_checkpoint(3)
ch.after_checkpoint('v1')  # must not raise
ch2 = BinaryOperatorAggregate(int, operator.add).from_checkpoint(0)
ch2.after_checkpoint('v2')  # must not raise
print('ok')
"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add libs/langgraph/langgraph/channels/base.py
git commit -m "feat(channels): add no-op after_checkpoint hook to BaseChannel"
```

---

## Task 4: Implement `DiffChannel`

**Files:**
- Create: `libs/langgraph/langgraph/channels/diff.py`
- Modify: `libs/langgraph/langgraph/channels/__init__.py`
- Modify: `libs/langgraph/tests/test_channels.py`

`DiffChannel` wraps a binary operator, accumulates incoming writes in `_pending`, and returns a `DiffDelta` from `checkpoint()`. `from_checkpoint` accepts a `DiffChainValue` and replays write-sets through the operator.

- [ ] **Step 1: Write failing unit tests in `libs/langgraph/tests/test_channels.py`**

Add at the end of the file:
```python
def test_diff_channel_basic_two_steps() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.channels.diff import DiffChannel
    from langgraph.checkpoint.base import DiffDelta
    from langgraph.graph.message import add_messages

    ch = DiffChannel(add_messages).from_checkpoint(MISSING)
    ch.after_checkpoint(None)

    # Step 1: one message added
    ch.update([HumanMessage(content="hi", id="h1")])
    d1 = ch.checkpoint()
    assert isinstance(d1, DiffDelta)
    assert len(d1.delta) == 1
    assert d1.prev_version is None  # first ever step
    ch.after_checkpoint("v1")

    # Step 2: another message
    ch.update([AIMessage(content="hello", id="a1")])
    d2 = ch.checkpoint()
    assert d2.prev_version == "v1"
    assert len(d2.delta) == 1
    ch.after_checkpoint("v2")

    # Full accumulated value is preserved in memory
    assert len(ch.get()) == 2
    assert ch.get()[0].content == "hi"
    assert ch.get()[1].content == "hello"


def test_diff_channel_after_checkpoint_no_op_when_unchanged() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.channels.diff import DiffChannel
    from langgraph.graph.message import add_messages

    ch = DiffChannel(add_messages).from_checkpoint(MISSING)
    ch.after_checkpoint(None)
    ch.update([HumanMessage(content="hi", id="h1")])
    ch.after_checkpoint("v1")

    # Same version: no-op
    ch.after_checkpoint("v1")
    assert ch._base_version == "v1"
    assert ch._pending == []


def test_diff_channel_from_checkpoint_chain() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.channels.diff import DiffChannel
    from langgraph.checkpoint.base import DiffChainValue
    from langgraph.graph.message import add_messages

    spec = DiffChannel(add_messages)
    chain = DiffChainValue(
        base=None,
        deltas=[
            [HumanMessage(content="hi", id="h1")],
            [AIMessage(content="hello", id="a1")],
            [HumanMessage(content="bye", id="h2")],
        ],
    )
    ch = spec.from_checkpoint(chain)
    msgs = ch.get()
    assert len(msgs) == 3
    assert msgs[0].content == "hi"
    assert msgs[1].content == "hello"
    assert msgs[2].content == "bye"


def test_diff_channel_from_checkpoint_backwards_compat() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.channels.diff import DiffChannel
    from langgraph.graph.message import add_messages

    # Old BinaryOperatorAggregate checkpoint: plain list
    spec = DiffChannel(add_messages)
    old_value = [HumanMessage(content="old", id="h1")]
    ch = spec.from_checkpoint(old_value)
    assert ch.get() == old_value


def test_diff_channel_overwrite_resets_chain() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.channels.diff import DiffChannel
    from langgraph.checkpoint.base import DiffDelta
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    ch = DiffChannel(add_messages).from_checkpoint(MISSING)
    ch.after_checkpoint(None)
    ch.update([HumanMessage(content="old", id="h1")])
    ch.after_checkpoint("v1")

    # Overwrite should create a root blob (prev_version=None)
    ch.update([Overwrite([HumanMessage(content="new", id="h2")])])
    d = ch.checkpoint()
    assert isinstance(d, DiffDelta)
    assert d.prev_version is None  # chain root
    assert len(d.delta) == 1
    assert d.delta[0].content == "new"


def test_diff_channel_unsupported_saver_raises() -> None:
    from langgraph.channels.diff import DiffChannel
    from langgraph.checkpoint.base import DiffDelta
    from langgraph.graph.message import add_messages

    # If a saver returns a raw DiffDelta (unsupported), from_checkpoint raises
    spec = DiffChannel(add_messages)
    raw_delta = DiffDelta(delta=[], prev_version=None)
    with pytest.raises(ValueError, match="DiffChannel received a raw DiffDelta"):
        spec.from_checkpoint(raw_delta)
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd libs/langgraph && TEST=tests/test_channels.py::test_diff_channel_basic_two_steps make test
```
Expected: `FAILED` — `ImportError: cannot import name 'DiffChannel'`

- [ ] **Step 3: Create `libs/langgraph/langgraph/channels/diff.py`**

```python
from __future__ import annotations

import collections.abc
from collections.abc import Callable, Sequence
from typing import Any, Generic

from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.channels.binop import _get_overwrite, _strip_extras
from langgraph.checkpoint.base import DiffChainValue, DiffDelta
from langgraph.errors import EmptyChannelError

__all__ = ("DiffChannel",)


class DiffChannel(Generic[Value], BaseChannel[list[Value], Value, DiffDelta]):
    """A channel that stores only per-step write deltas in checkpoints.

    Reconstructs the full accumulated list at load time by replaying the
    chain of deltas through the operator. Use with append-style reducers
    (e.g. ``add_messages``) on long-running threads to reduce checkpoint
    storage from O(N²) to O(N).

    Requires InMemorySaver or PostgresSaver; SqliteSaver is not supported.

    Usage::

        class State(TypedDict):
            messages: Annotated[list[AnyMessage], DiffChannel(add_messages)]
    """

    __slots__ = ("value", "operator", "_pending", "_base_version", "_overwritten")

    def __init__(
        self,
        operator: Callable[[list[Value], Any], list[Value]],
        typ: type = list,
    ) -> None:
        typ = _strip_extras(typ)
        if typ in (
            collections.abc.Sequence,
            collections.abc.MutableSequence,
        ):
            typ = list
        super().__init__(typ)
        self.operator = operator
        try:
            self.value: list[Value] = typ()
        except Exception:
            self.value = []
        self._pending: list[Any] = []
        self._base_version: str | None = None
        self._overwritten: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiffChannel):
            return False
        if (
            self.operator.__name__ != "<lambda>"
            and other.operator.__name__ != "<lambda>"
        ):
            return self.operator is other.operator
        return True

    @property
    def ValueType(self) -> Any:
        return list[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Any:
        return self.typ | list[self.typ]  # type: ignore[name-defined]

    def copy(self) -> Self:
        new = DiffChannel(self.operator, self.typ)
        new.key = self.key
        new.value = self.value[:]
        new._pending = self._pending[:]
        new._base_version = self._base_version
        new._overwritten = self._overwritten
        return new

    def from_checkpoint(self, checkpoint: Any) -> Self:
        new = DiffChannel(self.operator, self.typ)
        new.key = self.key
        if checkpoint is MISSING:
            new.value = []
        elif isinstance(checkpoint, DiffChainValue):
            accumulated: list[Value] = list(checkpoint.base) if checkpoint.base else []
            for step_writes in checkpoint.deltas:
                for write in step_writes:
                    accumulated = new.operator(accumulated, write)
            new.value = accumulated
        elif isinstance(checkpoint, DiffDelta):
            raise ValueError(
                "DiffChannel received a raw DiffDelta from the checkpoint saver. "
                "Your saver does not support incremental channel storage. "
                "Use InMemorySaver or PostgresSaver."
            )
        else:
            # Backwards compat: plain list from old BinaryOperatorAggregate checkpoint.
            new.value = list(checkpoint)
        new._pending = []
        new._base_version = None  # set by the subsequent after_checkpoint() call
        new._overwritten = False
        return new

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        seen_overwrite = False
        for value in values:
            is_overwrite, overwrite_value = _get_overwrite(value)
            if is_overwrite:
                if seen_overwrite:
                    from langgraph.errors import ErrorCode, InvalidUpdateError, create_error_message
                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
                self.value = list(overwrite_value) if overwrite_value is not None else []
                self._pending = list(self.value)
                self._overwritten = True
                seen_overwrite = True
            elif not seen_overwrite:
                self.value = self.operator(self.value, value)
                self._pending.append(value)
        return True

    def get(self) -> list[Value]:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING and self.value is not None

    def checkpoint(self) -> DiffDelta:
        return DiffDelta(
            delta=self._pending[:],
            prev_version=None if self._overwritten else self._base_version,
        )

    def after_checkpoint(self, version: Any) -> None:
        if version != self._base_version:
            self._base_version = version
            self._pending = []
            self._overwritten = False
```

- [ ] **Step 4: Export `DiffChannel` from `libs/langgraph/langgraph/channels/__init__.py`**

Open `libs/langgraph/langgraph/channels/__init__.py` and add `DiffChannel` to the imports and `__all__`. The file currently exports `BinaryOperatorAggregate`, `EphemeralValue`, `LastValue`, `Topic`, `UntrackedValue`. Add:

```python
from langgraph.channels.diff import DiffChannel
```

And add `"DiffChannel"` to `__all__` if present.

- [ ] **Step 5: Run all DiffChannel unit tests**

```bash
cd libs/langgraph && TEST="tests/test_channels.py::test_diff_channel_basic_two_steps tests/test_channels.py::test_diff_channel_after_checkpoint_no_op_when_unchanged tests/test_channels.py::test_diff_channel_from_checkpoint_chain tests/test_channels.py::test_diff_channel_from_checkpoint_backwards_compat tests/test_channels.py::test_diff_channel_overwrite_resets_chain tests/test_channels.py::test_diff_channel_unsupported_saver_raises" make test
```
Expected: all 6 `PASSED`

- [ ] **Step 6: Run full channel test suite for regressions**

```bash
cd libs/langgraph && TEST=tests/test_channels.py make test
```
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add libs/langgraph/langgraph/channels/diff.py libs/langgraph/langgraph/channels/__init__.py libs/langgraph/tests/test_channels.py
git commit -m "feat(channels): implement DiffChannel for incremental checkpoint storage"
```

---

## Task 5: Extend `InMemorySaver._load_blobs` for Chain Traversal

**Files:**
- Modify: `libs/checkpoint/langgraph/checkpoint/memory/__init__.py`
- Modify: `libs/checkpoint/tests/test_memory.py`

When `_load_blobs` encounters a blob with type `"diff"`, it follows the `prev_version` chain backwards through `self.blobs`, collects all deltas, and returns a `DiffChainValue` instead of a plain deserialized value.

- [ ] **Step 1: Write failing integration tests in `libs/checkpoint/tests/test_memory.py`**

Add at the end of the file (after the `TestMemorySaver` class):

```python
class TestInMemorySaverDiffChannel:
    def test_diff_channel_chain_reconstruction(self) -> None:
        """_load_blobs follows the diff chain and returns DiffChainValue."""
        from langgraph.checkpoint.base import DiffChainValue, DiffDelta

        saver = InMemorySaver()
        serde = JsonPlusSerializer()

        thread_id = "t1"
        ns = ""

        # Simulate two steps: v1 (root) and v2 (chained to v1)
        v1 = "00000000000000000000000000000001.1234567890000000"
        v2 = "00000000000000000000000000000002.1234567890000000"

        delta1 = DiffDelta(delta=["msg1"], prev_version=None)
        delta2 = DiffDelta(delta=["msg2"], prev_version=v1)

        saver.blobs[(thread_id, ns, "messages", v1)] = serde.dumps_typed(delta1)
        saver.blobs[(thread_id, ns, "messages", v2)] = serde.dumps_typed(delta2)

        channel_values = saver._load_blobs(thread_id, ns, {"messages": v2})

        assert "messages" in channel_values
        result = channel_values["messages"]
        assert isinstance(result, DiffChainValue)
        assert result.base is None
        assert result.deltas == [["msg1"], ["msg2"]]

    def test_diff_channel_mixed_old_and_new_blobs(self) -> None:
        """When chain hits an old non-diff blob, it becomes base."""
        from langgraph.checkpoint.base import DiffChainValue, DiffDelta

        saver = InMemorySaver()
        serde = JsonPlusSerializer()

        thread_id = "t2"
        ns = ""

        v_old = "00000000000000000000000000000001.0000000000000000"
        v_new = "00000000000000000000000000000002.0000000000000000"

        # Old-style full-list blob
        saver.blobs[(thread_id, ns, "messages", v_old)] = serde.dumps_typed(["old_msg"])
        # New diff blob chained to old
        delta = DiffDelta(delta=["new_msg"], prev_version=v_old)
        saver.blobs[(thread_id, ns, "messages", v_new)] = serde.dumps_typed(delta)

        channel_values = saver._load_blobs(thread_id, ns, {"messages": v_new})
        result = channel_values["messages"]
        assert isinstance(result, DiffChainValue)
        assert result.base == ["old_msg"]
        assert result.deltas == [["new_msg"]]
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd libs/checkpoint && TEST=tests/test_memory.py::TestInMemorySaverDiffChannel make test
```
Expected: `FAILED` — diff channel values not wrapped in `DiffChainValue`

- [ ] **Step 3: Update `_load_blobs` in `libs/checkpoint/langgraph/checkpoint/memory/__init__.py`**

Locate `_load_blobs` (around line 123):
```python
def _load_blobs(
    self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
) -> dict[str, Any]:
    channel_values: dict[str, Any] = {}
    for k, v in versions.items():
        kk = (thread_id, checkpoint_ns, k, v)
        if kk in self.blobs:
            vv = self.blobs[kk]
            if vv[0] != "empty":
                channel_values[k] = self.serde.loads_typed(vv)
    return channel_values
```

Replace with:
```python
def _load_blobs(
    self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
) -> dict[str, Any]:
    from langgraph.checkpoint.base import DiffChainValue

    channel_values: dict[str, Any] = {}
    diff_channels: dict[str, Any] = {}

    for k, v in versions.items():
        kk = (thread_id, checkpoint_ns, k, v)
        if kk not in self.blobs:
            continue
        vv = self.blobs[kk]
        if vv[0] == "diff":
            diff_channels[k] = v
        elif vv[0] != "empty":
            channel_values[k] = self.serde.loads_typed(vv)

    for k, current_version in diff_channels.items():
        chain_deltas: list[list[Any]] = []
        base: list[Any] | None = None
        version: str | None = current_version
        while version is not None:
            kk = (thread_id, checkpoint_ns, k, version)
            if kk not in self.blobs:
                break
            vv = self.blobs[kk]
            if vv[0] == "diff":
                payload = self.serde.loads_typed(vv)  # {"d": [...], "p": version|None}
                chain_deltas.append(payload["d"])
                version = payload["p"]
            else:
                base = self.serde.loads_typed(vv)
                break
        chain_deltas.reverse()
        channel_values[k] = DiffChainValue(base=base, deltas=chain_deltas)

    return channel_values
```

- [ ] **Step 4: Run the new tests**

```bash
cd libs/checkpoint && TEST=tests/test_memory.py::TestInMemorySaverDiffChannel make test
```
Expected: both `PASSED`

- [ ] **Step 5: Run the full memory test suite for regressions**

```bash
cd libs/checkpoint && TEST=tests/test_memory.py make test
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add libs/checkpoint/langgraph/checkpoint/memory/__init__.py libs/checkpoint/tests/test_memory.py
git commit -m "feat(checkpoint/memory): chain-traverse diff blobs in _load_blobs"
```

---

## Task 6: Update Pregel Layer (`channels_from_checkpoint` + `_put_checkpoint`)

**Files:**
- Modify: `libs/langgraph/langgraph/pregel/_checkpoint.py`
- Modify: `libs/langgraph/langgraph/pregel/_loop.py`
- Modify: `libs/langgraph/tests/test_pregel.py`

Two small changes: call `channel.after_checkpoint(version)` after `from_checkpoint` (so `DiffChannel` knows its starting version), and after `create_checkpoint` (so `DiffChannel` clears `_pending` and advances `_base_version`).

- [ ] **Step 1: Write the failing end-to-end integration test in `libs/langgraph/tests/test_pregel.py`**

Find the end of the test file and add:

```python
async def test_diff_channel_end_to_end_inmemory() -> None:
    """Full graph run: DiffChannel accumulates correctly across multiple turns."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.channels.diff import DiffChannel
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, DiffChannel(add_messages)]

    def respond(state: State) -> dict:
        n = len(state["messages"])
        return {"messages": [AIMessage(content=f"reply-{n}", id=f"ai-{n}")]}

    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "diff-test-1"}}

    # Turn 1
    graph.invoke({"messages": [HumanMessage(content="hello", id="h1")]}, config)
    # Turn 2
    graph.invoke({"messages": [HumanMessage(content="world", id="h2")]}, config)
    # Turn 3
    graph.invoke({"messages": [HumanMessage(content="bye", id="h3")]}, config)

    state = graph.get_state(config)
    msgs = state.values["messages"]
    # 3 human + 3 AI = 6 total
    assert len(msgs) == 6, f"expected 6 messages, got {len(msgs)}: {msgs}"
    assert msgs[0].content == "hello"
    assert msgs[2].content == "world"
    assert msgs[4].content == "bye"


async def test_diff_channel_time_travel() -> None:
    """Time-travel to an earlier checkpoint reconstructs the correct partial history."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.channels.diff import DiffChannel
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, DiffChannel(add_messages)]

    counter = {"n": 0}

    def respond(state: State) -> dict:
        counter["n"] += 1
        return {"messages": [AIMessage(content=f"ai-{counter['n']}", id=f"ai-{counter['n']}")]}

    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "diff-time-travel"}}

    # Run 2 turns
    graph.invoke({"messages": [HumanMessage(content="h1", id="h1")]}, config)
    graph.invoke({"messages": [HumanMessage(content="h2", id="h2")]}, config)

    # Collect checkpoint history
    history = list(graph.get_state_history(config))
    # history[0] = latest; find checkpoint after first turn (4 msgs: input + respond + input + respond)
    # We want the state after first turn = 2 messages
    after_turn1 = next(h for h in history if len(h.values.get("messages", [])) == 2)

    assert len(after_turn1.values["messages"]) == 2
    assert after_turn1.values["messages"][0].content == "h1"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd libs/langgraph && TEST="tests/test_pregel.py::test_diff_channel_end_to_end_inmemory tests/test_pregel.py::test_diff_channel_time_travel" make test
```
Expected: `FAILED` — `DiffChannel` does not receive `after_checkpoint` so `_base_version` is never set, causing each step to emit a root blob and reconstruction only returns the last step's messages.

- [ ] **Step 3: Update `channels_from_checkpoint` in `libs/langgraph/langgraph/pregel/_checkpoint.py`**

Locate `channels_from_checkpoint` (around line 58):
```python
    return (
        {
            k: v.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
            for k, v in channel_specs.items()
        },
        managed_specs,
    )
```

Replace the return statement with:
```python
    channels: dict[str, BaseChannel] = {}
    for k, v in channel_specs.items():
        ch = v.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
        ch.after_checkpoint(checkpoint["channel_versions"].get(k))
        channels[k] = ch
    return channels, managed_specs
```

- [ ] **Step 4: Update `_put_checkpoint` in `libs/langgraph/langgraph/pregel/_loop.py`**

Find `_put_checkpoint` and locate where `self.checkpoint` is assigned from `create_checkpoint`. The relevant block (around line 877) is:

```python
self.checkpoint = create_checkpoint(
    self.checkpoint,
    self.channels if do_checkpoint else None,
    self.step,
    id=self.checkpoint["id"] if exiting else None,
    updated_channels=self.updated_channels,
)
```

Immediately after that assignment, add the `after_checkpoint` notification for all channels that were actually checkpointed (`do_checkpoint and self.channels is not None`):

```python
self.checkpoint = create_checkpoint(
    self.checkpoint,
    self.channels if do_checkpoint else None,
    self.step,
    id=self.checkpoint["id"] if exiting else None,
    updated_channels=self.updated_channels,
)
if do_checkpoint and self.channels:
    for k, ch in self.channels.items():
        ch.after_checkpoint(self.checkpoint["channel_versions"].get(k))
```

- [ ] **Step 5: Run the integration tests**

```bash
cd libs/langgraph && TEST="tests/test_pregel.py::test_diff_channel_end_to_end_inmemory tests/test_pregel.py::test_diff_channel_time_travel" make test
```
Expected: both `PASSED`

- [ ] **Step 6: Run the full pregel test suite for regressions (this is a large suite — may take several minutes)**

```bash
cd libs/langgraph && make test
```
Expected: all existing tests pass

- [ ] **Step 7: Commit**

```bash
git add libs/langgraph/langgraph/pregel/_checkpoint.py libs/langgraph/langgraph/pregel/_loop.py libs/langgraph/tests/test_pregel.py
git commit -m "feat(pregel): call after_checkpoint hook when loading and saving channels"
```

---

## Task 7: Extend `PostgresSaver._load_blobs` for Range-Query Chain Reconstruction

**Files:**
- Modify: `libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py`
- Modify: `libs/checkpoint-postgres/tests/test_postgres.py` (or equivalent test file)

After the existing JOIN fetches one blob per channel, detect channels with `type = "diff"` and issue one additional SQL range query per diff channel (typically just `messages`) to retrieve the full chain.

- [ ] **Step 1: Find the Postgres test file**

```bash
ls libs/checkpoint-postgres/tests/
```

Use whatever test file exists (likely `test_postgres.py` or `test_async_postgres.py`).

- [ ] **Step 2: Write failing Postgres diff-chain tests**

These tests require a live Postgres instance. Add to the existing test class/file (check how the Postgres fixture is set up in the existing tests and reuse it):

```python
async def test_diff_channel_postgres_chain_reconstruction(postgres_url: str) -> None:
    """PostgresSaver reconstructs DiffChannel chain via range query."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.channels.diff import DiffChannel
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, DiffChannel(add_messages)]

    def respond(state: State) -> dict:
        n = len(state["messages"])
        return {"messages": [AIMessage(content=f"reply-{n}", id=f"ai-{n}")]}

    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")

    async with AsyncPostgresSaver.from_conn_string(postgres_url) as saver:
        await saver.setup()
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "pg-diff-test-1"}}

        await graph.ainvoke(
            {"messages": [HumanMessage(content="hi", id="h1")]}, config
        )
        await graph.ainvoke(
            {"messages": [HumanMessage(content="there", id="h2")]}, config
        )

        state = await graph.aget_state(config)
        msgs = state.values["messages"]
        assert len(msgs) == 4, f"expected 4, got {len(msgs)}"
        assert msgs[0].content == "hi"
        assert msgs[2].content == "there"
```

- [ ] **Step 3: Run to confirm test fails**

```bash
cd libs/checkpoint-postgres && TEST=tests/test_postgres.py::test_diff_channel_postgres_chain_reconstruction make test
```
Expected: `FAILED` — diff chain not assembled; only last delta returned

- [ ] **Step 4: Update `_load_blobs` in `base.py` to accept context kwargs and detect diff channels**

Locate `_load_blobs` in `libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py` (around line 187):
```python
def _load_blobs(
    self, blob_values: list[tuple[bytes, bytes, bytes]]
) -> dict[str, Any]:
    if not blob_values:
        return {}
    return {
        k.decode(): self.serde.loads_typed((t.decode(), v))
        for k, t, v in blob_values
        if t.decode() != "empty"
    }
```

Replace with:
```python
def _load_blobs(
    self,
    blob_values: list[tuple[bytes, bytes, bytes]],
    *,
    thread_id: str = "",
    checkpoint_ns: str = "",
) -> dict[str, Any]:
    from langgraph.checkpoint.base import DiffChainValue

    if not blob_values:
        return {}

    result: dict[str, Any] = {}
    diff_channel_payloads: dict[str, dict[str, Any]] = {}

    for k, t, v in blob_values:
        channel = k.decode()
        type_tag = t.decode()
        if type_tag == "diff":
            diff_channel_payloads[channel] = self.serde.loads_typed((type_tag, v))
        elif type_tag != "empty":
            result[channel] = self.serde.loads_typed((type_tag, v))

    if diff_channel_payloads:
        result.update(
            self._load_diff_chains(thread_id, checkpoint_ns, diff_channel_payloads)
        )

    return result

def _load_diff_chains(
    self,
    thread_id: str,
    checkpoint_ns: str,
    diff_channel_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Override in sync/async subclasses. Resolves diff-chain blobs to DiffChainValue."""
    raise NotImplementedError
```

- [ ] **Step 5: Override `_load_diff_chains` in sync `PostgresSaver` and update `_load_checkpoint_tuple` in both subclasses**

**5a — Sync `PostgresSaver`** (`libs/checkpoint-postgres/langgraph/checkpoint/postgres/__init__.py`):

Add this method to the `PostgresSaver` class (after `_load_checkpoint_tuple`):
```python
def _load_diff_chains(
    self,
    thread_id: str,
    checkpoint_ns: str,
    diff_channel_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from langgraph.checkpoint.base import DiffChainValue

    result: dict[str, Any] = {}
    with self._cursor() as cur:
        for channel, current_payload in diff_channel_payloads.items():
            # Walk the prev_version chain backwards collecting payloads.
            payloads: list[dict[str, Any]] = [current_payload]
            version_cursor: str | None = current_payload["p"]
            base: list[Any] | None = None

            while version_cursor is not None:
                cur.execute(
                    "SELECT type, blob FROM checkpoint_blobs "
                    "WHERE thread_id = %s AND checkpoint_ns = %s "
                    "AND channel = %s AND version = %s",
                    (thread_id, checkpoint_ns, channel, version_cursor),
                )
                row = cur.fetchone()
                if row is None:
                    break
                # row is a dict (dict_row factory): {"type": str, "blob": bytes}
                if row["type"] == "diff":
                    payload = self.serde.loads_typed(("diff", row["blob"]))
                    payloads.append(payload)
                    version_cursor = payload["p"]
                else:
                    base = self.serde.loads_typed((row["type"], row["blob"]))
                    break

            # payloads is newest→oldest; reverse for oldest→newest deltas
            payloads.reverse()
            result[channel] = DiffChainValue(
                base=base, deltas=[p["d"] for p in payloads]
            )
    return result
```

Update `_load_checkpoint_tuple` in sync `PostgresSaver` to pass `thread_id` and `checkpoint_ns`:
```python
def _load_checkpoint_tuple(self, value: DictRow) -> CheckpointTuple:
    return CheckpointTuple(
        {
            "configurable": {
                "thread_id": value["thread_id"],
                "checkpoint_ns": value["checkpoint_ns"],
                "checkpoint_id": value["checkpoint_id"],
            }
        },
        {
            **value["checkpoint"],
            "channel_values": {
                **(value["checkpoint"].get("channel_values") or {}),
                **self._load_blobs(
                    value["channel_values"],
                    thread_id=value["thread_id"],
                    checkpoint_ns=value["checkpoint_ns"],
                ),
            },
        },
        value["metadata"],
        (
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["parent_checkpoint_id"],
                }
            }
            if value["parent_checkpoint_id"]
            else None
        ),
        self._load_writes(value["pending_writes"]),
    )
```

**5b — Async `AsyncPostgresSaver`** (`libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py`):

Since `_load_checkpoint_tuple` is already `async`, override it to call an async version of the chain loader instead of `_load_blobs`. Add `_load_diff_chains_async` and update `_load_checkpoint_tuple`:

```python
async def _load_diff_chains_async(
    self,
    thread_id: str,
    checkpoint_ns: str,
    diff_channel_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from langgraph.checkpoint.base import DiffChainValue

    result: dict[str, Any] = {}
    async with self._cursor() as cur:
        for channel, current_payload in diff_channel_payloads.items():
            payloads: list[dict[str, Any]] = [current_payload]
            version_cursor: str | None = current_payload["p"]
            base: list[Any] | None = None

            while version_cursor is not None:
                await cur.execute(
                    "SELECT type, blob FROM checkpoint_blobs "
                    "WHERE thread_id = %s AND checkpoint_ns = %s "
                    "AND channel = %s AND version = %s",
                    (thread_id, checkpoint_ns, channel, version_cursor),
                )
                row = await cur.fetchone()
                if row is None:
                    break
                if row["type"] == "diff":
                    payload = self.serde.loads_typed(("diff", row["blob"]))
                    payloads.append(payload)
                    version_cursor = payload["p"]
                else:
                    base = self.serde.loads_typed((row["type"], row["blob"]))
                    break

            payloads.reverse()
            result[channel] = DiffChainValue(
                base=base, deltas=[p["d"] for p in payloads]
            )
    return result

async def _load_checkpoint_tuple(self, value: DictRow) -> CheckpointTuple:
    thread_id = value["thread_id"]
    checkpoint_ns = value["checkpoint_ns"]
    blob_values = value["channel_values"]

    # Load non-diff channels synchronously using the base _load_blobs,
    # but intercept diff channels for async resolution below.
    from langgraph.checkpoint.base import DiffChainValue
    non_diff: dict[str, Any] = {}
    diff_payloads: dict[str, dict[str, Any]] = {}
    if blob_values:
        for k, t, v in blob_values:
            channel = k.decode()
            type_tag = t.decode()
            if type_tag == "diff":
                diff_payloads[channel] = self.serde.loads_typed((type_tag, v))
            elif type_tag != "empty":
                non_diff[channel] = self.serde.loads_typed((type_tag, v))

    diff_values = (
        await self._load_diff_chains_async(thread_id, checkpoint_ns, diff_payloads)
        if diff_payloads
        else {}
    )

    return CheckpointTuple(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": value["checkpoint_id"],
            }
        },
        {
            **value["checkpoint"],
            "channel_values": {
                **(value["checkpoint"].get("channel_values") or {}),
                **non_diff,
                **diff_values,
            },
        },
        value["metadata"],
        (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": value["parent_checkpoint_id"],
                }
            }
            if value["parent_checkpoint_id"]
            else None
        ),
        await asyncio.to_thread(self._load_writes, value["pending_writes"]),
    )
```

- [ ] **Step 6: Run Postgres diff-channel test**

```bash
cd libs/checkpoint-postgres && TEST=tests/test_postgres.py::test_diff_channel_postgres_chain_reconstruction make test
```
Expected: `PASSED`

- [ ] **Step 7: Run full Postgres test suite for regressions**

```bash
cd libs/checkpoint-postgres && make test
```
Expected: all existing tests pass

- [ ] **Step 8: Commit**

```bash
git add libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py libs/checkpoint-postgres/tests/test_postgres.py
git commit -m "feat(checkpoint/postgres): range-query diff chain reconstruction in _load_blobs"
```

---

## Task 8: Format, Lint, and Final Integration Check

**Files:** All modified libraries

- [ ] **Step 1: Format and lint `libs/checkpoint`**

```bash
cd libs/checkpoint && make format && make lint
```
Fix any issues, then re-run until clean.

- [ ] **Step 2: Format and lint `libs/langgraph`**

```bash
cd libs/langgraph && make format && make lint
```
Fix any issues.

- [ ] **Step 3: Format and lint `libs/checkpoint-postgres`**

```bash
cd libs/checkpoint-postgres && make format && make lint
```
Fix any issues.

- [ ] **Step 4: Run all tests across affected libraries**

```bash
cd libs/checkpoint && make test
cd libs/langgraph && make test
```
Expected: all green.

- [ ] **Step 5: Verify `DiffChannel` is importable from the public API**

```bash
python -c "
from langgraph.channels import DiffChannel
from langgraph.channels.diff import DiffChannel as DC2
from langgraph.checkpoint.base import DiffDelta, DiffChainValue
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], DiffChannel(add_messages)]

print('DiffChannel public API: ok')
"
```
Expected: `DiffChannel public API: ok`

- [ ] **Step 6: Commit final cleanup**

```bash
git add -p  # stage any remaining formatting changes
git commit -m "chore: format and lint DiffChannel implementation"
```

---

## Implementation Notes

**PostgresSaver `_load_diff_chains` complexity:** Task 7 Step 5 describes two approaches. The single bulk range query (using `ANY(%s)` with channel list and `version <= %s` per channel) is preferred for production but requires passing `thread_id`, `checkpoint_ns`, and `channel_versions` down to `_load_blobs`. The per-link single-row query is simpler to implement first. Implement the bulk query for the async saver since async Postgres is the common production path.

**AsyncPostgresSaver:** The async variant in `libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py` has its own `_load_blobs` and `get_tuple`. Apply the same changes there, using `await cur.execute` / `await cur.fetchall` instead of synchronous cursor calls.

**`_pending` accumulation across tasks in one step:** Within one superstep, multiple tasks may write to the same channel. `apply_writes` calls `channel.update(values)` where `values` is the list of all writes from all tasks. `_pending` accumulates all of them. The `DiffDelta.delta` for that step is the full list of writes from all tasks — this is correct.

**`DiffChannel` with `MISSING` initial value:** `from_checkpoint(MISSING)` sets `value = []`. The channel is "available" if `value` is not `MISSING` (the MISSING sentinel from `_internal._typing`). Check `is_available()` returns `True` even for an empty list — messages channel starts empty then gets populated.

**`EmptyChannelError` on `get()`:** The current implementation raises if `self.value is MISSING`. But after `from_checkpoint(MISSING)`, `self.value = []` (not MISSING). This means `get()` returns `[]` rather than raising `EmptyChannelError` for a never-updated DiffChannel. This matches `add_messages` semantics where an empty messages list is valid state. If `EmptyChannelError` is needed (e.g. for graph routing), adjust `is_available` to also check `bool(self.value)`.
