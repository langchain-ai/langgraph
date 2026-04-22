# feat(channels): DeltaChannel — O(N) incremental checkpoint storage

## The problem

LangGraph checkpoints store the **full accumulated value** of every channel on every step. For a `messages` channel backed by `add_messages`, that means each checkpoint blob contains the entire conversation history up to that point.

Storage cost grows **O(N²)** in the number of turns:

| Step | Checkpoint blob |
|------|----------------|
| 1    | [msg_1] |
| 2    | [msg_1, msg_2] |
| N    | [msg_1, ..., msg_N] |

At 100K tokens of conversation data, a single thread accumulates ~250 MB; with large messages or file attachments costs scale even faster.

## The fix: `DeltaChannel`

`DeltaChannel` is an opt-in wrapper around any binary reducer that stores only a **sentinel marker** in `checkpoint_blobs` rather than the full accumulated value. The actual per-step writes stay in `checkpoint_writes` (which every checkpointer already writes unconditionally). At read time the saver walks the ancestor chain, collects all writes for the channel, and replays them through the reducer.

Storage scales **O(N)** — the sentinel blob is effectively zero bytes, and the writes table already exists.

```python
from langgraph.channels.delta import DeltaChannel
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Before: O(N²) storage
    messages: Annotated[list[AnyMessage], add_messages]

    # After: O(N) storage
    messages: Annotated[list[AnyMessage], DeltaChannel(add_messages)]
```

## Benchmarks

Simulated with realistic paragraph-length messages (~100 tokens each, ~400 chars). Each turn = one human + one AI message (~200 tokens total).

### Storage (InMemorySaver)

| turns | ctx | add_msgs | delta | savings |
|------:|----:|---------:|------:|--------:|
| 10 | ~2K tok | 108.6 KB | 4.0 KB | 27x |
| 25 | ~5K tok | 649.0 KB | 10.1 KB | 64x |
| 50 | ~10K tok | 2.6 MB | 20.2 KB | 126x |
| 100 | ~20K tok | 10.2 MB | 40.5 KB | 251x |
| 500 | ~100K tok | 252.6 MB | 202.8 KB | 1245x |

Savings grow with N because `add_messages` is O(N²) while `DeltaChannel` is O(N). The sentinel blob itself is essentially zero bytes.

### Read latency (avg of 5 `get_state` calls = cost per `invoke`)

| turns | ctx | add_msgs | delta |
|------:|----:|---------:|------:|
| 10 | ~2K tok | 0.1ms | 0.2ms |
| 25 | ~5K tok | 0.2ms | 0.5ms |
| 50 | ~10K tok | 0.4ms | 1.5ms |
| 100 | ~20K tok | 0.7ms | 4.8ms |
| 500 | ~100K tok | 5.8ms | 114.9ms |

**This cost is paid once per `invoke`/`stream` call, not per node.** Within a single invocation, all channels are loaded into memory once at the start and shared across every node — there is no per-node reconstruction. The 114.9ms at 500 turns is what you pay each time a user sends a new message, not on each step of the graph.

## How it works

**Write:** `DeltaChannel.checkpoint()` always emits `DeltaChannelSentinel()` — a tiny marker (zero payload bytes) stored in `checkpoint_blobs`. Per-step writes flow into `checkpoint_writes` as they normally do for every channel.

**Read:** The saver detects `DeltaChannelSentinel` values in `channel_values` and replaces them by calling `get_channel_writes` / `aget_channel_writes`, which walks the ancestor checkpoint chain and collects all writes for that channel (oldest→newest). `DeltaChannel.from_checkpoint()` replays those writes through the operator to reconstruct the full value.

**Saver implementations:**
- `InMemorySaver` — direct dict traversal of `self.storage` and `self.writes`, no I/O
- `PostgresSaver` (sync + async) — two queries: one cheap ID walk across the thread, one `ANY()` fetch of writes; no recursive CTE
- All other savers — `BaseCheckpointSaver.get_channel_writes` fallback via `list()`, with a re-entrancy guard to prevent infinite recursion

## Changes

**`libs/checkpoint`**
- `base/__init__.py` — add `DeltaChannelSentinel` marker dataclass; add `get_channel_writes` / `aget_channel_writes` to `BaseCheckpointSaver` with a `list()`-based fallback and re-entrancy guard

**`libs/checkpoint/memory`**
- `memory/__init__.py` — `get_channel_writes` via direct dict traversal; `_resolve_delta_channels` helper called in `get_tuple` / `aget_tuple` to replace sentinels with reconstructed write lists

**`libs/langgraph`**
- `channels/delta.py` — `DeltaChannel` implementation: `checkpoint()` always emits sentinel, `from_checkpoint()` replays writes list
- `channels/__init__.py` — export `DeltaChannel`
- `graph/state.py` — recognize `DeltaChannel` as a valid channel annotation
- `pregel/_checkpoint.py` / `pregel/_loop.py` — wire `after_checkpoint` hook; call it after each checkpointing step so `DeltaChannel` can advance internal state

**`libs/checkpoint-postgres`**
- `postgres/base.py` — `_get_channel_writes_cur` two-query ancestor walk (sync); `_resolve_delta_channels` called after `_load_blobs`
- `postgres/aio.py` — `_aget_channel_writes_cur` (async counterpart)

## Open questions

**Should we add a compile-time capability check?**

Currently misconfiguring `DeltaChannel` with an unsupported saver only errors at runtime on first reload. A protocol-based check at `compile()` time would give an early warning without requiring a manual boolean flag.

**`snapshot_every` for bounded reconstruction cost?**

Both per-invoke read latency and total write wall time grow O(N) per invoke / O(N²) total as the conversation lengthens. A `snapshot_every` parameter — periodically store a full snapshot in `checkpoint_blobs` to cap chain depth — would bound reconstruction cost and is a natural follow-up once the core design is stable.

## Backwards compatibility

| Scenario | Behaviour |
|----------|-----------|
| Existing graph using `add_messages` | Unaffected — no code or schema changes |
| `DeltaChannel` loading an old full-list checkpoint blob | Handled via backwards-compat path in `from_checkpoint` |
| `DeltaChannel` with `InMemorySaver` or `PostgresSaver` | Fully supported |
| Time-travel to a past checkpoint | Ancestor walk uses the version at that checkpoint — correct by construction |
| `Overwrite` value | Resets the effective chain; reconstruction starts from that step |

## Test plan

- [x] `DeltaChannel` unit tests: `update` → `checkpoint` lifecycle, `from_checkpoint` chain replay, backwards-compat with plain list, `Overwrite` resets chain
- [x] `InMemorySaver` `get_channel_writes`: assembles write list from dict storage
- [x] Serde round-trip for `DeltaChannelSentinel`
- [x] End-to-end graph tests: multi-turn conversations accumulate correctly, time-travel reconstructs correct partial history
- [x] `PostgresSaver` two-query chain reconstruction (sync + async)
- [x] `BaseCheckpointSaver` fallback path via `list()` with re-entrancy guard
- [x] Storage benchmark: `DeltaChannel` uses strictly less storage than `add_messages` at all measured turn counts

---

## Changes from previous base branch

The previous version stored `DeltaValue` objects (containing the per-step writes) directly in `checkpoint_blobs` and used a `DeltaChainValue` to represent the assembled chain. Reconstruction required a dedicated `get_delta_chain` / `aget_delta_chain` protocol and a recursive CTE in Postgres.

This version pivots to a simpler design:
- **Sentinel in blobs, writes in `checkpoint_writes`** — `checkpoint_blobs` stores only a zero-byte `DeltaChannelSentinel` marker. The actual per-step data already lives in `checkpoint_writes` (written unconditionally by every checkpointer), so blob storage is essentially free. This is why storage savings jump to 1245x at 500 turns.
- **No custom serde type for the delta payload** — `DeltaValue` / `DeltaChainValue` and the `"delta"` serde type tag are gone. Writes are deserialized with the same serde path they were originally written with.
- **Postgres: two queries instead of a recursive CTE** — fetch all `(checkpoint_id, parent_checkpoint_id)` pairs for the thread, walk the ancestor chain in Python, then fetch writes with a plain `ANY()` filter.
- **Universal fallback on `BaseCheckpointSaver`** — the base class now provides `get_channel_writes` via `list()`, so any third-party saver works without modification.
- **`snapshot_every` removed** — deferred as a follow-up; the simpler design is easier to reason about and delivers larger storage savings.
