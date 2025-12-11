# Understanding `channel_versions` for State Channels in LangGraph

## Overview

While `versions_seen` only tracks trigger channels, `channel_versions` tracks **ALL channels** including state channels like `fieldA` and `fieldB`. This document explains why state channel versions matter.

## The Two Version Tracking Mechanisms

| Mechanism | What it tracks | Purpose |
|-----------|---------------|---------|
| `channel_versions` | **All channels** (state + triggers) | Storage, recovery, change tracking |
| `versions_seen` | **Only triggers** | Scheduling, prevent duplicate execution |

---

## Why Track State Channel Versions?

### Purpose 1: Incremental Storage

When saving a checkpoint, LangGraph only serializes channels that have **changed** since the last checkpoint.

```python
# In checkpointer.put()
def put(self, config, checkpoint, metadata, new_versions):
    for k, v in new_versions.items():  # Only changed channels!
        self.blobs[(thread_id, ns, k, v)] = serialize(values[k])
```

The `new_versions` parameter is computed by comparing current versions with previous versions:

```python
def get_new_channel_versions(previous_versions, current_versions):
    """Get subset of current_versions that are newer than previous_versions."""
    return {
        k: v
        for k, v in current_versions.items()
        if v > previous_versions.get(k, null_version)
    }
```

**Benefit**: If `fieldA` didn't change in a step, it won't be re-serialized!

### Purpose 2: Version-Keyed Storage

Channel values are stored with their version as part of the key:

```python
# Storage structure in InMemorySaver
blobs = {
    (thread_id, ns, "fieldA", v02): b"Hello",         # Step 0
    (thread_id, ns, "fieldA", v03): b"Hello->A",      # Step 1  
    (thread_id, ns, "fieldA", v04): b"Hello->A->B",   # Step 2
    (thread_id, ns, "fieldB", v02): b"World",         # Step 0
    (thread_id, ns, "fieldB", v03): b"World->A",      # Step 1
    ...
}
```

**Benefit**: Can restore to ANY historical checkpoint - each version's value is stored independently.

### Purpose 3: Precise Recovery

When restoring a checkpoint, use `channel_versions` to load the correct value:

```python
# Restoring Step 1's checkpoint
checkpoint["channel_versions"] = {"fieldA": v03, "fieldB": v03}

# Load correct version of each value
fieldA = blobs[(thread_id, ns, "fieldA", v03)]  # "Hello->A"
fieldB = blobs[(thread_id, ns, "fieldB", v03)]  # "World->A"
```

---

## Example: Step-by-Step Channel Version Changes

Using the same graph:

```
        ┌──────────┐
        │  nodeA   │  (reads fieldA + fieldB)
        └────┬─────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌──────────┐
│  nodeB   │  │  nodeC   │  (nodeB reads fieldA, nodeC reads fieldB)
└────┬─────┘  └────┬─────┘
      │             │
      └──────┬──────┘
             ▼
        ┌──────────┐
        │  nodeD   │  (reads fieldA + fieldB)
        └──────────┘
```

### Step -1: Input

```
channel_versions:
    __start__: v01

channel_values:
    __start__: {'fieldA': 'Hello', 'fieldB': 'World'}

new_versions (to save): {__start__: v01}
  → Only __start__ is saved
```

### Step 0: `__start__` executes

```
channel_versions:
    __start__:        v02
    branch:to:nodeA:  v02
    fieldA:           v02  ← NEW!
    fieldB:           v02  ← NEW!

channel_values:
    branch:to:nodeA: None
    fieldA: Hello
    fieldB: World

new_versions (to save): {__start__: v02, branch:to:nodeA: v02, fieldA: v02, fieldB: v02}
  → All changed channels are saved
```

### Step 1: nodeA executes

```
channel_versions:
    __start__:        v02  ← unchanged
    branch:to:nodeA:  v03  ← updated (consumed)
    branch:to:nodeB:  v03  ← NEW!
    branch:to:nodeC:  v03  ← NEW!
    fieldA:           v03  ← updated!
    fieldB:           v03  ← updated!

channel_values:
    branch:to:nodeB: None
    branch:to:nodeC: None
    fieldA: Hello->A
    fieldB: World->A

new_versions (to save): {branch:to:nodeA: v03, branch:to:nodeB: v03, branch:to:nodeC: v03, fieldA: v03, fieldB: v03}
  → __start__ NOT saved (unchanged at v02)
```

### Step 2: nodeB and nodeC execute (parallel)

```
channel_versions:
    __start__:              v02  ← unchanged
    branch:to:nodeA:        v03  ← unchanged
    branch:to:nodeB:        v04  ← updated (consumed)
    branch:to:nodeC:        v04  ← updated (consumed)
    fieldA:                 v04  ← updated by nodeB!
    fieldB:                 v04  ← updated by nodeC!
    join:nodeB+nodeC:nodeD: v04  ← NEW!

channel_values:
    fieldA: Hello->A->B
    fieldB: World->A->C
    join:nodeB+nodeC:nodeD: {'nodeB', 'nodeC'}

new_versions (to save): {branch:to:nodeB: v04, branch:to:nodeC: v04, fieldA: v04, fieldB: v04, join:...: v04}
  → Only changed channels saved
```

### Step 3: nodeD executes

```
channel_versions:
    __start__:              v02  ← unchanged since Step 0!
    branch:to:nodeA:        v03  ← unchanged since Step 1
    branch:to:nodeB:        v04  ← unchanged
    branch:to:nodeC:        v04  ← unchanged
    fieldA:                 v05  ← updated by nodeD!
    fieldB:                 v05  ← updated by nodeD!
    join:nodeB+nodeC:nodeD: v05  ← updated (consumed)

channel_values:
    fieldA: Hello->A->B->D
    fieldB: World->A->C->D
    join:nodeB+nodeC:nodeD: set()

new_versions (to save): {fieldA: v05, fieldB: v05, join:...: v05}
  → Only 3 channels saved, not all 7!
```

---

## Storage Efficiency Visualization

```
Step 0:  Save [__start__, branch:to:nodeA, fieldA, fieldB]     = 4 channels
Step 1:  Save [branch:to:nodeA, branch:to:nodeB, branch:to:nodeC, fieldA, fieldB] = 5 channels
Step 2:  Save [branch:to:nodeB, branch:to:nodeC, fieldA, fieldB, join:...] = 5 channels
Step 3:  Save [fieldA, fieldB, join:...]                       = 3 channels

Without incremental storage: 7 channels × 4 steps = 28 serializations
With incremental storage:    4 + 5 + 5 + 3        = 17 serializations
                                                    = 39% savings!
```

---

## Time Travel: Restoring Any Checkpoint

Because each version is stored separately, you can restore to any point:

```
Want to restore Step 1?
  → checkpoint["channel_versions"] = {fieldA: v03, fieldB: v03, ...}
  → Load blobs[(thread_id, ns, "fieldA", v03)] = "Hello->A"
  → Load blobs[(thread_id, ns, "fieldB", v03)] = "World->A"

Want to restore Step 2?
  → checkpoint["channel_versions"] = {fieldA: v04, fieldB: v04, ...}
  → Load blobs[(thread_id, ns, "fieldA", v04)] = "Hello->A->B"
  → Load blobs[(thread_id, ns, "fieldB", v04)] = "World->A->C"
```

---

## Summary: State Channel Versions vs Trigger Channel Versions

| Aspect | State Channels (fieldA, fieldB) | Trigger Channels (branch:to:*) |
|--------|--------------------------------|-------------------------------|
| In `channel_versions`? | ✅ Yes | ✅ Yes |
| In `versions_seen`? | ❌ No | ✅ Yes |
| Version increases when? | Value is updated | Written to OR consumed |
| Used for scheduling? | ❌ No | ✅ Yes |
| Used for storage? | ✅ Yes (incremental save) | ✅ Yes |
| Used for recovery? | ✅ Yes (load correct version) | ✅ Yes |

---

## Code Reference

### Where `channel_versions` is updated

```python
# In apply_writes() - libs/langgraph/langgraph/pregel/_algo.py
for chan, vals in pending_writes_by_channel.items():
    if channels[chan].update(vals) and next_version is not None:
        checkpoint["channel_versions"][chan] = next_version  # ← Update version
        updated_channels.add(chan)
```

### Where incremental storage happens

```python
# In InMemorySaver.put() - libs/checkpoint/langgraph/checkpoint/memory/__init__.py
def put(self, config, checkpoint, metadata, new_versions):
    values = checkpoint.pop("channel_values")
    for k, v in new_versions.items():  # ← Only save changed channels
        self.blobs[(thread_id, checkpoint_ns, k, v)] = (
            self.serde.dumps_typed(values[k]) if k in values else ("empty", b"")
        )
```

---

## Running the Test

To see channel versions in action:

```bash
cd libs/langgraph
uv run python test_versions_seen.py
```

The output shows `channel_versions` for each step, where you can observe:
1. All channels (state + triggers) are tracked
2. Versions increment when values change
3. Some channels stay at the same version across multiple steps (unchanged)

