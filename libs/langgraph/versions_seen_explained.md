# Understanding `versions_seen` in LangGraph Checkpoints

## Overview

`versions_seen` is a nested dictionary in the checkpoint that tracks which channel versions each node has processed. It's defined in `libs/checkpoint/langgraph/checkpoint/base/__init__.py`:

```python
versions_seen: dict[str, ChannelVersions]
"""Map from node ID to map from channel name to version seen.
This keeps track of the versions of the channels that each node has seen.
Used to determine which nodes to execute next.
"""
```

## Data Structure

```
versions_seen = {
    "node_name": {
        "channel_name": version,
        ...
    },
    ...
}
```

## Key Point

**`versions_seen` only records trigger channels, NOT state channels!**

In StateGraph:
- `triggers` = edge control channels like `branch:to:nodeA`
- `channels` = state keys like `fieldA`, `fieldB`

So `versions_seen` records `branch:to:*` channels, **NOT** `fieldA` or `fieldB`.

---

## Example Graph

```
        ┌──────────┐
        │  nodeA   │
        └────┬─────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌──────────┐
│  nodeB   │  │  nodeC   │
└────┬─────┘  └────┬─────┘
      │             │
      └──────┬──────┘
             ▼
        ┌──────────┐
        │  nodeD   │
        └──────────┘
```

### State Definition

```python
class State(TypedDict):
    fieldA: str
    fieldB: str

class StateOnlyA(TypedDict):
    """Input schema for nodeB - only reads fieldA"""
    fieldA: str

class StateOnlyB(TypedDict):
    """Input schema for nodeC - only reads fieldB"""
    fieldB: str
```

---

## Compiled Graph Structure

### Channels Created

| Channel | Type | Purpose |
|---------|------|---------|
| `fieldA` | LastValue | State data |
| `fieldB` | LastValue | State data |
| `__start__` | EphemeralValue | Input channel |
| `branch:to:nodeA` | EphemeralValue | Edge control channel |
| `branch:to:nodeB` | EphemeralValue | Edge control channel |
| `branch:to:nodeC` | EphemeralValue | Edge control channel |
| `branch:to:nodeD` | EphemeralValue | Edge control channel |
| `join:nodeB+nodeC:nodeD` | NamedBarrierValue | Parallel join channel |

### Nodes Configuration

| Node | triggers | channels | Note |
|------|----------|----------|------|
| `__start__` | `["__start__"]` | `"__start__"` | Input node |
| `nodeA` | `["branch:to:nodeA"]` | `["fieldA", "fieldB"]` | Reads full state |
| `nodeB` | `["branch:to:nodeB"]` | `["fieldA"]` | Only reads fieldA (via `input_schema=StateOnlyA`) |
| `nodeC` | `["branch:to:nodeC"]` | `["fieldB"]` | Only reads fieldB (via `input_schema=StateOnlyB`) |
| `nodeD` | `["branch:to:nodeD", "join:nodeB+nodeC:nodeD"]` | `["fieldA", "fieldB"]` | Reads full state |

**Note**: `triggers` are edge control channels, `channels` are state fields the node reads. Use `input_schema` to control which fields a node reads.

---

## Step-by-Step Execution

### Input

```python
{"fieldA": "Hello", "fieldB": "World"}
```

---

### Step -1: Input Phase (source: input)

```
State values: {}

channel_versions:
    __start__: v01

versions_seen:
    __input__: {}
```

Initial checkpoint when input is received.

---

### Step 0: `__start__` executes (source: loop)

```
State values: {'fieldA': 'Hello', 'fieldB': 'World'}

channel_versions:
    __start__:        v02
    branch:to:nodeA:  v02
    fieldA:           v02
    fieldB:           v02

versions_seen:
    __input__: {}
    __start__:
        __start__: v01    ← __start__ node saw __start__ channel
```

**Note**: `fieldA` and `fieldB` are NOT in `versions_seen`!

---

### Step 1: nodeA executes (source: loop)

```
nodeA reads: fieldA='Hello', fieldB='World'
State values: {'fieldA': 'Hello->A', 'fieldB': 'World->A'}

channel_versions:
    __start__:        v02
    branch:to:nodeA:  v03
    branch:to:nodeB:  v03
    branch:to:nodeC:  v03
    fieldA:           v03
    fieldB:           v03

versions_seen:
    __input__: {}
    __start__:
        __start__: v01
    nodeA:
        branch:to:nodeA: v02    ← nodeA saw its trigger
```

**Key observation**:
- `nodeA`'s `versions_seen` only records `branch:to:nodeA`
- **NO** `fieldA` or `fieldB` because they are NOT triggers!

---

### Step 2: nodeB and nodeC execute in parallel (source: loop)

```
nodeB reads: fieldA='Hello->A'
nodeC reads: fieldB='World->A'
State values: {'fieldA': 'Hello->A->B', 'fieldB': 'World->A->C'}

channel_versions:
    __start__:              v02
    branch:to:nodeA:        v03
    branch:to:nodeB:        v04
    branch:to:nodeC:        v04
    fieldA:                 v04
    fieldB:                 v04
    join:nodeB+nodeC:nodeD: v04

versions_seen:
    __input__: {}
    __start__:
        __start__: v01
    nodeA:
        branch:to:nodeA: v02
    nodeB:
        branch:to:nodeB: v03    ← nodeB saw its trigger
    nodeC:
        branch:to:nodeC: v03    ← nodeC saw its trigger
```

---

### Step 3: nodeD executes (source: loop)

```
nodeD reads: fieldA='Hello->A->B', fieldB='World->A->C'
State values: {'fieldA': 'Hello->A->B->D', 'fieldB': 'World->A->C->D'}

channel_versions:
    __start__:              v02
    branch:to:nodeA:        v03
    branch:to:nodeB:        v04
    branch:to:nodeC:        v04
    fieldA:                 v05
    fieldB:                 v05
    join:nodeB+nodeC:nodeD: v05

versions_seen:
    __input__: {}
    __start__:
        __start__: v01
    nodeA:
        branch:to:nodeA: v02
    nodeB:
        branch:to:nodeB: v03
    nodeC:
        branch:to:nodeC: v03
    nodeD:
        join:nodeB+nodeC:nodeD: v04    ← nodeD saw join channel
```

---

## Summary Table

| Node | versions_seen records | Why? |
|------|----------------------|------|
| `__start__` | `__start__` | Its trigger is `__start__` |
| `nodeA` | `branch:to:nodeA` | Its trigger is `branch:to:nodeA` |
| `nodeB` | `branch:to:nodeB` | Its trigger is `branch:to:nodeB` |
| `nodeC` | `branch:to:nodeC` | Its trigger is `branch:to:nodeC` |
| `nodeD` | `join:nodeB+nodeC:nodeD` | One of its triggers (join channel) |

---

## Conclusion

- `versions_seen` **only records triggers**
- `fieldA` and `fieldB` **never appear** in `versions_seen`
- In StateGraph, triggers are edge control channels (`branch:to:*`), not state fields
- The purpose of `versions_seen` is to **prevent duplicate triggering**, so it only needs to track trigger channel versions

---

## Deep Dive: How `versions_seen` Determines the Last Node

### The Problem

When calling `update_state()` without specifying `as_node`, LangGraph needs to figure out which node "last updated" the state. This is done using `versions_seen`.

### The Algorithm

```python
last_seen_by_node = sorted(
    (v, n)
    for n, seen in checkpoint["versions_seen"].items()
    if n in self.nodes
    for v in seen.values()
)
```

This creates a sorted list of `(version, node_name)` tuples.

### Key Insight: Version = Superstep

**Nodes that execute in the same superstep (parallel execution) will have the same trigger channel version.**

This is because:
1. Each superstep increments the version counter
2. All nodes triggered in the same superstep see the same version
3. So `version` effectively identifies which superstep a node executed in

### Analysis by Step (Using Our Example)

#### Step 1: After nodeA executes

```
versions_seen:
    __start__: { __start__: v01 }
    nodeA:     { branch:to:nodeA: v02 }

last_seen_by_node = [(v01, "__start__"), (v02, "nodeA")]

Check: last[-1][0] != last[-2][0]?
       v02 != v01? ✅ YES

Result: as_node = "nodeA" (last node in the latest superstep)
```

#### Step 2: After nodeB and nodeC execute (parallel)

```
versions_seen:
    __start__: { __start__: v01 }
    nodeA:     { branch:to:nodeA: v02 }
    nodeB:     { branch:to:nodeB: v03 }  ← same version!
    nodeC:     { branch:to:nodeC: v03 }  ← same version!

last_seen_by_node = [(v01, "__start__"), (v02, "nodeA"), (v03, "nodeB"), (v03, "nodeC")]

Check: last[-1][0] != last[-2][0]?
       v03 != v03? ❌ NO (same version = same superstep)

Result: AMBIGUOUS! Multiple nodes executed in the last superstep.
        → Raises InvalidUpdateError("Ambiguous update, specify as_node")
```

#### Step 3: After nodeD executes

```
versions_seen:
    __start__: { __start__: v01 }
    nodeA:     { branch:to:nodeA: v02 }
    nodeB:     { branch:to:nodeB: v03 }
    nodeC:     { branch:to:nodeC: v03 }
    nodeD:     { join:nodeB+nodeC:nodeD: v04 }

last_seen_by_node = [(v01, "__start__"), (v02, "nodeA"), (v03, "nodeB"), (v03, "nodeC"), (v04, "nodeD")]

Check: last[-1][0] != last[-2][0]?
       v04 != v03? ✅ YES

Result: as_node = "nodeD" (only node in the latest superstep)
```

### Summary Table

| Step | Last Two Versions | Same Superstep? | as_node |
|------|-------------------|-----------------|---------|
| Step 1 | v02, v01 | No | ✅ nodeA |
| Step 2 | v03, v03 | **Yes (parallel!)** | ❌ Ambiguous |
| Step 3 | v04, v03 | No | ✅ nodeD |

### Visual Representation

```
Superstep Timeline:

  Superstep 0    Superstep 1    Superstep 2    Superstep 3
  (v01, v02)        (v03)          (v04)          (v05)
      │               │              │              │
      ▼               ▼              ▼              ▼
  ┌────────┐    ┌──────────┐   ┌─────────┐    ┌────────┐
  │__start__│    │  nodeA   │   │ nodeB   │    │ nodeD  │
  └────────┘    └──────────┘   │ nodeC   │    └────────┘
                               │(parallel)│
                               └─────────┘

When version[-1] == version[-2]:
  → Multiple nodes in the same superstep
  → Cannot determine which one was "last"
  → Ambiguous!
```

### The Logic Explained

```python
if last_seen_by_node:
    if len(last_seen_by_node) == 1:
        # Only one node ever executed
        as_node = last_seen_by_node[0][1]
    elif last_seen_by_node[-1][0] != last_seen_by_node[-2][0]:
        # Last two have different versions
        # → Last superstep had only ONE node
        # → That node is unambiguously the "last" one
        as_node = last_seen_by_node[-1][1]
    # else: versions are equal
    #     → Multiple nodes in the last superstep
    #     → Ambiguous, will raise error later
```

---

## How Scheduling Works

```python
def _triggers(channels, versions, seen, null_version, proc) -> bool:
    for chan in proc.triggers:  # Only checks triggers!
        if channels[chan].is_available() and \           # Condition 1: channel has value
           versions.get(chan, null_version) > seen.get(chan, null_version):  # Condition 2: version updated
            return True
    return False
```

Translation:
> Trigger the node if ANY trigger channel satisfies **BOTH** conditions:
> 1. `is_available()` - the channel has a value
> 2. `current_version > seen_version` - the version is newer than what the node has seen

**Important**: Both conditions must be met! This is why `EphemeralValue` channels (like `branch:to:*`) 
can have their version increase after being consumed, but won't re-trigger the node because 
`is_available()` returns `False` after consumption.

Since only triggers are checked, only trigger versions need to be recorded in `versions_seen`.

---

## Running the Test

To run the test script yourself:

```bash
cd libs/langgraph
uv run python test_versions_seen.py
```

