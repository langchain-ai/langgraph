# Pub-sub in LangGraph graphs

LangGraph graphs support two complementary ways to connect nodes:

| Mode | What it is | How a node is started |
|------|------------|------------------------|
| **Workflow** (default) | Direct control flow | Edges, `Command(goto=...)`, `Send` |
| **Pub-sub** | Classical publish / subscribe | Messages on a **topic** |

You can use either mode alone, or **both on the same node**. Pub-sub does **not** replace edges—it is an option next to them.

```text
WORKFLOW                         PUB-SUB
────────                         ───────
  A ──edge──► B                    publisher
                                    │ Publish("events", msg)
                                    ▼
                                  topic "events"
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                     subscriber_1        subscriber_2
```

- **Publishers** do not name subscribers.
- **Subscribers** do not name publishers.
- They only share a **topic** name.

This guide is for people who have never used pub-sub. Every example is runnable with the Graph API (`StateGraph`).

---

## When to use which

| Use **workflow** edges when… | Use **pub-sub** when… |
|------------------------------|------------------------|
| The next step is part of the main recipe | Side effects should not couple to the main path |
| You need a strict order (A then B then C) | Several listeners should react to the same event |
| Routing depends on a condition / `Command` | Producers should not know who is listening |
| Map-reduce with a known target node (`Send`) | You want fan-out by **topic**, not by node name |

**Combine them** when you have a clear workflow spine (ingest → process → respond) and optional side paths (audit log, metrics, cache warmers) that should not clutter the edge list.

---

## Concepts

### Topics

A **topic** is a named channel that collects published messages for one superstep (or longer if `accumulate=True`).

Declare a topic in either of two ways:

1. **As a state key** (payload visible in normal state):

   ```python
   from typing import Annotated, Sequence, TypedDict
   from langgraph.channels import Topic

   class State(TypedDict):
       events: Annotated[Sequence[str], Topic(str)]
   ```

2. **With `add_topic`** (side channel; not required on the TypedDict):

   ```python
   builder.add_topic("events", typ=str)
   ```

If the same name is both a `Topic` state key and passed to `add_topic`, LangGraph reuses the channel (they must agree on `accumulate`).

### Publish

From a node, return a `Publish` (alone or with a state update):

```python
from langgraph.types import Publish

return Publish("events", "user-created")
return {"count": 1}, Publish("events", "tick")
return [Command(update={"count": 1}), Publish("events", "tick")]
```

If the topic is a state key, writing that key also publishes:

```python
return {"events": "user-created"}  # wakes subscribers of "events"
```

### Subscribe and modes

```python
builder.add_node(
    "audit",
    audit_fn,
    mode="pubsub",              # only woken by topics
    subscribes=["events"],
)

builder.add_node(
    "hybrid",
    hybrid_fn,
    mode="both",               # edges AND topics
    subscribes=["events"],
)
```

| `mode` | Activated by |
|--------|----------------|
| `"workflow"` (default) | Edges / `Command` / `Send` |
| `"pubsub"` | Subscribed topics only |
| `"both"` or `("workflow", "pubsub")` | Either |

**Rules of thumb:**

- `subscribes=...` requires `pubsub` in the mode (if you omit `mode`, pubsub is added automatically and workflow is kept).
- `mode="workflow"` **plus** `subscribes=...` is a conflict → error.
- **Edges must not target pubsub-only nodes** (they would never run). Use `mode="both"` if the node should accept edges *and* topics.
- Publishing (`publishes=...` / `Publish`) does **not** change mode; any node may publish.

Declare `publishes=[...]` (or `builder.publish(node, topic)`) so the **diagram** can draw topic edges. Runtime still works if you only return `Publish`, but static visualization uses the declaration.

---

## Minimal example

```python
from typing import Annotated, Sequence, TypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.channels import Topic
from langgraph.types import Publish


class State(TypedDict):
    items: Annotated[list[str], operator.add]
    logs: Annotated[Sequence[str], Topic(str)]


def produce(state: State):
    item = f"item-{len(state['items'])}"
    return {"items": [item]}, Publish("logs", f"produced {item}")


def audit(state: State):
    # Runs because it subscribes to "logs" — no edge from produce
    print("audit saw", state["logs"])
    return {}


builder = StateGraph(State)
builder.add_node("produce", produce, publishes=["logs"])
builder.add_node("audit", audit, mode="pubsub", subscribes=["logs"])
builder.add_edge(START, "produce")
builder.add_edge("produce", END)
graph = builder.compile()

graph.invoke({"items": []})
# produce runs, publishes to logs, audit runs, graph finishes
```

---

## Fan-out (one publisher → many subscribers)

```python
from typing import Annotated, Sequence, TypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.channels import Topic
from langgraph.types import Publish


class State(TypedDict):
    bus: Annotated[Sequence[str], Topic(str)]
    a: Annotated[list[str], operator.add]
    b: Annotated[list[str], operator.add]


def publish(state: State):
    return Publish("bus", "hello")


def listener_a(state: State):
    return {"a": [f"A saw {list(state['bus'])}"]}


def listener_b(state: State):
    return {"b": [f"B saw {list(state['bus'])}"]}


builder = StateGraph(State)
builder.add_node("publish", publish, publishes=["bus"])
builder.add_node("a", listener_a, mode="pubsub", subscribes=["bus"])
builder.add_node("b", listener_b, mode="pubsub", subscribes=["bus"])
builder.add_edge(START, "publish")
builder.add_edge("publish", END)
graph = builder.compile()

print(graph.invoke({"a": [], "b": []}))
# Both a and b receive the same topic batch
```

---

## Fan-in (many publishers → one subscriber)

When several nodes publish in the **same** superstep, the subscriber sees **all** values as a sequence:

```python
def left(state):
    return Publish("bus", "L")

def right(state):
    return Publish("bus", "R")

def join(state):
    return {"sink": [f"got {sorted(state['bus'])}"]}

builder = StateGraph(State)  # State has bus: Topic and sink: list reducer
builder.add_node("left", left, publishes=["bus"])
builder.add_node("right", right, publishes=["bus"])
builder.add_node("join", join, mode="pubsub", subscribes=["bus"])
builder.add_edge(START, "left")
builder.add_edge(START, "right")
builder.add_edge("left", END)
builder.add_edge("right", END)
```

---

## Hybrid graph (workflow spine + side subscribers)

```python
from typing import Annotated, Sequence, TypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.channels import Topic
from langgraph.types import Publish


class State(TypedDict):
    steps: Annotated[list[str], operator.add]
    events: Annotated[Sequence[str], Topic(str)]


def ingest(state: State):
    return {"steps": ["ingest"]}, Publish("events", "ingested")


def process(state: State):
    return {"steps": ["process"]}


def audit(state: State):
    return {"steps": [f"audit:{list(state['events'])}"]}


builder = StateGraph(State)
builder.add_node("ingest", ingest, publishes=["events"])
builder.add_node("process", process)
builder.add_node("audit", audit, mode="pubsub", subscribes=["events"])
builder.add_edge(START, "ingest")
builder.add_edge("ingest", "process")
builder.add_edge("process", END)
graph = builder.compile()

print(graph.invoke({"steps": []}))
# steps includes ingest, process, and an audit:* entry
```

The main path stays a simple chain. Audit is not an edge destination; it only listens.

---

## Side topics (not on the state schema)

```python
class State(TypedDict):
    n: int
    sink: Annotated[list[str], operator.add]


def src(state: State):
    return {"n": state["n"] + 1}, Publish("metrics", {"n": state["n"]})


def metrics_sink(state: State):
    # Subscribed topics are added to the node input even if not on State
    return {"sink": [str(state.get("metrics"))]}  # type: ignore[typeddict-item]


builder = StateGraph(State)
builder.add_topic("metrics", typ=dict)
builder.add_node("src", src, publishes=["metrics"])
builder.add_node("metrics_sink", metrics_sink, mode="pubsub", subscribes=["metrics"])
builder.add_edge(START, "src")
builder.add_edge("src", END)
```

Prefer **state-key topics** when handlers should type the payload in the TypedDict. Use **side topics** for telemetry-style channels you do not want in the main schema.

---

## Fluent wiring

Equivalent to kwargs on `add_node`:

```python
builder.add_node("src", src)
builder.add_node("dst", dst)
builder.publish("src", "events")
builder.subscribe("dst", "events")  # adds pubsub mode if needed
```

---

## Visualization

```python
print(graph.get_graph().draw_mermaid())
```

You should see:

- Node metadata: `mode = workflow`, `mode = pubsub`, or `mode = pubsub+workflow`
- Optional `publishes = ...` / `subscribes = ...`
- Dashed edges labeled `topic:<name>` from publishers to subscribers

Example shape:

```text
__start__ --> produce
produce -. topic:logs .-> audit
produce --> __end__
```

Declare `publishes=` so those topic edges appear without executing real node logic (static analysis).

---

## `Publish` vs `Send`

| | `Publish(topic, value)` | `Send(node, arg)` |
|--|-------------------------|-------------------|
| Addressing | By **topic** (anonymous consumers) | By **node name** |
| Consumer input | Graph state (+ topic batch) | Custom `arg` as node input |
| Typical use | Events, audits, multi-listener | Map-reduce fan-out |
| Coupling | Loose | Explicit target |

Use `Send` when you know the worker node and want a custom per-task payload. Use `Publish` when any number of listeners should react without the producer listing them.

---

## Modes in one place

```python
# Workflow only (default) — edges / Command
builder.add_node("a", fn_a)

# Pub-sub only — topics; do not add edges TO this node
builder.add_node("b", fn_b, mode="pubsub", subscribes=["t"])

# Both — edges and topics
builder.add_node("c", fn_c, mode="both", subscribes=["t"])
# same as mode=("workflow", "pubsub")
```

---

## Pitfalls

1. **Nothing publishes → subscribers never run.** There must be a workflow path that eventually `Publish`es (or writes the Topic key).
2. **Edge into `mode="pubsub"` only** → compile error. Use `mode="both"` if the node should also be an edge target.
3. **Topic not in state** → payload is still injected under the topic name for subscribers, but it will not appear in typed `State` fields or default outputs unless you put it on the schema / output schema.
4. **Self-sustaining loops** (`subscribe` + `publish` on the same topic) can run until the recursion limit. Prefer a workflow sink (`END`) or stop publishing when done.
5. **Diagram missing topic edges** → add `publishes=["topic"]` on the publisher node.
6. **`accumulate=False` (default)** clears the topic after each step once updates are applied; use `accumulate=True` only when you need history across steps.
7. **Subgraphs** do not share topics with parents unless you pass state keys through shared schema design; there is no global process-wide bus.

---

## API checklist

```python
from langgraph.graph import StateGraph, START, END, Publish
from langgraph.channels import Topic
from langgraph.types import Publish  # same object

builder.add_topic(name, typ=Any, accumulate=False)
builder.add_node(
    name,
    fn,
    mode="workflow" | "pubsub" | "both" | ("workflow", "pubsub"),
    publishes=["topic", ...],
    subscribes=["topic", ...],
)
builder.publish(node, *topics)
builder.subscribe(node, *topics)
# runtime
return Publish("topic", value)
```

For low-level actor/channel graphs, the same idea exists as `Topic` + `NodeBuilder().subscribe_to(...).write_to(...)`. The Graph API above is the supported way to do classical pub-sub on `StateGraph`.
