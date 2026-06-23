# Graph Capabilities

**Problem:** LangGraph executes graphs well, but graphs are not first-class *reusable assets*. Teams fork code, couple on internal state, and cannot cleanly publish libraries internally or as OSS.

**Solution:** A **capability** is a versioned graph with a strict I/O contract, delivered as:

| Mode | Use case |
|------|----------|
| **Package** (this phase) | Local install, in-process compose, OSS libraries |
| **Service** (later phase) | Black-box remote invoke, org isolation, independent deploy |

Consumers depend on `capability_id` + semver + input/output schemas — never internal node names or private channels.

## Quick start (package mode)

```python
from langgraph.capability import CapabilitySpec, graph_capability, attach_capability
from langgraph.capability.examples.research import RESEARCH_CAPABILITY, ResearchInput
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Standalone invoke at the capability boundary
out = RESEARCH_CAPABILITY.invoke({"query": "langgraph capabilities"})

# Or compose in a parent graph (mapped I/O; no shared internal state)
class S(TypedDict):
    query: str
    findings: str

g = StateGraph(S)
attach_capability(
    g, "research", RESEARCH_CAPABILITY,
    input_mapper=lambda s: {"query": s["query"]},
    output_mapper=lambda o: {"findings": o["findings"]},
)
```

## Authoring rules (library capabilities)

1. Export `CapabilitySpec` with stable `capability_id` and semver `version`.
2. Define explicit `input_schema` / `output_schema` (and optional `public_params_schema`).
3. Use `state_boundary=ISOLATED` or `MAPPED` — **never `SHARED`** for published capabilities.
4. Single entrypoint: `build_<name>_graph(**public_params)` — no secrets in the package builder.
5. Declare `side_effects`; document the error model (defaults are fine).
6. Ship tests that invoke via the capability boundary and at least one parent composition.

## Versioning

- Schemas are the API: breaking I/O changes require a **major** semver bump.
- Pin package versions in application lockfiles.
- `select_capability_version()` resolves `*`, `1`, `1.2`, or exact `1.2.3` requests.

## Reference examples

- `langgraph.capability.examples.research` — `langgraph.research@1.0.0`
- `langgraph.capability.examples.review` — `langgraph.review@1.0.0`
- `langgraph.capability.examples.parent_app` — parent composing both via `attach_capability`

## Service mode (phase 2)

Same **capability id + semver + I/O schemas** as the package; delivery is a deployed graph.

```python
from langgraph.capability import ServiceEndpoint, service_capability_from_package
from langgraph.capability.examples.research import RESEARCH_CAPABILITY
from langgraph.capability.service import local_service_invoker  # tests/dev only

# Production: url + assistant_id → RemoteGraph under the hood
# Tests/dev: inject local_service_invoker(package_cap) to stay hermetic
svc = service_capability_from_package(
    RESEARCH_CAPABILITY,
    ServiceEndpoint(url="http://localhost:2024", assistant_id="research", version_label="1.0.0"),
)

# Parent composition (black-box node)
from langgraph.capability import attach_service_capability
attach_service_capability(parent_graph, "research", svc, input_mapper=..., output_mapper=...)
```

Provider deploy surface (example): `langgraph.capability.examples.service_deploy.build_research_service_graph`
wired in `langgraph.json` as graph id `research`. Callers never import internal nodes.

**Observability:** service mode exposes boundary events (`iter_boundary_events`) — run id, status,
version — not internal xray. Use package mode when you need subgraph traces.

## Ergonomics (phase 3)

### Catalog

```python
from langgraph.capability import CapabilityCatalog, default_example_catalog

cat = default_example_catalog()
cat.get_package("langgraph.research", "1")
cat.get_service("langgraph.research", "1.0.0")
print(cat.to_summary())  # org-facing inventory
```

### Config refs (deploy / app wiring)

| Scheme | Example |
|--------|---------|
| `python:module:attr` | `python:langgraph.capability.examples.research:RESEARCH_CAPABILITY` |
| `service:id@version?url=&assistant_id=` | `service:langgraph.research@1?url=http://localhost:2024&assistant_id=research` |
| `catalog:id@version:package\|service` | `catalog:langgraph.review@1:package` |

### `add_capability_node`

```python
from langgraph.capability import add_capability_node, default_example_catalog

add_capability_node(graph, "research", "langgraph.research", catalog=cat, mode="package", ...)
add_capability_node(graph, "review", "catalog:langgraph.review@1:package", catalog=cat, ...)
add_capability_node(graph, "research", RESEARCH_CAPABILITY, ...)  # direct object
```

## Phases

1. **Contract + package** — specs, package delivery, local composition, docs/tests
2. **Service mode** — remote black-box delivery via existing deploy/RemoteGraph patterns
3. **Ergonomics** (current) — catalog, config refs, `add_capability_node`
4. **Harden** — parity notes, multi-version windows, progress events
