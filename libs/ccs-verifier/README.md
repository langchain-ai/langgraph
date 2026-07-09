# CCS Verifier for LangGraph

**Runtime Integrity Governance Layer based on CCS 1.0 Specification**

This package provides formal behavioral conformance verification for LangGraph checkpointers, enforcing runtime state sovereignty at the persistence boundary.

## The Gap

| Layer | Question | Status |
|-------|----------|--------|
| Storage Conformance | "Did data persist correctly?" | ✅ Solved by `langgraph-checkpoint-conformance` |
| **Behavioral Conformance** | **"Was action permitted to execute?"** | **✅ Solved by `ccs-verifier`** |

Storage correctness does not imply behavioral compliance. A checkpoint system can be 100% storage-conformant while executing actions that violate runtime state sovereignty.

## Quick Start

```python
from langgraph.checkpoint.memory import MemorySaver
from ccs_verifier import CCSVerifierCheckpointer, Required

# Define Required(τ) rules
rules = [
    Required("permissions", contains="write_access"),
    Required("request_count", max_value=1000),
    Required("approval_status", predicate=lambda x: x == "approved")
]

# Wrap any checkpointer
inner = MemorySaver()
ccs_saver = CCSVerifierCheckpointer(inner, rules)

# All writes are validated
config = {"thread_id": "agent-001"}
checkpoint = {"channel_values": {...}}
ccs_saver.put(config, checkpoint, metadata, writes)
# ↑ Raises CCSConformanceError if Required(τ) not satisfied
```

## Performance

| Metric | Value |
|--------|-------|
| Average validation latency | **1.47 µs per validation** |
| Throughput | **681,338 validations/sec** |
| Overhead | <0.01% of checkpoint write time |

## Architecture

**Decorator Pattern** — wraps `BaseCheckpointSaver`, intercepts `put()` to validate Required(τ) before write. Zero modification to LangGraph core.

```
┌─────────────────────────────────────────┐
│         CCSVerifierCheckpointer         │
│                                         │
│  put() → validate Required(τ) → write   │
│    ↓              ↓              ↓      │
│  [Rules]    [DecisionLog]   [Inner]    │
└─────────────────────────────────────────┘
```

## Test Results

**7/7 functional tests passed:**

| Test | Required(τ) | Expected | Result |
|------|-------------|----------|--------|
| 1 | `contains="read_data"` | ALLOWED | ✅ |
| 2 | `contains="read_data"` (missing) | REJECTED | ✅ |
| 3 | `max_value=1000` (exceeded) | REJECTED | ✅ |
| 4 | `predicate=lambda x: x=="approved"` | ALLOWED | ✅ |
| 5 | `predicate=lambda x: x=="approved"` (denied) | REJECTED | ✅ |
| 6 | Multi-constraint (all satisfied) | ALLOWED | ✅ |
| 7 | Multi-constraint (permission missing) | REJECTED | ✅ |

**Integration Test:**
- **Case A (Compliant):** ✅ Write succeeded
- **Case B (Illegal):** ❌ BLOCKED — `CCSConformanceError` raised
- **Inner MemorySaver writes:** 1 (only compliant writes passed)

## Use Cases

| Use Case | Required(τ) Rule |
|----------|------------------|
| Permission enforcement | `Required("permissions", contains="write_access")` |
| Rate limiting | `Required("request_count", max_value=1000)` |
| Human approval gate | `Required("approval_status", predicate=lambda x: x=="approved")` |
| Compliance rules | `Required("data_classification", equals="public")` |

## References

- **CCS 1.0 Specification:** https://github.com/Correctover/ccs-integration-kit/releases/tag/v1.0.0
- **DOI:** 10.5281/zenodo.21234580
- **LangGraph Audit:** https://gist.github.com/Correctover/43b9e8d991b71921561544c2ac3d9985
- **Test Results:** https://gist.github.com/Correctover/22478198b4f6cc5101e50022469f3094

---

**This implementation is a reference implementation. Its governance logic is formally bound to CCS 1.0 specification (DOI: 10.5281/zenodo.21234580).**

**This is not merely a code merge. This is the upgrade of production AI Agents from "probabilistic execution" to "auditable state machine transitions."**
