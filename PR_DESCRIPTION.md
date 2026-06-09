# PR Description: Production-Ready Human-in-the-Loop (HITL) Component

## Overview
This PR introduces a high-level Human Approval Workflow component to LangGraph, making it easier to build enterprise AI workflows that require human oversight.

## Changes
### 1. `ApprovalNode` (`libs/prebuilt`)
- A new reusable node that simplifies the "interrupt and resume" pattern.
- Supports `approve`, `reject`, and `modify` actions out of the box.
- Fires structured events (`ApprovalRequested`, `ApprovalOutcome`).

### 2. `Pregel` Enhancements (`libs/langgraph`)
- Added `graph.pause(config)`: Gracefully stops a running thread after the current step.
- Added `graph.resume(config, value)`: Convenience method for resuming interrupted threads.
- Updated `RunControl` to support weak references for thread tracking.

### 3. Documentation & Examples
- Added `APPROVAL.md` with architecture and usage details.
- Added three real-world examples: Financial Transaction, Customer Support, and Document Processing.
- Added performance benchmarks for pause/resume latency.

## Why this is needed
Currently, users have to implement manual interrupt logic using `interrupt()` and `Command(resume=...)` repeatedly. The `ApprovalNode` provides a standard, extensible way to handle these scenarios.

## Verification
- Added comprehensive tests in `libs/prebuilt/tests/test_approval.py`.
- Verified 100% pass rate on new tests.
- Benchmarked resume latency: ~5.5ms (vs 11.8ms baseline invoke).

## Checklist
- [x] Python 3.11+
- [x] Full type hints
- [x] Async support (`aresume`)
- [x] Pydantic models (v2 compatible)
- [x] No breaking changes
- [x] Documentation included
