# LangGraph Kusto Checkpointer - Implementation Plan

## Overview
Build a production-quality Azure Data Explorer (Kusto/ADX) backed checkpointer for LangGraph that replicates the behavior and contracts of the official Postgres checkpointer.

## Reference Implementation
- **Source**: `libs/checkpoint-postgres/`
- **Key Files**:
  - `base.py`: Core SQL queries, migrations, and base class
  - `aio.py`: Async implementation with connection management
  - `_internal.py`: Connection utilities
  - Tests: `test_sync.py`, `test_async.py`

## Architecture Decision Records

### ADR-1: Kusto as Append-Only Store with Materialized Views
**Decision**: Treat Kusto as an append-only datastore with "latest-wins" semantics via materialized views using `arg_max()` aggregation.

**Rationale**: 
- Kusto excels at append operations; updates are expensive
- Materialized views with `arg_max(checkpoint_id, *)` pre-compute the latest checkpoint per thread
- ~10-100x faster than `ORDER BY checkpoint_id DESC | TAKE 1` on large datasets
- O(1) index lookup vs O(n log n) full table scan
- View updates automatically in near real-time (seconds to minutes)

**Implementation**:
- Create `LatestCheckpoints` materialized view in `provision.kql`
- Use materialized view for `aget_tuple()` when no specific checkpoint_id provided
- Fall back to base `Checkpoints` table for specific ID queries and listing operations

### ADR-2: Dual Client Pattern
**Decision**: Use separate clients for query (`KustoClient`) and ingest (`QueuedIngestClient`/`StreamingIngestClient`).
**Rationale**: Aligns with Kusto best practices; queued ingest for reliability, optional streaming for low-latency scenarios.

### ADR-3: Batch Writes with Configurable Flush
**Decision**: Buffer writes and flush in configurable batches (default: immediate for sync, batched for async).
**Rationale**: Optimize for Kusto's ingestion pipeline while maintaining checkpoint semantics.

## Milestones & Task Breakdown

### M1: Project Setup & Schema (2-3 hours)
**Tasks**:
1. Create package structure (`libs/checkpoint-kusto/`)
2. Setup `pyproject.toml` with dependencies:
   - `azure-kusto-data>=4.3.1`
   - `azure-kusto-ingest>=4.3.1`
   - `langgraph-checkpoint>=2.1.2,<4.0.0`
   - `orjson>=3.10.1`
3. Create `provision.kql` with:
   - Checkpoints table schema
   - CheckpointWrites table schema
   - Retention policies (default: 90 days)
   - Cache policies for query optimization
4. Add `py.typed` for type checking

**Deliverables**: Package skeleton, Kusto DDL script

**Risks**: None (straightforward setup)

---

### M2: Base Implementation - Core Data Layer (6-8 hours)
**Tasks**:
1. Implement `base.py`:
   - `BaseKustoSaver` extending `BaseCheckpointSaver[str]`
   - KQL query templates (SELECT, INSERT equivalents)
   - `_load_blobs`, `_dump_blobs` (mirror Postgres logic)
   - `_load_writes`, `_dump_writes`
   - `_search_where` for filter construction
   - `get_next_version` (reuse Postgres logic)
2. Connection management utilities in `_internal.py`:
   - `KustoConn` type alias
   - Context manager for client lifecycle
3. Serialization validation:
   - Ensure JSON+ serde compatibility
   - Round-trip tests for complex types

**Deliverables**: Base class with all helper methods

**Risks**: 
- KQL syntax differences from SQL (mitigation: extensive testing)
- Serialization edge cases (mitigation: comprehensive unit tests)

---

### M3: Async Implementation (8-10 hours)
**Tasks**:
1. Implement `aio.py`:
   - `AsyncKustoSaver` class
   - `asetup()`: Health check + table validation
   - `aget_tuple()`: Query single checkpoint (latest or by ID)
   - `alist()`: Query with filters, pagination, ordering
   - `aput()`: Insert checkpoint + blobs
   - `aput_writes()`: Insert checkpoint writes
   - `adelete_thread()`: Delete all data for thread_id
   - Context manager support (`from_connection_string`)
2. Implement batching logic:
   - Configurable batch size (default: 100)
   - Automatic flush on batch size
   - Manual `flush()` method
3. Lock management:
   - Use `asyncio.Lock` for async safety
   - Thread-safe operation

**Deliverables**: Fully functional async checkpointer

**Risks**:
- Kusto ingestion latency (mitigation: streaming ingest option + docs)
- Async/await complexity (mitigation: thorough testing)

---

### M4: Sync Implementation & Compatibility Layer (4-6 hours)
**Tasks**:
1. Implement sync wrapper pattern (mirror Postgres):
   - `list()`: Wrapper around `alist()`
   - `get_tuple()`: Wrapper around `aget_tuple()`
   - `put()`: Wrapper around `aput()`
   - `put_writes()`: Wrapper around `aput_writes()`
   - `delete_thread()`: Wrapper around `adelete_thread()`
2. Thread-safety validation
3. Event loop management for sync calls

**Deliverables**: Sync API matching Postgres interface

**Risks**: 
- Event loop conflicts (mitigation: thread-local loop detection)

---

### M5: Instrumentation & Observability (3-4 hours)
**Tasks**:
1. Structured logging:
   - Log all query/ingest operations
   - Include thread_id, checkpoint_id in log context
   - Use JSON formatter for production
2. Metrics (using Python `logging` + OpenTelemetry hooks):
   - Counters: `puts_total`, `aputs_total`, `get_tuple_calls`, `list_calls`
   - Histograms: `get_tuple_duration_seconds`, `list_duration_seconds`
   - Gauges: `batch_queue_size`, `ingestion_lag_seconds`
3. Health check method:
   - Query Kusto system tables
   - Validate schema
   - Return status dict

**Deliverables**: Comprehensive logging and metrics

**Risks**: Performance overhead (mitigation: configurable verbosity)

---

### M6: Testing Suite (10-12 hours)
**Tasks**:
1. Unit tests (`tests/test_unit.py`):
   - Serialization round-trips
   - Query builder logic
   - Filter construction
   - Blob handling
2. Integration tests (`tests/test_sync.py`, `tests/test_async.py`):
   - Mirror Postgres test structure
   - Parameterize for different connection modes
   - Test all CRUD operations
   - Test metadata filtering
   - Test pagination and ordering
   - Test thread deletion
3. Contract tests (`tests/test_contract.py`):
   - Validate BaseCheckpointSaver interface compliance
   - Compare behavior with Postgres implementation
4. Load tests (`tests/test_load.py`):
   - Concurrent writes (50+ RPS)
   - Large checkpoint handling (>1MB)
   - Pagination performance
5. Test fixtures:
   - Mock Kusto client (for unit tests)
   - Live Kusto cluster setup (env-gated)
   - Docker Compose for local ADX emulator (if available)

**Deliverables**: >90% code coverage, all tests passing

**Risks**:
- ADX emulator limitations (mitigation: use mock + live cluster tests)
- Test environment setup complexity (mitigation: detailed docs)

---

### M7: Documentation (4-5 hours)
**Tasks**:
1. `README.md`:
   - Architecture overview with diagram
   - Installation instructions
   - Provisioning steps (run `provision.kql`)
   - Usage examples (sync + async)
   - Configuration options
   - Performance tuning guide
   - Troubleshooting section
2. `SECURITY.md`:
   - AAD/Managed Identity setup
   - Least privilege permissions (Viewer for query, Ingestor for writes)
   - Connection string security
3. API documentation:
   - Comprehensive docstrings
   - Type hints throughout
4. Examples (`examples/`):
   - Basic usage
   - Custom serialization
   - Batch optimization
   - Monitoring integration

**Deliverables**: Production-ready documentation

**Risks**: None

---

### M8: Packaging & CI/CD (2-3 hours)
**Tasks**:
1. Finalize `pyproject.toml`:
   - Version: 1.0.0
   - Dependencies locked
   - Build configuration
2. Setup `Makefile`:
   - `make format`: Run black, isort
   - `make lint`: Run ruff, mypy
   - `make test`: Run pytest
   - `make test-integration`: Run with live Kusto
3. CI configuration (if applicable):
   - GitHub Actions workflow
   - Matrix testing (Python 3.10, 3.11, 3.12)
   - Integration tests (manual trigger)

**Deliverables**: Ready-to-publish package

**Risks**: None

---

## Implementation Estimates

| Milestone | Effort | Dependencies |
|-----------|--------|--------------|
| M1: Setup | 2-3h | None |
| M2: Base | 6-8h | M1 |
| M3: Async | 8-10h | M2 |
| M4: Sync | 4-6h | M3 |
| M5: Instrumentation | 3-4h | M3, M4 |
| M6: Testing | 10-12h | M2, M3, M4 |
| M7: Documentation | 4-5h | All |
| M8: Packaging | 2-3h | All |
| **Total** | **39-51h** | |

## Key Design Decisions

### 1. Query Patterns
**Postgres**: Uses SQL with JSONB operators, array aggregations
**Kusto**: Use KQL with:
- `mv-expand` for array flattening
- `parse_json()` for JSON parsing
- `summarize make_list()` for aggregations
- `order by checkpoint_id desc | take 1` for latest

### 2. Ingestion Strategy
**Default**: Queued ingestion (5-minute lag acceptable)
**Option**: Streaming ingestion via flag (low latency, higher cost)
**Batch Size**: 100 items default, configurable

### 3. Data Model
- **Checkpoints**: Exact schema match with Postgres
- **CheckpointWrites**: Replace `blob BYTEA` with `value_json STRING`
- **No Migrations**: Kusto schema changes via `.alter table` (not in-band)

### 4. Error Handling
- Retry logic for transient Kusto errors (429, 503)
- Exponential backoff (max 3 retries)
- Structured error messages with context

## Non-Goals
- ❌ Row-level updates/deletes (use retention policies instead)
- ❌ Synchronous schema migrations (manual `.kql` scripts)
- ❌ Cross-region replication logic (use Kusto follower DBs)
- ❌ Shallow checkpointer variant (focus on full implementation first)

## Success Criteria
1. ✅ All unit tests passing (>90% coverage)
2. ✅ All integration tests passing against live Kusto
3. ✅ Contracts match Postgres checkpointer behavior
4. ✅ Documentation complete with examples
5. ✅ Load test: 50 RPS sustained for 5 minutes
6. ✅ Type checking clean (mypy strict mode)
7. ✅ Linting clean (ruff)

## STATUS Format (Updated After Each Milestone)

```
## STATUS - [DATE]

### Completed
- [x] M1: Project Setup & Schema
- [ ] M2: Base Implementation
- [ ] M3: Async Implementation
- [ ] M4: Sync Implementation
- [ ] M5: Instrumentation
- [ ] M6: Testing
- [ ] M7: Documentation
- [ ] M8: Packaging

### Current Focus
Working on: M2 - Base Implementation

### Blockers
None

### Next Steps
1. Implement base.py core methods
2. Setup KQL query templates
3. Add serialization helpers
```

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Kusto ingestion latency affects UX | Medium | Medium | Document latency expectations; offer streaming mode |
| KQL query complexity | Low | Medium | Extensive testing; query optimization guide |
| Serialization edge cases | Low | High | Comprehensive unit tests; validation suite |
| Test environment setup | High | Low | Mock clients + optional live cluster tests |
| API compatibility drift | Low | High | Contract tests; version pinning |

---

**Plan Complete. Starting Implementation Immediately.**
