# Ingestion Simplification Summary

**Date**: October 22, 2025  
**Version**: v3.0.0  
**Type**: Breaking Change

## Overview

Removed support for queued and inline ingestion, simplifying the library to use **streaming ingestion only**. This change reduces complexity, improves performance consistency, and provides a better developer experience.

## Changes Made

### 1. Removed Queued Ingestion

**Before (v2.x)**:
```python
async with AsyncKustoSaver.from_connection_string(
    cluster_uri=uri,
    database=db,
    ingest_mode="queued",  # or "streaming"
) as saver:
    ...
```

**After (v3.x)**:
```python
async with AsyncKustoSaver.from_connection_string(
    cluster_uri=uri,
    database=db,
    # Always uses streaming ingestion
) as saver:
    ...
```

### 2. Removed Inline Ingestion

Previously, the library supported inline KQL ingestion using `.ingest inline` commands:
- `INSERT_CHECKPOINT_KQL` template
- `INSERT_CHECKPOINT_WRITES_KQL` template
- `_format_kql_record()` method for CSV formatting
- `_format_kql_records()` method for batch CSV formatting

These have been completely removed. All ingestion now uses the `StreamingIngestClient` with JSON format.

### 3. Code Changes

#### `base.py`
- ❌ Removed `INSERT_CHECKPOINT_KQL` constant
- ❌ Removed `INSERT_CHECKPOINT_WRITES_KQL` constant
- ❌ Removed `_format_kql_record()` method
- ❌ Removed `_format_kql_records()` method

#### `aio.py`
- ❌ Removed `ingest_mode` parameter from `__init__()`
- ❌ Removed `ingest_mode` parameter from `from_connection_string()`
- ❌ Removed `self.ingest_mode` attribute
- ✅ Changed `ingest_client` type from `Union[AsyncQueuedIngestClient, AsyncStreamingIngestClient]` to `AsyncStreamingIngestClient`
- ✅ Removed conditional client creation (always creates `AsyncStreamingIngestClient`)
- ✅ Updated docstrings to reflect streaming-only behavior

#### `_ainternal.py`
- ❌ Removed `AsyncQueuedIngestClient` import
- ✅ Changed `AsyncKustoIngestConn` type alias from `Union[...]` to `AsyncStreamingIngestClient`
- ✅ Updated `get_ingest_client()` signature to use concrete type

#### `_internal.py`
- ❌ Removed `QueuedIngestClient` import  
- ✅ Changed `KustoIngestConn` type alias from `Union[...]` to `StreamingIngestClient`
- ✅ Updated `get_ingest_client()` signature to use concrete type

#### Documentation
- ✅ Updated `README.md` to remove queued mode references
- ✅ Updated architecture diagram
- ✅ Removed "Ingestion Modes" comparison table
- ✅ Updated performance tuning section
- ✅ Updated troubleshooting section
- ✅ Added v3.0.0 entry to `CHANGELOG.md`

#### Examples & Tests
- ✅ Updated `examples/basic_usage.py` to remove `ingest_mode` parameter
- ✅ Updated `tests/test_async.py` to remove `ingest_mode` parameter

## Benefits

### 1. Simpler API
- **No configuration complexity**: Developers don't need to choose between ingestion modes
- **Fewer parameters**: Removed `ingest_mode` from all constructors
- **Clear defaults**: Streaming is always used, no decision paralysis

### 2. Consistent Performance
- **Predictable latency**: Data always available within <1 second after flushing
- **No mode-specific behavior**: All code paths use the same ingestion mechanism
- **Better for interactive workloads**: Streaming is ideal for LangGraph's use case

### 3. Reduced Complexity
- **Smaller codebase**: Removed ~50 lines of inline ingestion formatting code
- **Fewer dependencies**: No need for `QueuedIngestClient`
- **Simpler testing**: Only one ingestion path to test

### 4. Better Performance
- **Lower latency**: <1 second vs 2-5 minutes for queued mode
- **Efficient format**: JSON ingestion via streaming client is optimized
- **No CSV overhead**: Eliminated CSV formatting and escaping logic

## Migration Guide

### Code Changes Required

1. **Remove `ingest_mode` parameter**:
   ```python
   # Before
   AsyncKustoSaver.from_connection_string(..., ingest_mode="streaming")
   
   # After
   AsyncKustoSaver.from_connection_string(...)  # No ingest_mode
   ```

2. **Update flush expectations**:
   ```python
   # Before (queued mode)
   await saver.flush()
   # Wait 2-5 minutes for data to appear
   
   # After (streaming mode)
   await saver.flush()
   # Data available within <1 second
   ```

3. **No configuration changes needed for**:
   - `batch_size` - still controls batching behavior
   - `flush_interval` - still controls auto-flush timing
   - All other parameters unchanged

### Breaking Changes

| Component | Change | Migration |
|-----------|--------|-----------|
| `AsyncKustoSaver.__init__()` | Removed `ingest_mode` param | Delete parameter |
| `AsyncKustoSaver.from_connection_string()` | Removed `ingest_mode` param | Delete parameter |
| `BaseKustoSaver.INSERT_CHECKPOINT_KQL` | Removed constant | Not publicly used |
| `BaseKustoSaver.INSERT_CHECKPOINT_WRITES_KQL` | Removed constant | Not publicly used |
| `BaseKustoSaver._format_kql_record()` | Removed method | Not publicly used |
| `BaseKustoSaver._format_kql_records()` | Removed method | Not publicly used |

### No Breaking Changes For

- ✅ Database schema (no DDL changes)
- ✅ Query behavior (reads unchanged)
- ✅ Checkpoint data format (serialization unchanged)
- ✅ Async/sync API structure

## Performance Impact

### Latency Improvement

| Operation | Before (Queued) | After (Streaming) | Change |
|-----------|----------------|-------------------|--------|
| **Flush → Available** | 2-5 minutes | <1 second | **99%+ faster** |
| **Write Throughput** | High | Medium-High | Similar |
| **Query Performance** | Unchanged | Unchanged | No change |

### Resource Usage

- **Memory**: Slightly lower (no queued client buffering)
- **Network**: Similar (both use batching)
- **Storage**: Identical (same data format)

## Compatibility

### Supported Platforms
- ✅ Python 3.10+
- ✅ Azure Data Explorer (Kusto)
- ✅ All Azure regions

### Dependencies
- ✅ `azure-kusto-data` (unchanged)
- ✅ `azure-kusto-ingest` (unchanged, just using StreamingIngestClient)
- ✅ `langgraph` (unchanged)

## Testing

All existing tests pass with streaming ingestion:
- ✅ Checkpoint save/load
- ✅ Thread isolation
- ✅ Metadata filtering
- ✅ List with pagination
- ✅ Delete operations
- ✅ Concurrent access

## Rollback Plan

If you need to revert to v2.x with queued ingestion support:

1. Pin to previous version:
   ```bash
   pip install langgraph-checkpoint-kusto==2.0.0
   ```

2. Restore `ingest_mode` parameter in your code:
   ```python
   async with AsyncKustoSaver.from_connection_string(
       ...,
       ingest_mode="queued",  # v2.x only
   ) as saver:
       ...
   ```

## Conclusion

The removal of queued and inline ingestion simplifies the library significantly while providing better performance for the typical LangGraph use case. Streaming ingestion's <1 second latency is ideal for interactive agent applications, and the simpler API reduces configuration overhead.

**Recommended Action**: Upgrade to v3.0.0 and remove the `ingest_mode` parameter from your code. You'll get better performance and a simpler API.
