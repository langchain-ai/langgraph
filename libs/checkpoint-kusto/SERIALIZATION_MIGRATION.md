# Serialization Migration to JsonPlusSerializer

## Overview

This document describes the complete migration from a hybrid orjson/serde serialization approach to using `JsonPlusSerializer` exclusively throughout the Kusto checkpointer.

## Migration Date

Completed: 2024 (v3.0.0)

## Rationale

**Why migrate to JsonPlusSerializer?**

1. **Better LangChain Integration**: JsonPlusSerializer is designed specifically to handle LangChain objects (messages, documents, etc.)
2. **Consistency**: Single serialization interface throughout the codebase
3. **Type Preservation**: Automatic handling of complex types without manual encoding
4. **Maintainability**: Fewer lines of code, less conditional logic
5. **Future-Proof**: Maintained by the LangGraph team with ongoing improvements

## Changes Made

### 1. Import Changes

**File**: `base.py`
```python
# Added:
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
```

**File**: `aio.py`
```python
# Added:
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
```

### 2. Default Serializer

**File**: `aio.py`
```python
def __init__(
    self,
    query_client: AsyncKustoClient,
    ingest_client: AsyncStreamingIngestClient,
    database: str,
    *,
    batch_size: int = 100,
    flush_interval: float = 30.0,
    serde: SerializerProtocol | None = None,
) -> None:
    # Changed from: super().__init__(serde=serde)
    super().__init__(serde=serde or JsonPlusSerializer())
```

### 3. Method Updates

All serialization methods in `base.py` were updated:

#### `_migrate_pending_sends()` (lines ~119-149)
```python
# BEFORE:
pending_sends_str = orjson.dumps(tasks).decode('utf-8')

# AFTER:
_, blob = self.serde.dumps_typed(tasks)
pending_sends_str = blob.decode('utf-8')
```

#### `_load_blobs()` (lines ~151-179)
```python
# BEFORE (27 lines):
try:
    value = orjson.loads(value_str)
except orjson.JSONDecodeError:
    import base64
    blob = base64.b64decode(value_str.encode('ascii'))
    value = self.serde.loads_typed((type_str, blob))

# AFTER (11 lines):
blob = value_str.encode('utf-8')
value = self.serde.loads_typed((type_str, blob))
```

#### `_dump_blobs()` (lines ~181-221)
```python
# BEFORE (44 lines):
try:
    value_str = orjson.dumps(value).decode('utf-8')
    type_str = "json"
except TypeError:
    type_str, blob = self.serde.dumps_typed(value)
    if isinstance(blob, bytes):
        import base64
        value_str = base64.b64encode(blob).decode('ascii')

# AFTER (25 lines):
type_str, blob = self.serde.dumps_typed(value)
if isinstance(blob, bytes):
    value_str = blob.decode('utf-8')
else:
    value_str = blob
```

#### `_load_writes()` (lines ~223-251)
```python
# BEFORE (30 lines):
try:
    value = orjson.loads(value_str)
except orjson.JSONDecodeError:
    import base64
    blob = base64.b64decode(value_str.encode('ascii'))
    value = self.serde.loads_typed((type_str, blob))

# AFTER (19 lines):
blob = value_str.encode('utf-8')
value = self.serde.loads_typed((type_str, blob))
```

#### `_dump_writes()` (lines ~253-290)
```python
# BEFORE (46 lines):
try:
    value_str = orjson.dumps(value).decode('utf-8')
    type_str = "json"
except TypeError:
    type_str, blob = self.serde.dumps_typed(value)
    if isinstance(blob, bytes):
        import base64
        value_str = base64.b64encode(blob).decode('ascii')

# AFTER (33 lines):
type_str, blob = self.serde.dumps_typed(value)
if isinstance(blob, bytes):
    value_str = blob.decode('utf-8')
else:
    value_str = blob
```

## Impact Summary

### Code Reduction
- **Before**: ~173 lines of serialization logic
- **After**: ~88 lines of serialization logic
- **Reduction**: ~49% fewer lines

### Complexity Reduction
- ❌ Removed: try/except blocks for orjson failures
- ❌ Removed: base64 encoding/decoding logic
- ❌ Removed: conditional logic for JSON vs binary
- ✅ Added: consistent `self.serde` interface

### Benefits
1. **Simpler Code**: Single code path for all data types
2. **Better Error Messages**: JsonPlusSerializer provides clear error messages
3. **Type Safety**: Automatic type handling without manual checks
4. **LangChain Support**: Native support for Message objects, Documents, etc.
5. **Maintainability**: Easier to understand and modify

## Migration Pattern

The consistent pattern applied across all methods:

### For Serialization (dump):
```python
# Step 1: Use serde to serialize
type_str, blob = self.serde.dumps_typed(value)

# Step 2: Convert bytes to string for Kusto storage
if isinstance(blob, bytes):
    value_str = blob.decode('utf-8')
else:
    value_str = blob
```

### For Deserialization (load):
```python
# Step 1: Convert string from Kusto to bytes
blob = value_str.encode('utf-8')

# Step 2: Use serde to deserialize
value = self.serde.loads_typed((type_str, blob))
```

## Backward Compatibility

**Breaking Change**: This is a breaking change for v3.0.0

- Existing checkpoints stored with the old hybrid approach may not deserialize correctly
- Users should migrate their data or continue using v2.x.x
- The new approach is forward-compatible with all LangChain types

## Testing Recommendations

1. **Basic Serialization**: Test with simple types (str, int, dict, list)
2. **LangChain Messages**: Test with HumanMessage, AIMessage, SystemMessage
3. **Complex Objects**: Test with nested structures, custom types
4. **Round-Trip**: Ensure data survives serialize → store → load → deserialize
5. **Memory**: Verify checkpoint memory works across sessions

## Custom Serializers

Users can still provide custom serializers:

```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.kusto import AsyncKustoSaver

# Option 1: Use default JsonPlusSerializer
checkpointer = AsyncKustoSaver(...)

# Option 2: Use custom serializer
custom_serde = MyCustomSerializer()
checkpointer = AsyncKustoSaver(..., serde=custom_serde)
```

## Related Files

- `libs/checkpoint-kusto/langgraph/checkpoint/kusto/base.py` - Base serialization logic
- `libs/checkpoint-kusto/langgraph/checkpoint/kusto/aio.py` - Async implementation
- `libs/checkpoint-kusto/tutorial_04_openai_chatbot.py` - Example usage with LangChain messages

## References

- LangGraph JsonPlusSerializer: `langgraph.checkpoint.serde.jsonplus`
- LangGraph Serialization: https://langchain-ai.github.io/langgraph/concepts/persistence/
- Tutorial 04: OpenAI Chatbot with Memory (demonstrates JsonPlusSerializer with real LLM)

## Rollback Plan

If issues arise:
1. Revert to v2.x.x which uses the hybrid approach
2. Or use a custom serializer that mimics the old behavior
3. File an issue on GitHub with reproduction steps

## Conclusion

The migration to JsonPlusSerializer provides a cleaner, more maintainable codebase with better support for LangChain objects. The reduced complexity makes the code easier to understand and debug, while the LangGraph-native serializer ensures compatibility with future LangChain updates.
