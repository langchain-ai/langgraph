# PR: Fix server to accept both context and config for remote runs

## Overview

This PR removes the mutual exclusivity validation that prevented requests from including both `context` and `config` parameters. The server now accepts both, keeps them separate, and forwards them to the runtime.

## Problem

Previously, the LangGraph API server rejected requests with HTTP 400 when both `context` and `config.configurable` were present:

```
Cannot specify both configurable and context. Prefer setting context alone. 
Context was introduced in LangGraph 0.6.0 and is the long term planned 
replacement for configurable.
```

This limitation prevented users from:
- Migrating gradually from `config` to `context`
- Using both parameters when beneficial (e.g., thread_id in config, user metadata in context)
- Composing graphs that need both parameter types

## Solution

### Core Change

**File**: `langgraph_api/models/run.py`  
**Function**: `create_valid_run`  
**Lines**: ~218-237

**Before** (the problematic validation):
```python
config = payload.get("config") or {}
context = payload.get("context") or {}
configurable = config.setdefault("configurable", {})

if configurable and context:
    raise HTTPException(
        status_code=400,
        detail="Cannot specify both configurable and context. Prefer setting context alone. Context was introduced in LangGraph 0.6.0 and is the long term planned replacement for configurable.",
    )

# Keep config and context in sync for user provided params
if context:
    configurable = context.copy()
    config["configurable"] = configurable
else:
    context = configurable.copy()
```

**After** (minimal fix that accepts both):
```python
config = payload.get("config") or {}
context = payload.get("context") or {}
configurable = config.setdefault("configurable", {})

# Keep config and context in sync for user provided params
# Fill configurable from context when only context is provided (backward compat)
# Fill context from configurable when only config is provided (backward compat)
# When both provided, keep them separate and forward both to runtime
if context and not configurable:
    # Only context provided - copy to configurable for backward compatibility
    configurable.update(context)
elif configurable and not context:
    # Only config provided - copy to context for backward compatibility  
    context = configurable.copy()
elif context and configurable:
    # Both provided - keep them separate, but ensure thread_id consistency
    # If thread_id is in context but not in configurable, copy it over
    if "thread_id" in context and "thread_id" not in configurable:
        configurable["thread_id"] = context["thread_id"]
```

### Key Changes

1. **Removed** the `HTTPException` that rejected both parameters
2. **Added** logic to handle three cases:
   - **Context only**: Copy to configurable (existing behavior preserved)
   - **Config only**: Copy to context (existing behavior preserved)  
   - **Both provided**: Keep separate, optionally sync thread_id
3. **Preserved** all existing behavior for single-parameter calls
4. **Minimal** change - no refactoring of unrelated code

### Behavior

| Scenario | Before | After | Notes |
|----------|---------|-------|-------|
| Both context + config | ❌ HTTP 400 | ✅ HTTP 200 | **Fixed** - now accepted |
| Config only | ✅ HTTP 200 | ✅ HTTP 200 | **No change** - backward compatible |
| Context only | ✅ HTTP 200 | ✅ HTTP 200 | **No change** - backward compatible |
| Neither | ✅ Works | ✅ Works | **No change** |

### Thread ID Handling

When both `context` and `config` are provided:
- If `context.thread_id` exists but `config.configurable.thread_id` doesn't, copy it over
- This ensures thread continuity regardless of which parameter contains the ID
- Users can put thread_id in either location

## Testing

### Test File

**New file**: `tests/test_context_and_config_e2e.py`

Contains 6 tests:
1. `test_invoke_both_context_and_config` - Verify /runs/wait accepts both
2. `test_stream_both_context_and_config` - Verify /runs/stream accepts both
3. `test_config_only_still_works` - Regression test for config-only
4. `test_context_only_still_works` - Regression test for context-only
5. `test_thread_id_from_context_to_config` - Verify thread_id mapping
6. `test_memory_persists_with_both_parameters` - Verify memory/checkpointing works

### Manual Verification

Tested with real LangGraph server (`langgraph dev --port 2024`):

```bash
# Test 1: Both parameters on /runs/wait
curl -X POST http://127.0.0.1:2024/runs/wait \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id":"agent",
    "input":{"messages":[{"role":"user","content":"Hello"}]},
    "config":{"configurable":{"thread_id":"t1"}},
    "context":{"user_id":"u1"}
  }'
# Result: HTTP 200 ✅

# Test 2: Both parameters on /runs/stream  
curl -X POST http://127.0.0.1:2024/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id":"agent",
    "input":{"messages":[{"role":"user","content":"Hello"}]},
    "config":{"configurable":{"thread_id":"t2"}},
    "context":{"user_id":"u2"}
  }'
# Result: HTTP 200 ✅ (SSE stream)

# Test 3: Config only (regression)
curl -X POST http://127.0.0.1:2024/runs/wait \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id":"agent",
    "input":{"messages":[{"role":"user","content":"Hello"}]},
    "config":{"configurable":{"thread_id":"t3"}}
  }'
# Result: HTTP 200 ✅

# Test 4: Context only (regression)
curl -X POST http://127.0.0.1:2024/runs/wait \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id":"agent",
    "input":{"messages":[{"role":"user","content":"Hello"}]},
    "context":{"thread_id":"t4"}
  }'
# Result: HTTP 200 ✅
```

### All Tests Pass ✅

```
=== VALIDATION SUMMARY ===

Test 1: Both context + config on /runs/wait
Status: 200, Response: OK

Test 2: Both context + config on /runs/stream
Status: 200, Response: OK (SSE events)

Test 3: Config only
Status: 200, Response: OK

Test 4: Context only
Status: 200, Response: OK
```

## Impact Analysis

### Breaking Changes
**None** - This is purely additive functionality that removes a restriction.

### Backward Compatibility
**Full** - All existing API calls work exactly as before:
- Config-only calls: ✅ Unchanged behavior
- Context-only calls: ✅ Unchanged behavior
- No parameters: ✅ Unchanged behavior

### Performance
**No impact** - Logic is O(1) dictionary operations, same as before.

### Security
**No impact** - No new attack vectors. Parameters still validated by existing schemas.

## Migration Guide

### For Users Currently Using Config Only

No action needed. Your code continues to work as-is.

```python
# This still works exactly the same
remote_graph.invoke(
    input={"messages": [...]},
    config={"configurable": {"thread_id": "my-thread"}}
)
```

### For Users Currently Using Context Only

No action needed. Your code continues to work as-is.

```python
# This still works exactly the same
remote_graph.invoke(
    input={"messages": [...]},
    context={"thread_id": "my-thread", "user_id": "user123"}
)
```

### For Users Who Want to Use Both

You can now pass both parameters:

```python
# This now works! Previously returned HTTP 400
remote_graph.invoke(
    input={"messages": [...]},
    config={"configurable": {"thread_id": "my-thread"}},
    context={"user_id": "user123", "request_id": "req456"}
)
```

**Benefits:**
- Gradual migration from config to context
- Separate concerns (thread management vs. request metadata)
- Compose graphs that need both parameter types
- Better middleware support (context can carry request-scoped data)

## Files Changed

### Modified
- `langgraph_api/models/run.py` - Removed mutual exclusivity check

### Added  
- `tests/test_context_and_config_e2e.py` - Comprehensive test coverage

## Checklist

- [x] Removed mutual exclusivity validation
- [x] Added logic to handle both parameters together
- [x] Preserved backward compatibility for single-parameter calls
- [x] Added comprehensive test coverage (6 tests)
- [x] Verified with manual curl tests against real server
- [x] All tests pass (HTTP 200 on all scenarios)
- [x] No breaking changes
- [x] Documentation updated in code comments

## Related Issues

This fixes the issue where RemoteGraph client calls that include both `context` and `config` were rejected by the server with HTTP 400.

**Before:**
```python
# This failed with HTTP 400
remote_graph.invoke(
    input=data,
    config={"configurable": {"thread_id": "t1"}},
    context={"user_id": "u1"}
)
# Error: "Cannot specify both configurable and context"
```

**After:**
```python
# This now succeeds with HTTP 200
remote_graph.invoke(
    input=data,
    config={"configurable": {"thread_id": "t1"}},
    context={"user_id": "u1"}
)
# Works! Both parameters accepted and forwarded to runtime
```

## Deployment Notes

- **Zero downtime**: Change is backward compatible
- **No database migrations** required
- **No environment variable changes** required
- **Client updates**: Optional - existing clients work without changes

## Verification Steps for Reviewers

1. **Review the code change**:
   - Check `langgraph_api/models/run.py` lines ~218-237
   - Verify mutual exclusivity check was removed
   - Verify both parameters are now handled

2. **Run the tests**:
   ```bash
   pytest tests/test_context_and_config_e2e.py -v
   ```

3. **Manual testing** (optional):
   ```bash
   langgraph dev --port 2024
   # Then run the 4 curl commands shown above
   ```

4. **Verify backward compatibility**:
   - Existing tests should still pass
   - No changes to API response shapes
   - Config-only and context-only still work

## Questions or Concerns?

Please review and provide feedback. This is a minimal, safe change that:
- ✅ Removes a limitation
- ✅ Adds no new complexity
- ✅ Maintains full backward compatibility
- ✅ Has comprehensive test coverage
- ✅ Verified working with real server

---

**PR Title**: `fix: server accept both context and config for remote runs`

**PR Labels**: `bug`, `server`, `api`, `backward-compatible`

