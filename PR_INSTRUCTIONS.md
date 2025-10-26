# PR Creation Instructions

## ✅ COMPLETED: Pushed to GitHub

Your changes are now on GitHub:
- **Repository**: https://github.com/vaishnavigavi/langgraph
- **Branch**: `fix-remote-graph-context-support`
- **Commit**: 4868809e

---

## Next: Create Pull Requests

You need **2 PRs** for **2 different repositories**:

---

## PR #1: Client Tests (langgraph) ⬅️ START HERE

### Create the PR
The browser should have opened automatically. If not:
https://github.com/vaishnavigavi/langgraph/compare/fix-remote-graph-context-support?expand=1

### PR Details

**Base repository**: `langchain-ai/langgraph`  
**Base branch**: `main`  
**Head repository**: `vaishnavigavi/langgraph`  
**Compare branch**: `fix-remote-graph-context-support`

**Title**:
```
test: add RemoteGraph tests for context and config together
```

**Description**:
```markdown
## Overview
Adds comprehensive client-side tests for RemoteGraph that verify the client can properly send both `context` and `config` parameters together.

## What's Included
- ✅ Client tests with mock server (hermetic testing)
- ✅ Tests for invoke, ainvoke, stream with both parameters
- ✅ Middleware context capture validation
- ✅ Memory persistence tests across multiple scenarios
- ✅ Backward compatibility tests (config-only, context-only)

## Files Changed
- `tests/remote/test_remote_context_and_config.py` (567 lines)
- `PATCH_FOR_UPSTREAM.diff` (server fix for langgraph-api)
- `PR_SUMMARY.md`, `UPSTREAM_PR_READY.md` (documentation)

## Testing
```bash
pytest tests/remote/test_remote_context_and_config.py -v
# All 7 tests pass with mock server
```

## Note
This PR contains **client-side tests only**. The corresponding **server-side fix** needs to be applied to the `langgraph-api` repository. The server fix is documented in `PATCH_FOR_UPSTREAM.diff` and `UPSTREAM_PR_READY.md`.

## Verification
All acceptance criteria verified:
- ✅ RemoteGraph sends both context and config without error
- ✅ Middleware receives context values (user_id, request_id)
- ✅ Memory persists with thread_id from config, context, or both
- ✅ No regression for config-only or context-only calls

## Related
- Server-side fix will be submitted to https://github.com/langchain-ai/langgraph-api
- QA verification: All HTTP 200 responses with real server (after server fix applied)
```

---

## PR #2: Server Fix (langgraph-api) ⬅️ DO SECOND

### If You Have Access to langchain-ai/langgraph-api

```bash
# 1. Clone the repo
cd ~/Desktop
git clone https://github.com/langchain-ai/langgraph-api
cd langgraph-api

# 2. Create branch
git checkout -b fix/accept-both-context-and-config

# 3. Apply the server fix
git apply ~/Desktop/langgraph/PATCH_FOR_UPSTREAM.diff

# 4. Add test file
cp ~/Desktop/langgraph/test_server_project/test_context_and_config_e2e.py tests/

# 5. Commit and push
git add -A
git commit -m "fix: server accept both context and config for remote runs"
git push origin fix/accept-both-context-and-config

# 6. Create PR
open "https://github.com/langchain-ai/langgraph-api/compare/main...fix/accept-both-context-and-config?expand=1"
```

**Use this PR description**: Copy from `PR_SUMMARY.md` (already created)

### If You DON'T Have Access

Create an issue instead:
1. Go to: https://github.com/langchain-ai/langgraph-api/issues/new
2. Title: `[Fix Request] Server reject requests with both context and config`
3. Description: Paste contents of `UPSTREAM_PR_READY.md`
4. Attach: `PATCH_FOR_UPSTREAM.diff`

---

## Summary

### What You Just Pushed ✅
- Client-side tests proving RemoteGraph can send both parameters
- Complete documentation for the server fix
- Patch file ready to apply to langgraph-api

### What Happens Next
1. **PR #1** (client tests) → Goes to `langchain-ai/langgraph`
2. **PR #2** (server fix) → Goes to `langchain-ai/langgraph-api`

### Why Two PRs?
- **langgraph**: Client library (RemoteGraph) - already works, just adding tests
- **langgraph-api**: Server that runs graphs - needs the fix to accept both params

---

## Quick Reference

**Your fork**: https://github.com/vaishnavigavi/langgraph  
**Your branch**: fix-remote-graph-context-support  
**Upstream langgraph**: https://github.com/langchain-ai/langgraph  
**Upstream langgraph-api**: https://github.com/langchain-ai/langgraph-api

**Files for server fix**:
- `PATCH_FOR_UPSTREAM.diff` - The code change
- `test_server_project/test_context_and_config_e2e.py` - Tests for server
- `PR_SUMMARY.md` - Complete PR description
- `UPSTREAM_PR_READY.md` - Deployment guide

---

## Troubleshooting

**Q: Can't see the "Create PR" button?**  
A: Make sure you're comparing `vaishnavigavi:fix-remote-graph-context-support` → `langchain-ai:main`

**Q: Don't have access to langgraph-api repo?**  
A: That's normal! Create an issue instead and share the patch file.

**Q: Should I wait for PR #1 to merge before creating PR #2?**  
A: No, but mention both PRs in each other's descriptions so reviewers know they're related.

---

Good luck! 🚀

