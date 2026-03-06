# PR Summary: Fix time travel (replay/fork) for subgraphs

## Problem

When replaying or forking from a previous checkpoint, subgraphs could behave incorrectly:

1. **Stale `interrupt()` values in parent graphs**: On replay, cached `RESUME` writes from a prior run would be returned by `interrupt()` calls instead of re-firing the interrupt. This affected both parent and subgraphs — any graph with a cached resume value would silently reuse it rather than pausing again. For parent graphs specifically, replaying from a checkpoint after an interrupt had been resolved would skip the interrupt entirely, using the old resume value as if the user had just provided it.

2. **Stale `interrupt()` values in subgraphs**: The same cached `RESUME` problem applied to subgraphs, but without a way to propagate the "replaying" signal from parent to child, subgraphs had no way to know they should drop their cached resumes.

3. **Subgraph checkpoint misalignment**: When forking (via `update_state`) and then invoking, subgraphs with their own checkpointer could pick up checkpoints from *after* the fork point. The subgraph didn't know it was replaying from an earlier state, so it would restore a checkpoint that hadn't happened yet in the forked timeline.

## Solution

### New `CONFIG_KEY_REPLAYING` flag

A new internal config key `CONFIG_KEY_REPLAYING` (added in `_constants.py`) distinguishes **replaying** from **resuming**:

- **Resuming** (`CONFIG_KEY_RESUMING`): continuing after an interrupt, possibly with new `Command(resume=...)` values. Cached writes should be preserved.
- **Replaying** (`CONFIG_KEY_REPLAYING`): re-executing from a historical checkpoint. Cached `RESUME` writes should be dropped so interrupts re-fire.

The flag is set when a `checkpoint_id` is explicitly provided in the config (i.e., the user is pointing at a specific historical checkpoint).

### Renamed `skip_done_tasks` → `is_replaying`

The old `skip_done_tasks` boolean had inverted semantics and an unclear name. It's been renamed to `is_replaying` with straightforward `True`/`False` meaning:

- `is_replaying = True`: first tick after loading a historical checkpoint — re-run tasks, don't match cached pending writes
- `is_replaying = False`: normal execution — match cached writes, skip already-completed tasks

### Dropping stale `RESUME` writes on replay

In `_first()`, when `is_replaying` is true and the user is NOT providing a resume value (no `Command(resume=...)`, no `CONFIG_KEY_RESUMING`), all cached `RESUME` pending writes are stripped. This forces `interrupt()` calls to fire fresh.

### Subgraph checkpoint rollback

Two new methods (`_get_checkpoint_before_parent` / `_aget_checkpoint_before_parent`) handle the subgraph case:

- When entering a subgraph loop in replay mode (no explicit `checkpoint_id` for the subgraph), the code finds the subgraph checkpoint that was current *at the parent's checkpoint time* using the parent checkpoint as an upper bound.
- It returns an `empty_checkpoint()` with only `channel_values` restored — clearing `channel_versions` and `versions_seen` forces all nodes to re-trigger (otherwise the scheduling logic would see them as up-to-date and skip them).
- For forks specifically (`source=update`), the code looks through the fork's `parent_config` to find the original (pre-fork) checkpoint ID, since the fork was created after the subgraph's historical checkpoints.

### Propagating flags to subgraphs

Both `CONFIG_KEY_RESUMING` and `CONFIG_KEY_REPLAYING` are now propagated to subgraphs in `_first()`, so nested graphs also know they're in replay mode.

## Files changed

| File | What changed |
|------|-------------|
| `langgraph/_internal/_constants.py` | Added `CONFIG_KEY_REPLAYING` constant |
| `langgraph/pregel/_loop.py` | Core logic: renamed flag, replay write stripping, subgraph checkpoint rollback, flag propagation |
| `tests/test_time_travel.py` | ~1800 lines of sync tests covering replay, fork, interrupts, subgraphs |
| `tests/test_time_travel_async.py` | ~1870 lines of async equivalents |

## Key behavioral changes

- **Replay now re-fires interrupts in parent graphs** instead of silently reusing old resume values — the `RESUME` write stripping in `_first()` applies to the outermost graph too
- **Replay now re-fires interrupts in subgraphs** thanks to the new `CONFIG_KEY_REPLAYING` flag being propagated from parent to child
- **Fork + subgraph** correctly rolls back the subgraph to its state at the fork point
- **Multi-interrupt resume** still works: when providing `Command(resume=...)`, previously-resolved interrupt values are preserved (only the unresolved ones re-fire)
