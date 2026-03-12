# RESUME Writes Stripping: Complete Flow Reference

## Legend

| Column | Meaning |
|---|---|
| **Level** | P = Parent, S = Subgraph |
| **`is_replaying`** | `CONFIG_KEY_CHECKPOINT_ID` key exists in `config[CONF]` (line 249) |
| **`__enter__` via** | Which branch loads the checkpoint: **ckpt_id** (explicit checkpoint_id in checkpoint_config), **replay_state** (parent's ReplayState), **latest** (fetch most recent) |
| **`RESUMING`** | Value of `CONFIG_KEY_RESUMING` in configurable (set by parent for subgraphs, absent for outer graph) |
| **`is_resuming`** | Computed at line 633 — controls whether to "proceed past previous checkpoint" |
| **`in_map`** | `replaying_from_checkpoint_map` — subgraph's ns found in checkpoint_map |
| **Strip?** | Are RESUME pending writes stripped? (line 662-671) |

## Setup

```
Parent:  START → executor (subgraph, checkpointer=True) → END
Subgraph: START → step_a → ask_1 (interrupt) → ask_2 (interrupt) → END
```

## The Table

| # | Scenario | Level | User call | `__enter__` via | `is_replaying` | `RESUMING` | `is_resuming` | `in_map` | Strip? | Why correct |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Fresh run** | P | `invoke({"v":[]}, cfg)` | latest (None) | False | _(absent)_ | False | — | N/A | No checkpoint yet, no writes to strip |
| 1 | | S | _(Send from parent)_ | latest (None) | True¹ | False | False | False | N/A | No checkpoint yet |
| 2 | **Resume single interrupt** | P | `invoke(Cmd(resume="a"), cfg)` | latest | False | _(absent)_ | True | — | No | Resuming — keep RESUME writes for interrupt() to return answer |
| 2 | | S | _(Send)_ | latest | True¹ | True | True | False | No | `RESUMING=True` → keep. interrupt() returns "a", node completes |
| 3 | **Resume 1st of 2 interrupts** | P | `invoke(Cmd(resume="a1"), cfg)` | latest | False | _(absent)_ | True | — | No | Keep RESUME writes — ask_1's answer must survive |
| 3 | | S | _(Send)_ | latest | True¹ | True | True | False | **No** | ask_1 gets "a1" from RESUME write. ask_2 has no RESUME write → interrupt() re-fires. Correct. |
| 4 | **Replay parent ckpt** (parent was mid-subgraph) | P | `invoke(None, parent_hist_cfg)` | ckpt_id | True | _(absent)_ | True | — | **Yes** | Replaying — strip stale RESUME writes so interrupts re-fire |
| 4 | | S | _(Send)_ | replay_state² | True¹ | _(popped)_³ | False | False | **Yes** | `is_replaying=T`, `RESUMING` absent → strip. Subgraph replays cleanly |
| 5 | **Time-travel to subgraph ckpt** (THE BUG) | P | `invoke(None, sub_cfg)` | ckpt_id⁴ | True | _(absent)_ | True | — | **Yes** | Parent replays from historical checkpoint |
| 5 | | S | _(Send)_ | **ckpt_id**⁵ | True¹ | **True** | **True** | **True** | **Yes** ✨ | `in_map=True` overrides `RESUMING=True` → force strip. THE FIX. |
| 5 | | S _(without fix)_ | _(Send)_ | ckpt_id⁵ | True¹ | **True** | **True** | _(no check)_ | **No** ❌ | BUG: `RESUMING=True` prevents strip → stale RESUME values → interrupt() doesn't re-fire |
| 6 | **Fork from subgraph ckpt** | P | `invoke(None, update_state(sub_cfg,...))` | ckpt_id | True | _(absent)_ | True | — | **Yes** | Same as case 5 — fork creates new ckpt, but checkpoint_map still resolves |
| 6 | | S | _(Send)_ | ckpt_id⁵ | True¹ | True | True | **True** | **Yes** ✨ | Same fix applies |
| 7 | **Resume after case 5 re-interrupts** | P | `invoke(Cmd(resume="a2"), cfg)` | latest | False | _(absent)_ | True | — | No | Normal resume — keep RESUME writes |
| 7 | | S | _(Send)_ | latest | True¹ | True | True | False⁶ | **No** | ask_2 gets "a2" from fresh RESUME write. Correct. |

## Footnotes

**¹** `is_replaying` is always `True` for subgraphs on tick 1 because `_algo.py` sets `CONFIG_KEY_CHECKPOINT_ID: None` — the key exists (even with `None` value), so `key in dict` is `True`. After tick 1, line 563 sets `is_replaying = False`.

**²** `replay_state` branch: parent passed `CONFIG_KEY_REPLAY_STATE = ReplayState(parent_ckpt_id)`. The subgraph uses `replay_state.get_checkpoint()` which does `checkpointer.list(before=parent_ckpt_id, limit=1)` to find the subgraph's checkpoint from before the replay point.

**³** The `replay_state` branch in `__enter__` (line 1158) explicitly pops `CONFIG_KEY_RESUMING` from config. This makes `is_resuming = False` in `_first()` because for nested graphs the fallback (`self.input is None or input_is_command`) is False (input is a Send arg).

**⁴** Parent `__init__` clears `checkpoint_ns → ""` and `checkpoint_id → None` (line 273-277), then resolves `""` from checkpoint_map → gets `parent_checkpoint_id` onto `checkpoint_config` (line 278-290).

**⁵** Subgraph `__init__` resolves its namespace (e.g. `"executor:task_id"`) from checkpoint_map → gets `subgraph_checkpoint_id` onto `checkpoint_config`. This is why the new first branch in `__enter__` (line 1141) fires — `checkpoint_config` has a truthy `checkpoint_id`.

**⁶** After case 5 completes/re-interrupts and user resumes, the config is a normal thread config with no checkpoint_map entry for the subgraph. `in_map` is False, so normal resume logic applies.

## The core tension (case 5)

The parent **can't distinguish** these cases when propagating flags to subgraphs:

| Parent sees | What's actually happening | Subgraph should strip RESUME? |
|---|---|---|
| `input=None`, has checkpoint | Resume after interrupt | Yes (replaying) |
| `input=None`, has checkpoint | Resume after interrupt | No (resuming) |
| `input=Command(resume=...)` | Active resume | No (resuming) |
| `input=None`, has checkpoint | Time-travel to subgraph | Yes (replaying) |

The **only** distinguishing signal at the subgraph level is whether its namespace appears in `checkpoint_map`.
