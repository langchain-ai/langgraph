"""Log channel storage for `InMemorySaver` (transcript log prototype).

The reference implementation of the log storage design: entries stored once,
keyed in path order; one step row per committing channel version; segments
for linear runs, with parent pointers between them. Reads are range reads
over a segment chain, and lifecycle is reachability from the checkpoints that
remain.

Rows, per `(thread_id, checkpoint_ns, channel)`:

- segment: `seg -> (parent_seg | None, fork_pos)`
- step: `version -> StepRow(seg, start, count, batch)`, unique on `(seg, start)`
- entry: `seg -> {(start, batch, task_path, task_id, idx, j): json_text}`

`batch` names the execution that wrote a step's entries: the checkpoint the
step started from for entries that came through `put_writes`, the committing
checkpoint for entries a `LogStep` carried inline.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import LogPage, log_range, writes_sort_key
from langgraph.checkpoint.serde.types import LogAppend, LogStep

_SEGMENT_NS = uuid.UUID("6f6c6f67-0000-4000-8000-6c6f67736567")

EntryKey = tuple[int, str, str, str, int, int]


@dataclass
class StepRow:
    seg: str
    start: int
    count: int
    batch: str
    exec: str

    @property
    def end(self) -> int:
        return self.start + self.count


@dataclass
class ChannelLog:
    segments: dict[str, tuple[str | None, int]] = field(default_factory=dict)
    steps: dict[str, StepRow] = field(default_factory=dict)
    step_at: dict[tuple[str, int], str] = field(default_factory=dict)
    entries: defaultdict[str, dict[EntryKey, str]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    batch_segments: defaultdict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def copy(self) -> ChannelLog:
        new = ChannelLog()
        new.segments = dict(self.segments)
        new.steps = {
            v: StepRow(r.seg, r.start, r.count, r.batch, r.exec)
            for v, r in self.steps.items()
        }
        new.step_at = dict(self.step_at)
        for seg, rows in self.entries.items():
            new.entries[seg] = dict(rows)
        for batch, segs in self.batch_segments.items():
            new.batch_segments[batch] = list(segs)
        return new

    # -- structure ---------------------------------------------------------

    def committed(self, seg: str) -> bool:
        return any(s == seg for s, _ in self.step_at)

    def tip(self, seg: str) -> int:
        ends = [self.steps[v].end for (s, _), v in self.step_at.items() if s == seg]
        return max(ends) if ends else self.segments[seg][1]

    def chain(self, seg: str, upto: int) -> list[tuple[str, int]]:
        """`(segment, upto)` pairs from the leaf's segment to the root."""
        out: list[tuple[str, int]] = []
        cur: str | None = seg
        while cur is not None:
            out.append((cur, upto))
            cur, upto = self.segments[cur]
        return out

    def segment_for(self, leaf: tuple[str | None, int], batch: str, exec_: str) -> str:
        """The fork rule: continue the leaf's segment if the leaf is its tip;
        otherwise use a new segment forking at the leaf, reusing one this
        batch already started and nothing committed to."""
        seg, end = leaf
        if seg is not None and self.tip(seg) == end:
            return seg
        for cand in self.batch_segments.get(batch, ()):
            if self.segments[cand] == leaf and not self.committed(cand):
                return cand
        new = str(
            uuid.uuid5(
                _SEGMENT_NS, f"{batch}:{exec_}:{len(self.batch_segments[batch])}"
            )
        )
        self.segments[new] = leaf
        self.batch_segments[batch].append(new)
        return new

    def step_texts(self, seg: str, start: int, batch: str) -> list[str]:
        rows = self.entries.get(seg, {})
        keys = sorted(
            (k for k in rows if k[0] == start and k[1] == batch),
            key=lambda k: (writes_sort_key(k[2], k[3], k[4]), k[5]),
        )
        return [rows[k] for k in keys]

    def replace_write(
        self,
        seg: str,
        start: int,
        batch: str,
        task_path: str,
        task_id: str,
        idx: int,
        texts: Iterable[str],
    ) -> None:
        rows = self.entries[seg]
        for k in [k for k in rows if k[:5] == (start, batch, task_path, task_id, idx)]:
            del rows[k]
        for j, text in enumerate(texts):
            rows[(start, batch, task_path, task_id, idx, j)] = text

    def is_committed_step(self, seg: str, start: int, batch: str) -> bool:
        v = self.step_at.get((seg, start))
        return v is not None and self.steps[v].batch == batch

    def pending(self, batch: str) -> list[tuple[str, str, int, LogAppend, str]]:
        """Uncommitted writes stored under `batch`, as
        `(task_path, task_id, idx, LogAppend, exec)`, in write order."""
        groups: dict[tuple[str, int, str, str, int], list[tuple[int, str]]] = (
            defaultdict(list)
        )
        for seg, rows in self.entries.items():
            for (start, b, task_path, task_id, idx, j), text in rows.items():
                if b != batch or self.is_committed_step(seg, start, b):
                    continue
                groups[(seg, start, task_path, task_id, idx)].append((j, text))
        out = []
        for (_, start, task_path, task_id, idx), items in groups.items():
            texts = tuple(t for _, t in sorted(items))
            out.append((task_path, task_id, idx, LogAppend(start, texts), ""))
        out.sort(key=lambda w: writes_sort_key(w[0], w[1], w[2]))
        return out


class InMemoryLog:
    """Log storage for one `InMemorySaver`."""

    def __init__(self) -> None:
        self.logs: dict[tuple[str, str, str], ChannelLog] = {}
        self.lock = threading.RLock()

    def _log(self, thread_id: str, ns: str, channel: str) -> ChannelLog:
        key = (thread_id, ns, channel)
        if key not in self.logs:
            self.logs[key] = ChannelLog()
        return self.logs[key]

    def is_log_version(
        self, thread_id: str, ns: str, channel: str, version: Any
    ) -> bool:
        log = self.logs.get((thread_id, ns, channel))
        return log is not None and version in log.steps

    def marker(
        self, thread_id: str, ns: str, channel: str, version: Any
    ) -> LogStep | None:
        log = self.logs.get((thread_id, ns, channel))
        if log is None or version not in log.steps:
            return None
        row = log.steps[version]
        return LogStep(row.start, row.count, row.exec)

    def leaf(
        self, thread_id: str, ns: str, channel: str, version: Any
    ) -> tuple[str | None, int]:
        if version is None:
            return (None, 0)
        log = self.logs.get((thread_id, ns, channel))
        if log is None or version not in log.steps:
            raise ValueError(
                f"log channel {channel!r}: version {version} has no step row"
            )
        row = log.steps[version]
        return (row.seg, row.end)

    # -- writes ------------------------------------------------------------

    def put_append(
        self,
        thread_id: str,
        ns: str,
        channel: str,
        parent_version: Any,
        batch: str,
        task_id: str,
        task_path: str,
        idx: int,
        value: LogAppend,
    ) -> None:
        with self.lock:
            log = self._log(thread_id, ns, channel)
            leaf = self.leaf(thread_id, ns, channel, parent_version)
            if value.start != leaf[1]:
                raise ValueError(
                    f"log channel {channel!r}: append at {value.start}, but the "
                    f"checkpoint's log has {leaf[1]} entries"
                )
            seg = log.segment_for(leaf, batch, value.exec)
            if log.is_committed_step(seg, value.start, batch):
                return  # already committed: a reused write, nothing to do
            log.replace_write(
                seg, value.start, batch, task_path, task_id, idx, value.entries
            )

    def put_step(
        self,
        thread_id: str,
        ns: str,
        channel: str,
        parent_version: Any,
        parent_id: str | None,
        checkpoint_id: str,
        version: Any,
        step: LogStep,
    ) -> None:
        with self.lock:
            log = self._log(thread_id, ns, channel)
            leaf = self.leaf(thread_id, ns, channel, parent_version)
            if step.start != leaf[1]:
                raise ValueError(
                    f"log channel {channel!r}: step starts at {step.start}, but "
                    f"the parent checkpoint's log has {leaf[1]} entries"
                )
            inline = step.entries is not None
            batch = checkpoint_id if inline or parent_id is None else parent_id
            seg = log.segment_for(leaf, batch, step.exec)
            if inline:
                log.replace_write(seg, step.start, batch, "", "", 0, step.entries)
            texts = log.step_texts(seg, step.start, batch)
            if len(texts) != step.count:
                raise ValueError(
                    f"log channel {channel!r}: the step at {step.start} commits "
                    f"{step.count} entries, but {len(texts)} were written"
                )
            existing = log.step_at.get((seg, step.start))
            if existing is not None and existing != version:
                raise ValueError(
                    f"log channel {channel!r}: position {step.start} of segment "
                    f"{seg} was already committed by another step (concurrent "
                    "writers on one checkpoint)"
                )
            log.steps[version] = StepRow(seg, step.start, step.count, batch, step.exec)
            log.step_at[(seg, step.start)] = version

    # -- reads -------------------------------------------------------------

    def read(
        self,
        thread_id: str,
        ns: str,
        channel: str,
        version: Any,
        batch: str | None,
        *,
        after: int | None,
        before: int | None,
        limit: int | None,
    ) -> LogPage:
        with self.lock:
            log = self.logs.get((thread_id, ns, channel))
            pending: list[Any] = []
            if log is not None and batch is not None:
                pending = [
                    json.loads(text)
                    for *_, value, _ in log.pending(batch)
                    for text in value.entries
                ]
            if log is None or version is None or version not in log.steps:
                return LogPage([], 0, pending)
            row = log.steps[version]
            lo, hi = log_range(row.end, after, before, limit)
            out: list[tuple[int, Any]] = []
            for seg, upto in log.chain(row.seg, row.end):
                if upto <= lo:
                    break
                starts = sorted(s for (sg, s) in log.step_at if sg == seg and s < upto)
                for start in starts:
                    srow = log.steps[log.step_at[(seg, start)]]
                    if srow.end <= lo or srow.start >= hi:
                        continue
                    texts = log.step_texts(seg, start, srow.batch)
                    if len(texts) != srow.count:
                        raise ValueError(
                            f"log channel {channel!r}: step at {start} has "
                            f"{len(texts)} entries, expected {srow.count}"
                        )
                    for i, text in enumerate(texts):
                        if lo <= start + i < hi:
                            out.append((start + i, json.loads(text)))
            out.sort(key=lambda e: e[0])
            return LogPage(out, row.end, pending)

    def pending_writes(
        self, thread_id: str, ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, str, int, LogAppend]]:
        """Uncommitted log writes stored on a checkpoint, as
        `(task_path, task_id, channel, idx, LogAppend)`."""
        with self.lock:
            out = []
            for (t, n, channel), log in self.logs.items():
                if t != thread_id or n != ns:
                    continue
                for task_path, task_id, idx, value, _ in log.pending(checkpoint_id):
                    out.append((task_path, task_id, channel, idx, value))
            return out

    # -- lifecycle ---------------------------------------------------------

    def delete_thread(self, thread_id: str) -> None:
        with self.lock:
            for key in [k for k in self.logs if k[0] == thread_id]:
                del self.logs[key]

    def copy_thread(self, source: str, target: str) -> None:
        with self.lock:
            for (t, ns, channel), log in list(self.logs.items()):
                if t == source:
                    self.logs[(target, ns, channel)] = log.copy()

    def collect(
        self,
        thread_id: str,
        remaining: dict[str, dict[str, Any]],
        version_of: Callable[[str, str, str], Any],
    ) -> None:
        """Keep exactly what remaining checkpoints reach.

        `remaining` maps each namespace to its remaining checkpoint ids;
        `version_of(ns, checkpoint_id, channel)` returns a checkpoint's
        version of a channel.
        """
        with self.lock:
            for (t, ns, channel), log in list(self.logs.items()):
                if t != thread_id:
                    continue
                cids = remaining.get(ns, {})
                reach: dict[str, int] = {}
                for cid in cids:
                    version = version_of(ns, cid, channel)
                    if version is None or version not in log.steps:
                        continue
                    row = log.steps[version]
                    for seg, upto in log.chain(row.seg, row.end):
                        reach[seg] = max(reach.get(seg, -1), upto)
                kept_steps = {
                    v: r
                    for v, r in log.steps.items()
                    if r.seg in reach and r.start < reach[r.seg]
                }
                kept_keys = {(r.seg, r.start, r.batch) for r in kept_steps.values()}
                for seg in list(log.entries):
                    rows = log.entries[seg]
                    for k in list(rows):
                        start, batch = k[0], k[1]
                        if (seg, start, batch) in kept_keys:
                            continue
                        if batch in cids and not log.is_committed_step(
                            seg, start, batch
                        ):
                            continue  # pending at a remaining checkpoint
                        del rows[k]
                    if not rows:
                        del log.entries[seg]
                log.steps = kept_steps
                log.step_at = {(r.seg, r.start): v for v, r in kept_steps.items()}
                live = set(reach) | set(log.entries)
                log.segments = {s: p for s, p in log.segments.items() if s in live}
                for batch in list(log.batch_segments):
                    segs = [s for s in log.batch_segments[batch] if s in live]
                    if segs:
                        log.batch_segments[batch] = segs
                    else:
                        del log.batch_segments[batch]
                if not log.steps and not log.entries:
                    del self.logs[(t, ns, channel)]
