"""Verify that SEND values land in checkpoint_blobs (TASKS channel),
not just in checkpoint_writes.

Strategy:
  1. Build a tiny graph: node "1" emits a Send to node "2" via
     conditional edges. interrupt_before=["2"] freezes the loop after
     super-step 1 completes (cp_1 persisted) but before "2" runs.
  2. Run graph.invoke against PostgresSaver.
  3. Read checkpoint_blobs raw via psycopg.
  4. Find the row with channel = '__pregel_tasks' and decode the blob
     using the saver's serde. Assert it contains a non-empty list of
     Send objects.
  5. As a control, also dump checkpoint_writes to confirm the *same*
     value is independently present in the writes table.
"""

from __future__ import annotations

import operator
from typing import Annotated
from typing_extensions import TypedDict
from uuid import uuid4

from psycopg import Connection
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.types import TASKS
from langgraph.graph import START, StateGraph
from langgraph.types import Send

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5441/"


class State(TypedDict):
    history: Annotated[list[str], operator.add]


def make_graph(checkpointer: PostgresSaver, *, with_interrupt: bool = True):
    def node_one(state: State) -> State:
        return {"history": ["1"]}

    def node_two(state: dict) -> State:
        return {"history": [f"2:{state['payload']}"]}

    def fanout(state: State):
        return [Send("two", {"payload": x}) for x in ("a", "b", "c")]

    builder = StateGraph(State)
    builder.add_node("one", node_one)
    builder.add_node("two", node_two)
    builder.add_edge(START, "one")
    builder.add_conditional_edges("one", fanout, ["two"])
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["two"] if with_interrupt else [],
    )


def main() -> None:
    database = f"verify_{uuid4().hex[:12]}"
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        uri = DEFAULT_POSTGRES_URI + database
        with Connection.connect(
            uri, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            saver = PostgresSaver(conn)
            saver.setup()

            graph = make_graph(saver)
            thread_id = "t1"
            cfg = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke({"history": []}, cfg)
            print("=== invoke result (interrupted before 'two') ===")
            print(result)
            print()

            # Inspect raw checkpoint_blobs.
            rows = conn.execute(
                """
                SELECT thread_id, checkpoint_ns, channel, version, type,
                       octet_length(blob) AS nbytes, blob
                FROM checkpoint_blobs
                WHERE thread_id = %s
                ORDER BY channel, version
                """,
                (thread_id,),
            ).fetchall()

            print(f"=== checkpoint_blobs ({len(rows)} rows) ===")
            for r in rows:
                preview = (
                    "(empty)"
                    if r["blob"] is None
                    else f"{r['blob'][:60]!r}{'...' if len(r['blob']) > 60 else ''}"
                )
                print(
                    f"  channel={r['channel']!r:30s} v={r['version']!s:18s} "
                    f"type={r['type']:8s} nbytes={r['nbytes']!s:>5} blob={preview}"
                )
            print()

            tasks_rows = [r for r in rows if r["channel"] == TASKS]
            assert tasks_rows, (
                f"expected at least one row in checkpoint_blobs for {TASKS!r}, "
                f"got channels={[r['channel'] for r in rows]}"
            )

            # Decode the latest TASKS blob using the saver's serde.
            tasks_row = tasks_rows[-1]
            decoded = saver.serde.loads_typed((tasks_row["type"], tasks_row["blob"]))
            print(f"=== decoded {TASKS!r} blob ({tasks_row['type']}) ===")
            print(f"  type(decoded) = {type(decoded).__name__}")
            print(f"  len(decoded)  = {len(decoded)}")
            for i, v in enumerate(decoded):
                print(f"  [{i}] type={type(v).__name__} value={v!r}")
            print()

            assert isinstance(decoded, list), (
                f"expected list of Sends, got {type(decoded).__name__}"
            )
            assert len(decoded) == 3, f"expected 3 Sends (a, b, c), got {len(decoded)}"
            for v in decoded:
                assert isinstance(v, Send), (
                    f"expected Send instance in blob, got {type(v).__name__}"
                )
                assert v.node == "two"
            print("✅ ASSERTION PASSED: checkpoint_blobs contains the Sends.")
            print()

            # Control: confirm the same Sends are also in checkpoint_writes.
            wrows = conn.execute(
                """
                SELECT task_id, idx, channel, type, octet_length(blob) AS nbytes, blob
                FROM checkpoint_writes
                WHERE thread_id = %s AND channel = %s
                ORDER BY task_id, idx
                """,
                (thread_id, TASKS),
            ).fetchall()
            print(
                f"=== checkpoint_writes (channel={TASKS!r}, {len(wrows)} rows) ==="
            )
            for r in wrows:
                v = saver.serde.loads_typed((r["type"], r["blob"]))
                print(
                    f"  task={r['task_id']} idx={r['idx']} type={r['type']} "
                    f"nbytes={r['nbytes']} -> {v!r}"
                )

            assert len(wrows) == 3, (
                f"expected 3 writes-table rows for {TASKS!r}, got {len(wrows)}"
            )
            print(
                "✅ ASSERTION PASSED: checkpoint_writes also contains the Sends."
            )
            print()
            print(
                "Conclusion (cp_1): SEND is double-stored. blobs row carries "
                "the authoritative Send list; writes rows carry the same "
                "payload as a redundant per-task record."
            )

            # ----------------------------------------------------------------
            # Now resume: let 'two' actually run, so the Sends get consumed.
            # We expect the next checkpoint's TASKS blob to be cleared,
            # proving the accumulate=False lifecycle.
            # ----------------------------------------------------------------
            graph2 = make_graph(saver, with_interrupt=False)
            result2 = graph2.invoke(None, cfg)
            print()
            print("=== invoke result (resumed; 'two' fanout completed) ===")
            print(result2)
            print()

            rows2 = conn.execute(
                """
                SELECT channel, version, type, octet_length(blob) AS nbytes, blob
                FROM checkpoint_blobs
                WHERE thread_id = %s AND channel = %s
                ORDER BY version
                """,
                (thread_id, TASKS),
            ).fetchall()
            print(f"=== checkpoint_blobs rows for {TASKS!r} across all cps ===")
            for r in rows2:
                decoded_v = saver.serde.loads_typed((r["type"], r["blob"]))
                print(
                    f"  v={r['version']} type={r['type']} nbytes={r['nbytes']} "
                    f"-> {decoded_v!r}"
                )

            assert len(rows2) >= 2, (
                "expected at least 2 versions of TASKS blob across the run"
            )
            latest = saver.serde.loads_typed(
                (rows2[-1]["type"], rows2[-1]["blob"])
            )
            assert latest == [], (
                f"expected latest TASKS blob to be cleared (==[]) "
                f"after Sends were consumed, got {latest!r}"
            )
            print()
            print(
                "✅ ASSERTION PASSED: after Sends were consumed, the latest "
                "TASKS blob is [] (Topic with accumulate=False clears itself "
                "via update(EMPTY_SEQ) in apply_writes line 304-311)."
            )
    finally:
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


if __name__ == "__main__":
    main()
