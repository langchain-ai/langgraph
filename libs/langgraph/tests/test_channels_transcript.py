from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from langgraph._internal._typing import MISSING
from langgraph.channels.transcript import TranscriptChannel, TranscriptRef
from langgraph.graph.message import REMOVE_ALL_MESSAGES


class FakeStore:
    """Commits refs like a transcript-aware checkpointer and reads them back."""

    def __init__(self) -> None:
        self.transcripts: dict[str, dict] = {}

    def commit(self, ch: TranscriptChannel) -> TranscriptRef:
        ref = ch.checkpoint()
        t = self.transcripts.setdefault(
            ref.transcript_id,
            {"parent": ref.parent_transcript_id, "fork": ref.fork_seq, "entries": {}},
        )
        for e in ref.entries:
            t["entries"][e.seq] = e
        return ref

    def _read(self, tid: str, head: int) -> list:
        t = self.transcripts[tid]
        out = self._read(t["parent"], min(head, t["fork"])) if t["parent"] else []
        return out + [t["entries"][s] for s in sorted(t["entries"]) if s <= head]

    def load(self, ref: TranscriptRef, *, tip: bool = True) -> TranscriptChannel:
        return TranscriptChannel().from_checkpoint(
            TranscriptRef(
                ref.transcript_id,
                ref.head,
                entries=self._read(ref.transcript_id, ref.head),
                is_tip=tip,
            )
        )


def contents(ch: TranscriptChannel) -> list[str]:
    return [m.content for m in ch.get()]


def test_appends_and_resends() -> None:
    store = FakeStore()
    ch = TranscriptChannel().from_checkpoint(MISSING)
    ch.update([[HumanMessage("h0", id="h0"), AIMessage("a0", id="a0")]])
    ch = store.load(store.commit(ch))
    ch.update([[HumanMessage("h1", id="h1")]])
    ref = store.commit(ch)
    assert [e.seq for e in ref.entries] == [3]
    ch = store.load(ref)
    assert contents(ch) == ["h0", "a0", "h1"]

    ch.update([[AIMessage("a0", id="a0"), HumanMessage("h1", id="h1")]])
    assert ch.checkpoint().entries == []


def test_edit_forks_and_keeps_original() -> None:
    store = FakeStore()
    ch = TranscriptChannel().from_checkpoint(MISSING)
    ch.update([[HumanMessage("h0", id="h0"), AIMessage("a0", id="a0")]])
    ch = store.load(store.commit(ch))
    ch.update([[HumanMessage("h1", id="h1"), AIMessage("a1", id="a1")]])
    original = store.commit(ch)
    ch = store.load(original)

    ch.update([[AIMessage("a0-edited", id="a0")]])
    edited = store.commit(ch)
    assert edited.parent_transcript_id == original.transcript_id
    assert edited.fork_seq == 1
    assert contents(store.load(edited)) == ["h0", "a0-edited", "h1", "a1"]
    assert contents(store.load(original, tip=False)) == ["h0", "a0", "h1", "a1"]

    # A second fork before the first is committed branches off the same parent.
    ch.update([[RemoveMessage(id="h1")]])
    removed = store.commit(ch)
    assert removed.parent_transcript_id == original.transcript_id
    assert contents(store.load(removed)) == ["h0", "a0-edited", "a1"]


def test_edit_before_any_commit_has_no_parent() -> None:
    ch = TranscriptChannel().from_checkpoint(MISSING)
    ch.update([[HumanMessage("x", id="x"), AIMessage("y", id="y")]])
    ch.update([[AIMessage("y2", id="y")]])
    ref = ch.checkpoint()
    assert (ref.parent_transcript_id, ref.fork_seq) == ("", 0)
    assert contents(ch) == ["x", "y2"]


def test_edit_inside_reset_replays_reset() -> None:
    store = FakeStore()
    ch = TranscriptChannel().from_checkpoint(MISSING)
    ch.update([[HumanMessage("h0", id="h0")]])
    ch = store.load(store.commit(ch))
    ch.update(
        [[RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage("p", id="p"), AIMessage("q", id="q")]]
    )
    ch = store.load(store.commit(ch))
    ch.update([[AIMessage("q2", id="q")]])
    ref = store.commit(ch)
    assert [(e.seq, e.kind) for e in ref.entries] == [(2, "reset")]
    assert contents(store.load(ref)) == ["p", "q2"]


def test_load_non_tip_forks() -> None:
    store = FakeStore()
    ch = TranscriptChannel().from_checkpoint(MISSING)
    ch.update([[HumanMessage("h0", id="h0")]])
    first = store.commit(ch)
    ch = store.load(first)
    ch.update([[AIMessage("a0", id="a0")]])
    store.commit(ch)

    branch = store.load(first, tip=False)
    branch.update([[AIMessage("other", id="o")]])
    ref = store.commit(branch)
    assert ref.parent_transcript_id == first.transcript_id
    assert contents(store.load(ref)) == ["h0", "other"]
