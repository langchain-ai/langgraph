"""Compare token-level LLM streaming: v1 graph.stream() vs v2 StreamingHandler.

Runs the same single-node chatbot graph six ways:
  1. v1 sync   — graph.stream(stream_mode="messages")
  2. v2 sync   — StreamingHandler.stream() -> run.messages
  3. v2 sync   — same, with a custom TokenMetrics transformer
  4. v1 async  — graph.astream(stream_mode="messages")
  5. v2 async  — StreamingHandler.astream() -> run.messages
  6. v2 async  — same, with a custom TokenMetrics transformer

The TokenMetrics transformer demonstrates v2's extensibility: it tracks
first-token latency, token count, and throughput as a reusable component
that runs alongside the built-in projections without any changes to the
consumption loop. The v1 equivalent requires weaving all that bookkeeping
into your stream loop inline.

Requirements:
    pip install langchain-openai
    export OPENAI_API_KEY=...

Usage:
    python streaming_comparison.py
    python streaming_comparison.py --sync-only
    python streaming_comparison.py --async-only
    python streaming_comparison.py --question "your question here"
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, add_messages
from langgraph.stream import StreamChannel, StreamingHandler, StreamTransformer
from langgraph.stream._types import ProtocolEvent


# ---------------------------------------------------------------------------
# Graph setup
# ---------------------------------------------------------------------------


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


_llm = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    return _llm


def chatbot(state: State) -> dict[str, Any]:
    return {"messages": [_get_llm().invoke(state["messages"])]}


async def achatbot(state: State) -> dict[str, Any]:
    return {"messages": [await _get_llm().ainvoke(state["messages"])]}


def build_graph(*, use_async_node: bool = False):
    builder = StateGraph(State)
    builder.add_node("chatbot", achatbot if use_async_node else chatbot)
    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")
    return builder.compile()


# ---------------------------------------------------------------------------
# Custom transformer: TokenMetrics
# ---------------------------------------------------------------------------


@dataclass
class TokenSnapshot:
    """A throughput measurement pushed to the channel.

    Periodic snapshots have `is_final=False`. The last item pushed in
    `finalize()` has `is_final=True` and includes `first_token_latency`.
    """

    token_index: int
    elapsed: float
    tokens_per_sec: float
    is_final: bool = False
    first_token_latency: float | None = None


class TokenMetricsTransformer(StreamTransformer):
    """Tracks token throughput and first-token latency.

    Observes "messages" events and counts chunks with non-empty content.
    Pushes a TokenSnapshot to its StreamChannel every N tokens so
    consumers can monitor throughput in real time. A final snapshot with
    `is_final=True` is pushed when the run completes.

    All data is consumed through the stream via
    `run.extensions["token_metrics"]` — no need to read transformer
    internals.

    This is the kind of cross-cutting concern that v2 transformers handle
    cleanly: write it once, attach it to any graph, and it works alongside
    all other projections without touching the consumption loop.
    """

    def __init__(self, *, snapshot_every: int = 10) -> None:
        self._channel: StreamChannel[TokenSnapshot] = StreamChannel("token_metrics")
        self._snapshot_every = snapshot_every
        self._t0: float | None = None
        self._first_token_time: float | None = None
        self._token_count = 0

    def init(self) -> dict[str, Any]:
        return {"token_metrics": self._channel}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True

        # Only count root-namespace events
        if event["params"]["namespace"]:
            return True

        chunk = event["params"]["data"]
        # messages data is (chunk, metadata) tuple
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            return True
        message, _metadata = chunk
        if not hasattr(message, "content") or not message.content:
            return True

        now = time.time()
        if self._t0 is None:
            self._t0 = now
        if self._first_token_time is None:
            self._first_token_time = now

        self._token_count += 1

        # Push periodic snapshots
        if self._token_count % self._snapshot_every == 0:
            elapsed = now - self._t0
            self._channel.push(
                TokenSnapshot(
                    token_index=self._token_count,
                    elapsed=elapsed,
                    tokens_per_sec=self._token_count / elapsed if elapsed > 0 else 0,
                )
            )

        return True

    def finalize(self) -> None:
        now = time.time()
        elapsed = now - self._t0 if self._t0 else 0
        self._channel.push(
            TokenSnapshot(
                token_index=self._token_count,
                elapsed=elapsed,
                tokens_per_sec=self._token_count / elapsed if elapsed > 0 else 0,
                is_final=True,
                first_token_latency=(
                    self._first_token_time - self._t0
                    if self._first_token_time and self._t0
                    else None
                ),
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def separator(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# V1 sync: graph.stream(stream_mode="messages")
# ---------------------------------------------------------------------------


def run_v1_sync(question: str) -> None:
    separator("V1 SYNC: graph.stream(stream_mode='messages')")

    graph = build_graph()
    input_data = {"messages": [HumanMessage(content=question)]}

    t0 = time.time()
    first_token_time = None
    token_count = 0

    for chunk, metadata in graph.stream(input_data, stream_mode="messages"):
        if hasattr(chunk, "content") and chunk.content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1
            print(chunk.content, end="", flush=True)

    elapsed = time.time() - t0
    ftl = f"{first_token_time - t0:.3f}s" if first_token_time else "n/a"
    tps = token_count / elapsed if elapsed > 0 else 0
    print(f"\n  {token_count} tokens | first: {ftl} | {tps:.1f} tok/s | {elapsed:.2f}s total")


# ---------------------------------------------------------------------------
# V1 sync with inline metrics (the v1 equivalent of the transformer)
# ---------------------------------------------------------------------------


def run_v1_sync_with_metrics(question: str) -> None:
    separator("V1 SYNC + INLINE METRICS")
    print("  (same bookkeeping the transformer does, but woven into the loop)")

    graph = build_graph()
    input_data = {"messages": [HumanMessage(content=question)]}

    # All of this state lives in your consumption loop.
    # Duplicate it everywhere you stream this graph.
    t0 = time.time()
    first_token_time = None
    token_count = 0
    snapshot_every = 10
    snapshots: list[dict] = []

    for chunk, metadata in graph.stream(input_data, stream_mode="messages"):
        if hasattr(chunk, "content") and chunk.content:
            now = time.time()
            if first_token_time is None:
                first_token_time = now
            token_count += 1
            print(chunk.content, end="", flush=True)

            # Inline throughput tracking — mixed into your display logic
            if token_count % snapshot_every == 0:
                elapsed = now - t0
                snapshots.append({
                    "token_index": token_count,
                    "elapsed": elapsed,
                    "tokens_per_sec": token_count / elapsed if elapsed > 0 else 0,
                })

    # Compute summary inline
    elapsed = time.time() - t0
    ftl = first_token_time - t0 if first_token_time else None
    tps = token_count / elapsed if elapsed > 0 else 0

    print(f"\n  {token_count} tokens | first: {ftl:.3f}s | {tps:.1f} tok/s | {elapsed:.2f}s total")
    if snapshots:
        print(f"  Throughput snapshots: {len(snapshots)}")
        for s in snapshots:
            print(f"    token {s['token_index']:>4d}: {s['tokens_per_sec']:.1f} tok/s")


# ---------------------------------------------------------------------------
# V2 sync: StreamingHandler → run.messages
# ---------------------------------------------------------------------------


def run_v2_sync(question: str) -> None:
    separator("V2 SYNC: StreamingHandler.stream() -> run.messages")

    graph = build_graph()
    handler = StreamingHandler(graph)
    input_data = {"messages": [HumanMessage(content=question)]}

    run = handler.stream(input_data)

    # run.messages yields (chunk, metadata) tuples as the LLM produces them
    for chunk, metadata in run.messages:
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)

    output = run.output
    assert output is not None
    print(f"\n  Final: {output['messages'][-1].content[:80]}...")


# ---------------------------------------------------------------------------
# V2 sync with custom transformer
# ---------------------------------------------------------------------------


def run_v2_sync_with_transformer(question: str) -> None:
    separator("V2 SYNC + TokenMetrics TRANSFORMER")
    print("  (metrics computed by a reusable transformer — loop stays clean)")

    graph = build_graph()
    handler = StreamingHandler(graph)
    input_data = {"messages": [HumanMessage(content=question)]}

    run = handler.stream(input_data, transformers=[TokenMetricsTransformer(snapshot_every=10)])

    # The consumption loop is unchanged — just print tokens.
    # The transformer silently tracks metrics in the background.
    for chunk, metadata in run.messages:
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)

    _ = run.output

    # All metrics are read from the stream, not the transformer instance.
    # The channel contains periodic snapshots + a final summary snapshot.
    snapshots = list(run.extensions["token_metrics"])
    final = next((s for s in snapshots if s.is_final), None)
    periodic = [s for s in snapshots if not s.is_final]

    if final:
        ftl = f"{final.first_token_latency:.3f}s" if final.first_token_latency is not None else "n/a"
        print(f"\n  {final.token_index} tokens | first: {ftl} | {final.tokens_per_sec:.1f} tok/s | {final.elapsed:.2f}s total")
    if periodic:
        print(f"  Throughput snapshots: {len(periodic)}")
        for snap in periodic:
            print(f"    token {snap.token_index:>4d}: {snap.tokens_per_sec:.1f} tok/s")


# ---------------------------------------------------------------------------
# V1 async: graph.astream(stream_mode="messages")
# ---------------------------------------------------------------------------


async def run_v1_async(question: str) -> None:
    separator("V1 ASYNC: graph.astream(stream_mode='messages')")

    graph = build_graph(use_async_node=True)
    input_data = {"messages": [HumanMessage(content=question)]}

    t0 = time.time()
    first_token_time = None
    token_count = 0

    async for chunk, metadata in graph.astream(input_data, stream_mode="messages"):
        if hasattr(chunk, "content") and chunk.content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1
            print(chunk.content, end="", flush=True)

    elapsed = time.time() - t0
    ftl = f"{first_token_time - t0:.3f}s" if first_token_time else "n/a"
    tps = token_count / elapsed if elapsed > 0 else 0
    print(f"\n  {token_count} tokens | first: {ftl} | {tps:.1f} tok/s | {elapsed:.2f}s total")


# ---------------------------------------------------------------------------
# V1 async with inline metrics
# ---------------------------------------------------------------------------


async def run_v1_async_with_metrics(question: str) -> None:
    separator("V1 ASYNC + INLINE METRICS")
    print("  (same bookkeeping the transformer does, but woven into the loop)")

    graph = build_graph(use_async_node=True)
    input_data = {"messages": [HumanMessage(content=question)]}

    t0 = time.time()
    first_token_time = None
    token_count = 0
    snapshot_every = 10
    snapshots: list[dict] = []

    async for chunk, metadata in graph.astream(input_data, stream_mode="messages"):
        if hasattr(chunk, "content") and chunk.content:
            now = time.time()
            if first_token_time is None:
                first_token_time = now
            token_count += 1
            print(chunk.content, end="", flush=True)

            if token_count % snapshot_every == 0:
                elapsed = now - t0
                snapshots.append({
                    "token_index": token_count,
                    "elapsed": elapsed,
                    "tokens_per_sec": token_count / elapsed if elapsed > 0 else 0,
                })

    elapsed = time.time() - t0
    ftl = first_token_time - t0 if first_token_time else None
    tps = token_count / elapsed if elapsed > 0 else 0

    print(f"\n  {token_count} tokens | first: {ftl:.3f}s | {tps:.1f} tok/s | {elapsed:.2f}s total")
    if snapshots:
        print(f"  Throughput snapshots: {len(snapshots)}")
        for s in snapshots:
            print(f"    token {s['token_index']:>4d}: {s['tokens_per_sec']:.1f} tok/s")


# ---------------------------------------------------------------------------
# V2 async: StreamingHandler → run.messages
# ---------------------------------------------------------------------------


async def run_v2_async(question: str) -> None:
    separator("V2 ASYNC: StreamingHandler.astream() -> run.messages")

    graph = build_graph(use_async_node=True)
    handler = StreamingHandler(graph)
    input_data = {"messages": [HumanMessage(content=question)]}

    run = await handler.astream(input_data)

    async for chunk, metadata in run.messages:
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)

    output = await run.output
    assert output is not None
    print(f"\n  Final: {output['messages'][-1].content[:80]}...")


# ---------------------------------------------------------------------------
# V2 async with custom transformer
# ---------------------------------------------------------------------------


async def run_v2_async_with_transformer(question: str) -> None:
    separator("V2 ASYNC + TokenMetrics TRANSFORMER")
    print("  (metrics computed by a reusable transformer — loop stays clean)")

    graph = build_graph(use_async_node=True)
    handler = StreamingHandler(graph)
    input_data = {"messages": [HumanMessage(content=question)]}

    run = await handler.astream(input_data, transformers=[TokenMetricsTransformer(snapshot_every=10)])

    async for chunk, metadata in run.messages:
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)

    _ = await run.output

    snapshots = [s async for s in run.extensions["token_metrics"]]
    final = next((s for s in snapshots if s.is_final), None)
    periodic = [s for s in snapshots if not s.is_final]

    if final:
        ftl = f"{final.first_token_latency:.3f}s" if final.first_token_latency is not None else "n/a"
        print(f"\n  {final.token_index} tokens | first: {ftl} | {final.tokens_per_sec:.1f} tok/s | {final.elapsed:.2f}s total")
    if periodic:
        print(f"  Throughput snapshots: {len(periodic)}")
        for snap in periodic:
            print(f"    token {snap.token_index:>4d}: {snap.tokens_per_sec:.1f} tok/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Token streaming comparison: v1 vs v2")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--async-only", action="store_true")
    parser.add_argument(
        "--question",
        default="Explain quantum entanglement in 2-3 sentences.",
    )
    args = parser.parse_args()

    if not args.async_only:
        # V1 baseline
        run_v1_sync(args.question)
        # V1 with inline metrics — the manual approach
        run_v1_sync_with_metrics(args.question)
        # V2 basic
        run_v2_sync(args.question)
        # V2 with transformer — same metrics, zero loop changes
        run_v2_sync_with_transformer(args.question)

    if not args.sync_only:
        # V1 baseline
        asyncio.run(run_v1_async(args.question))
        # V1 with inline metrics
        asyncio.run(run_v1_async_with_metrics(args.question))
        # V2 basic
        asyncio.run(run_v2_async(args.question))
        # V2 with transformer
        asyncio.run(run_v2_async_with_transformer(args.question))


if __name__ == "__main__":
    main()
