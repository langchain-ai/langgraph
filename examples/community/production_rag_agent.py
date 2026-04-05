"""
Production RAG Agent Example
Author: Rehan Malik

Demonstrates a production-ready RAG agent with:
- Hybrid retrieval (dense + sparse)
- Reranking
- Confidence-based retry logic
- Structured output validation
"""

from typing import TypedDict
from dataclasses import dataclass


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    rerank_top_k: int = 3
    similarity_threshold: float = 0.7
    max_retries: int = 2


class AgentState(TypedDict):
    query: str
    retrieved_docs: list
    reranked_docs: list
    answer: str
    sources: list
    confidence: float
    retry_count: int


def retrieve(state: AgentState) -> AgentState:
    """Retrieve documents using hybrid search (dense + BM25)."""
    state["retrieved_docs"] = [
        {"text": f"Doc for: {state['query']}", "score": 0.85, "source": "kb_1"}
    ]
    return state


def rerank(state: AgentState) -> AgentState:
    """Cross-encoder reranking for precision."""
    state["reranked_docs"] = sorted(
        state["retrieved_docs"], key=lambda d: d["score"], reverse=True
    )
    return state


def generate(state: AgentState) -> AgentState:
    """Generate answer with source attribution."""
    context = "\n".join(d["text"] for d in state["reranked_docs"])
    state["answer"] = f"Based on {len(state['reranked_docs'])} sources: [generated answer]"
    state["sources"] = [d["source"] for d in state["reranked_docs"]]
    state["confidence"] = 0.92
    return state


def should_retry(state: AgentState) -> str:
    if state.get("confidence", 0) < 0.7 and state.get("retry_count", 0) < 2:
        return "retrieve"
    return "end"


if __name__ == "__main__":
    state: AgentState = {
        "query": "How does RLHF work?",
        "retrieved_docs": [], "reranked_docs": [],
        "answer": "", "sources": [], "confidence": 0.0, "retry_count": 0
    }
    for step in [retrieve, rerank, generate]:
        state = step(state)
    print(f"Answer: {state['answer']}")
    print(f"Confidence: {state['confidence']}")
