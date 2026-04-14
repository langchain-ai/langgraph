# -*- coding: utf-8 -*-
"""
LangGraph agent that runs Chart Library's edge-mining loop as an explicit
StateGraph. Each phase (cohort → explain → refine → report) is a node; the
graph can branch, fork, and persist state across invocations.

Compared to the "single LLM with tools" pattern, this gives you:
    * Deterministic ordering of tool calls (retrieval THEN analysis)
    * Easy fork-and-compare of multiple refined cohorts
    * Checkpointable state (memory saver / redis saver plug in directly)
    * No model cost for the retrieval/analysis steps — only the final write-up

Run:
    pip install langgraph langchain langchain-anthropic requests
    export ANTHROPIC_API_KEY=...
    export CHART_LIBRARY_KEY=cl_...   # chartlibrary.io/developers
    python cohort_edge_mining_graph.py NVDA 2024-06-18
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional, TypedDict

import requests

try:
    from langgraph.graph import StateGraph, END
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:  # pragma: no cover
    raise SystemExit("pip install langgraph langchain langchain-anthropic requests")

CHART_BASE = "https://chartlibrary.io"
CHART_KEY = os.environ["CHART_LIBRARY_KEY"]
H = {"Authorization": f"Bearer {CHART_KEY}", "Content-Type": "application/json"}


# ── Tool wrappers ──────────────────────────────────────────

def cohort(symbol: str, date: str, **kw) -> dict:
    body = {
        "anchor": {"symbol": symbol, "date": date},
        "filters": kw.get("filters", {}),
        "horizons": kw.get("horizons", [5, 10]),
        "top_k": kw.get("top_k", 500),
        "include_path_stats": True,
    }
    r = requests.post(f"{CHART_BASE}/api/v1/cohort", headers=H, json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def explain(cohort_id: str, horizon: int = 5) -> dict:
    r = requests.get(
        f"{CHART_BASE}/api/v1/cohort/{cohort_id}/explain",
        headers=H, params={"horizon": horizon}, timeout=30,
    )
    r.raise_for_status()
    return r.json()


def refine(cohort_id: str, filter_name: str) -> dict:
    extra = {"regime": {filter_name: True}} if filter_name.startswith("same_") else {}
    r = requests.post(
        f"{CHART_BASE}/api/v1/cohort/{cohort_id}/filter",
        headers=H,
        json={"extra_filters": extra, "include_path_stats": True},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── Graph state ────────────────────────────────────────────

class CohortState(TypedDict, total=False):
    symbol: str
    date: str
    initial: dict
    explain: dict
    winning_filter: Optional[str]
    refined: Optional[dict]
    report: str


# ── Nodes ──────────────────────────────────────────────────

def node_initial_cohort(state: CohortState) -> dict:
    res = cohort(state["symbol"], state["date"])
    print(f"[initial cohort] cohort_id={res['cohort_id']}  n={res['n_effective']}/{res['n_raw']}"
          f"  5d_p50={res['distributions']['5']['return_pct']['p50']:+.2f}%"
          f"  elapsed={res['elapsed_ms']}ms")
    return {"initial": res}


def node_explain(state: CohortState) -> dict:
    res = explain(state["initial"]["cohort_id"], horizon=5)
    # Pick the candidate filter with the largest |above_entry_shift|.
    winners = sorted(
        (r for r in res["rankings"] if not r.get("skipped") and r["n_after"] >= 50),
        key=lambda r: -abs(r.get("above_entry_shift_pp") or 0),
    )
    best = winners[0]["filter"] if winners else None
    for row in res["rankings"]:
        if row.get("skipped"):
            continue
        shift = row.get("above_entry_shift_pp", 0) or 0
        marker = " <-- picked" if row["filter"] == best else ""
        print(f"[explain] {row['filter']:<18s}  n_after={row['n_after']:4d}  shift={shift:+.2f}pp{marker}")
    return {"explain": res, "winning_filter": best}


def node_refine(state: CohortState) -> dict:
    if not state.get("winning_filter"):
        print("[refine] no winning filter, skipping")
        return {"refined": None}
    res = refine(state["initial"]["cohort_id"], state["winning_filter"])
    print(f"[refine] filter={state['winning_filter']}  n={res['n_effective']}/{res['n_raw']}"
          f"  5d_p50={res['distributions']['5']['return_pct']['p50']:+.2f}%"
          f"  elapsed={res['elapsed_ms']}ms")
    return {"refined": res}


def node_report(state: CohortState) -> dict:
    llm = ChatAnthropic(model="claude-sonnet-4-5", max_tokens=800)
    initial = state["initial"]
    refined = state.get("refined")

    context = {
        "symbol": state["symbol"],
        "date": state["date"],
        "baseline": {
            "n": initial["n_effective"],
            "survivorship": initial["survivorship"],
            "5d": initial["distributions"]["5"],
            "10d": initial["distributions"]["10"],
        },
        "winning_filter": state.get("winning_filter"),
        "refined": {
            "n": refined["n_effective"],
            "5d": refined["distributions"]["5"],
            "10d": refined["distributions"]["10"],
        } if refined else None,
        "explain_rankings": state["explain"]["rankings"],
    }

    messages = [
        SystemMessage(content=(
            "You are a stock-research analyst. Produce a 150-word briefing "
            "using ONLY the numbers in the provided context. Quote sample "
            "size, disclose survivorship, and note if the refined cohort "
            "materially changes the picture vs the baseline. Never invent "
            "numbers. End with a 1-sentence sizing/risk implication."
        )),
        HumanMessage(content=f"Data:\n{json.dumps(context, default=str, indent=2)}"),
    ]
    result = llm.invoke(messages)
    return {"report": result.content}


# ── Graph construction ────────────────────────────────────

def build_graph():
    g = StateGraph(CohortState)
    g.add_node("initial_cohort", node_initial_cohort)
    g.add_node("explain", node_explain)
    g.add_node("refine", node_refine)
    g.add_node("report", node_report)

    g.set_entry_point("initial_cohort")
    g.add_edge("initial_cohort", "explain")
    g.add_edge("explain", "refine")
    g.add_edge("refine", "report")
    g.add_edge("report", END)
    return g.compile()


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    date = sys.argv[2] if len(sys.argv) > 2 else "2024-06-18"
    graph = build_graph()
    final = graph.invoke({"symbol": symbol, "date": date})
    print("\n─── REPORT ───")
    print(final["report"])
