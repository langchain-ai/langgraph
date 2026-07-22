"""
DPX × LangGraph — Autonomous Invoice Settlement Agent
=====================================================
A typed state-graph that handles the full settlement lifecycle.
A single /flow-check node replaces the prior screen → esg_score → route
sequence, running all three in parallel on the server and returning a
ready-to-use settle body on PROCEED.

    flow_check → [PROCEED → settle → done]
               → [HOLD / BLOCKED → blocked → done]

Usage:
    pip install langgraph langchain-anthropic dpx-sdk
    export ANTHROPIC_API_KEY=...
    python langgraph_settlement_agent.py

Set SANDBOX=false in your env to execute on Base mainnet.
"""

from __future__ import annotations

import os
import json
from typing import TypedDict, Literal
import httpx
from langgraph.graph import StateGraph, END

# ── DPX endpoints ──────────────────────────────────────────────────────────────
AGENT_URL = "https://agent.untitledfinancial.com"

SANDBOX = os.environ.get("SANDBOX", "true").lower() != "false"

# ── Sample invoice ─────────────────────────────────────────────────────────────
INVOICE = {
    "id":                   "INV-2026-07-001",
    "amount":               85_000,
    "source_currency":      "USD",
    "destination_currency": "EUR",
    "recipient_address":    "0x0000000000000000000000000000000000000001",
    "counterparty_name":    "Berlin Supplier GmbH",
    "counterparty_lei":     "EXAMPLE_LEI_00000000000",
}


# ── Graph state ────────────────────────────────────────────────────────────────

class SettlementState(TypedDict):
    invoice:           dict
    flow_check_result: dict | None
    flow_status:       Literal["PROCEED", "HOLD", "BLOCKED"] | None
    settle_body:       dict | None
    settlement_result: dict | None
    recommended_token: str | None
    error:             str | None
    log:               list[str]


# ── Node functions ─────────────────────────────────────────────────────────────

def node_flow_check(state: SettlementState) -> SettlementState:
    inv = state["invoice"]
    log = list(state.get("log", []))
    log.append(
        f"[flow_check] {inv['counterparty_name']} | "
        f"{inv['amount']:,} {inv['source_currency']}→{inv['destination_currency']}"
    )

    try:
        r = httpx.get(
            f"{AGENT_URL}/flow-check",
            params={
                "amount":           inv["amount"],
                "from":             inv["source_currency"],
                "to":               inv["destination_currency"],
                "recipientAddress": inv["recipient_address"],
                "lei":              inv.get("counterparty_lei"),
            },
            timeout=20,
        )
        result = r.json() if r.is_success else {"status": "BLOCKED", "error": r.text[:200]}
    except Exception as e:
        result = {"status": "BLOCKED", "error": str(e)}

    status = result.get("status", "BLOCKED")
    token  = result.get("settleBody", {}).get("token") or "USDC"
    log.append(f"[flow_check] Status: {status} | Token: {token}")

    return {
        **state,
        "flow_check_result": result,
        "flow_status":       status,
        "settle_body":       result.get("settleBody"),
        "recommended_token": token,
        "log":               log,
    }


def node_settle(state: SettlementState) -> SettlementState:
    inv        = state["invoice"]
    settle_body = state.get("settle_body") or {}
    token      = state.get("recommended_token") or "USDC"
    log        = list(state.get("log", []))
    log.append(f"[settle] Executing via {token}")

    body = {
        "amount":               inv["amount"],
        "sourceCurrency":       inv["source_currency"],
        "destinationCurrency":  inv["destination_currency"],
        "recipientAddress":     inv["recipient_address"],
        "token":                token,
        "sandbox":              SANDBOX,
        # pass quoteId from flow-check if available — avoids a second quote round-trip
        **({k: v for k, v in settle_body.items() if k in ("quoteId",)}),
    }

    try:
        r = httpx.post(f"{AGENT_URL}/settle", json=body, timeout=30)
        if r.status_code == 402:
            result = {"status": "x402_required", "payment_required": r.json()}
        elif r.is_success:
            result = r.json()
        else:
            result = {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        result = {"error": str(e)}

    sid = result.get("settlementId") or result.get("simulatedId") or "unknown"
    log.append(f"[settle] Complete — ID: {sid}")
    return {**state, "settlement_result": result, "log": log}


def node_blocked(state: SettlementState) -> SettlementState:
    inv    = state["invoice"]
    log    = list(state.get("log", []))
    result = state.get("flow_check_result") or {}
    reason = result.get("blockReason") or result.get("holdReason") or result.get("error") or "compliance block"
    status = state.get("flow_status", "BLOCKED")
    log.append(f"[{status.lower()}] Settlement halted — {reason}")
    return {
        **state,
        "error": f"Settlement {status}: {reason}. Counterparty: {inv['counterparty_name']}",
        "log":   log,
    }


# ── Routing logic ──────────────────────────────────────────────────────────────

def route_after_flow_check(state: SettlementState) -> str:
    status = state.get("flow_status", "BLOCKED")
    if status == "PROCEED":
        return "settle"
    return "blocked"


# ── Build graph ────────────────────────────────────────────────────────────────

builder: StateGraph = StateGraph(SettlementState)

builder.add_node("flow_check", node_flow_check)
builder.add_node("settle",     node_settle)
builder.add_node("blocked",    node_blocked)

builder.set_entry_point("flow_check")

builder.add_conditional_edges(
    "flow_check",
    route_after_flow_check,
    {"settle": "settle", "blocked": "blocked"},
)

builder.add_edge("settle",  END)
builder.add_edge("blocked", END)

graph = builder.compile()


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"DPX × LangGraph — Invoice Settlement Agent")
    print(f"Invoice: {INVOICE['id']}")
    print(f"Amount:  {INVOICE['amount']:,} {INVOICE['source_currency']} → {INVOICE['destination_currency']}")
    print(f"Mode:    {'SANDBOX' if SANDBOX else 'LIVE (Base mainnet)'}")
    print(f"{'='*60}\n")

    initial_state: SettlementState = {
        "invoice":           INVOICE,
        "flow_check_result": None,
        "flow_status":       None,
        "settle_body":       None,
        "settlement_result": None,
        "recommended_token": None,
        "error":             None,
        "log":               [],
    }

    final_state = graph.invoke(initial_state)

    print("\n── Execution log ──")
    for line in final_state.get("log", []):
        print(f"  {line}")

    if final_state.get("error"):
        print(f"\n✗  {final_state['error']}")
    else:
        result = final_state.get("settlement_result", {})
        sid    = result.get("settlementId") or result.get("simulatedId") or "—"
        token  = final_state.get("recommended_token") or "USDC"
        print(f"\n── Settlement complete ──")
        print(f"  ID:    {sid}")
        print(f"  Token: {token}")
        print(f"\n  Full result:")
        print(f"  {json.dumps(result, indent=2)}")
