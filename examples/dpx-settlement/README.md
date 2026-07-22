# DPX × LangGraph — Autonomous Invoice Settlement

A typed state-graph for autonomous invoice settlement with a built-in compliance gate.

```
flow_check → [PROCEED → settle → END]
           → [HOLD / BLOCKED → blocked → END]
```

## What DPX is

DPX is a settlement rail for AI agents. The `/flow-check` endpoint runs AML/sanctions screening, FATF R16 checks, ESG scoring, oracle stability, and stablecoin routing in a single call — returning a ready-to-use settle body on PROCEED.

No API key. No onboarding. No custody.

## Setup

```bash
pip install langgraph httpx
python settlement_agent.py
```

Set `SANDBOX=false` to execute on Base mainnet. Sandbox mode runs oracle and compliance against live endpoints — nothing moves on-chain.

## Graph structure

| Node | What it does |
|------|--------------|
| `flow_check` | Calls `/flow-check` — AML, ESG, oracle, stablecoin routing in parallel |
| `settle` | Calls `/settle` with token and quoteId from flow-check |
| `blocked` | Logs HOLD/BLOCKED reason, halts the graph |

## State

```python
class SettlementState(TypedDict):
    invoice:           dict
    flow_check_result: dict | None
    flow_status:       Literal["PROCEED", "HOLD", "BLOCKED"] | None
    settle_body:       dict | None
    settlement_result: dict | None
    recommended_token: str | None
    error:             str | None
    log:               list[str]
```

## Resources

- Docs: [docs.untitledfinancial.com](https://docs.untitledfinancial.com)
- MCP server (Claude Desktop / Cursor): `npx @untitledfinancial/dpx-mcp`
- Full agent example: [github.com/untitledfinancial/dpx-agent-public](https://github.com/untitledfinancial/dpx-agent-public)
