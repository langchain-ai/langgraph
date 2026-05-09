# AWS Infrastructure Audit Agent

A stateful LangGraph pipeline that audits AWS infrastructure from a plain-English request.

## Architecture

Four nodes with conditional routing — the graph takes different paths depending on what `discover` finds:

```
START → plan → discover → [route_after_discovery] → deep_dive → report → END
                                                  ↘ report → END (clean account)
```

A clean account skips `deep_dive` entirely, reducing cost by ~40% on low-violation scans.

## Concepts demonstrated

- `StateGraph` with typed `AuditState` (Pydantic v2)
- Conditional edges based on runtime state (`route_after_discovery`)
- `MemorySaver` checkpointing — audit resumes from last completed node if interrupted
- Separation of `findings` (full record) and `violations` (routing subset)
- `DEMO_MODE` at the tool level — swap any tool to real boto3 without touching graph logic

## Run

```bash
pip install langgraph langchain-core anthropic pydantic
python aws_infrastructure_audit.py "check IAM for MFA issues"
python aws_infrastructure_audit.py "scan all services for security issues"
```

`DEMO_MODE=true` by default — no AWS credentials or API key needed.

## Source

Extracted from [ai-sentinel-ecosystem](https://github.com/TanishkaMarrott/ai-sentinel-ecosystem) — a production system governing AWS cloud lab accounts.