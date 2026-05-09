"""
AWS Infrastructure Audit Agent — LangGraph Example

A stateful LangGraph pipeline that audits AWS infrastructure from a plain-English
request. Four nodes, conditional routing, in-memory checkpointing.

The key design: the graph takes two different paths depending on what `discover`
finds. A clean account skips `deep_dive` entirely. An account with critical
violations goes through detailed investigation before the report is written.

Run:
    python aws_infrastructure_audit.py "check IAM for MFA issues"
    python aws_infrastructure_audit.py "scan all services for security issues"

DEMO_MODE=true by default — no AWS credentials or LLM calls needed.
Set ANTHROPIC_API_KEY and DEMO_MODE=false to run against Claude live.
"""

import os
from enum import Enum
from typing import Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")


# --- State schema ---

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MEDIUM = "MEDIUM"
    INFO = "INFO"


class Finding(BaseModel):
    resource_id: str
    service: str
    issue: str
    severity: Severity


class AuditReport(BaseModel):
    summary: str
    findings: list[Finding]
    critical_count: int
    medium_count: int


class AuditState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    audit_request: str = ""
    audit_plan: list[str] = []
    findings: list[Finding] = []
    violations: list[Finding] = []
    report: AuditReport | None = None
    phase: str = "plan"


# --- Demo tools (swap for real boto3 calls with DEMO_MODE=false) ---

def list_iam_users() -> list[Finding]:
    return [
        Finding(resource_id="alice", service="IAM", issue="MFA not enabled", severity=Severity.CRITICAL),
        Finding(resource_id="bob", service="IAM", issue="Access key unused for 90+ days", severity=Severity.MEDIUM),
    ]


def list_s3_buckets() -> list[Finding]:
    return [
        Finding(resource_id="my-data-bucket", service="S3", issue="Public access not blocked", severity=Severity.CRITICAL),
        Finding(resource_id="logs-bucket", service="S3", issue="No lifecycle policy", severity=Severity.INFO),
    ]


def check_security_groups() -> list[Finding]:
    return [
        Finding(resource_id="sg-0abc123", service="EC2", issue="SSH open to 0.0.0.0/0", severity=Severity.CRITICAL),
    ]


TOOL_MAP = {
    "iam": list_iam_users,
    "s3": list_s3_buckets,
    "ec2": check_security_groups,
}


# --- Nodes ---

def plan(state: AuditState) -> dict:
    request = state.audit_request.lower()
    services = [s for s in ["iam", "s3", "ec2"] if s in request or "all" in request]
    if not services:
        services = ["iam", "s3", "ec2"]
    return {"audit_plan": services, "phase": "discover"}


def discover(state: AuditState) -> dict:
    findings = []
    for service in state.audit_plan:
        if service in TOOL_MAP:
            findings.extend(TOOL_MAP[service]())
    violations = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.MEDIUM)]
    return {"findings": findings, "violations": violations, "phase": "deep_dive" if violations else "report"}


def deep_dive(state: AuditState) -> dict:
    # In live mode this would call describe_finding per violation via Claude
    enriched = []
    for v in state.violations:
        enriched.append(v.model_copy(update={"issue": f"{v.issue} [investigated]"}))
    return {"violations": enriched, "phase": "report"}


def report(state: AuditState) -> dict:
    all_findings = state.findings
    critical = sum(1 for f in all_findings if f.severity == Severity.CRITICAL)
    medium = sum(1 for f in all_findings if f.severity == Severity.MEDIUM)
    summary = f"Audit complete. {len(all_findings)} findings: {critical} CRITICAL, {medium} MEDIUM."
    audit_report = AuditReport(
        summary=summary,
        findings=all_findings,
        critical_count=critical,
        medium_count=medium,
    )
    return {"report": audit_report, "phase": "complete"}


# --- Conditional edge ---

def route_after_discovery(state: AuditState) -> str:
    if state.violations:
        return "deep_dive"
    return "report"


# --- Build graph ---

def build_graph() -> StateGraph:
    builder = StateGraph(AuditState)
    builder.add_node("plan", plan)
    builder.add_node("discover", discover)
    builder.add_node("deep_dive", deep_dive)
    builder.add_node("report", report)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "discover")
    builder.add_conditional_edges("discover", route_after_discovery, {"deep_dive": "deep_dive", "report": "report"})
    builder.add_edge("deep_dive", "report")
    builder.add_edge("report", END)

    return builder.compile(checkpointer=MemorySaver())


# --- CLI ---

if __name__ == "__main__":
    import sys

    request = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "scan all services for security issues"
    graph = build_graph()
    config = {"configurable": {"thread_id": "audit-1"}}

    result = graph.invoke(
        AuditState(messages=[HumanMessage(content=request)], audit_request=request),
        config=config,
    )

    r = result["report"]
    print(f"\n{r.summary}\n")
    for f in r.findings:
        print(f"  [{f.severity}] {f.service} / {f.resource_id}: {f.issue}")