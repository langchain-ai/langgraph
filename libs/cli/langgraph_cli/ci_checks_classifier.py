"""Normalize PR check rollups into deterministic automation states."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

NormalizedCIState = Literal["failed", "pending", "no_checks", "policy_blocked"]

_NO_CHECKS_SENTINEL = "no checks reported"

_FAILED_TOKENS = {
    "fail",
    "failed",
    "failure",
    "error",
    "timed_out",
    "timedout",
    "action_required",
    "startup_failure",
}
_PENDING_TOKENS = {
    "pending",
    "queued",
    "in_progress",
    "inprogress",
    "requested",
    "waiting",
}
_PASSING_TOKENS = {"pass", "passed", "success", "successful", "neutral"}
_SKIPPED_TOKENS = {"skip", "skipped", "skipping"}
_CANCELLED_TOKENS = {"cancel", "cancelled", "canceled"}


class CICheckClassification(TypedDict):
    schema_version: int
    state: NormalizedCIState
    checks_total: int
    failed_count: int
    pending_count: int
    passing_count: int
    skipped_count: int
    cancelled_count: int
    unknown_count: int
    no_checks_reported: bool
    merge_state_status: str | None


def _as_lower_token(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token or None


def _classify_check_bucket(check: Mapping[str, Any]) -> str:
    for raw_token in (check.get("bucket"), check.get("state")):
        token = _as_lower_token(raw_token)
        if token is None:
            continue
        if token in _FAILED_TOKENS:
            return "failed"
        if token in _PENDING_TOKENS:
            return "pending"
        if token in _PASSING_TOKENS:
            return "passing"
        if token in _SKIPPED_TOKENS:
            return "skipped"
        if token in _CANCELLED_TOKENS:
            return "cancelled"
    return "unknown"


def parse_gh_pr_checks_output(raw_output: str) -> tuple[list[Mapping[str, Any]], bool]:
    """Parse `gh pr checks` output into check rows and no-checks signal."""
    stripped = raw_output.strip()
    if not stripped:
        return [], True
    if _NO_CHECKS_SENTINEL in stripped.lower():
        return [], True

    payload = json.loads(stripped)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)], False

    if isinstance(payload, Mapping):
        checks = payload.get("checks")
        if isinstance(checks, list):
            return [row for row in checks if isinstance(row, Mapping)], False

    raise ValueError("Unsupported checks payload format")


def classify_ci_checks(
    checks: Sequence[Mapping[str, Any]],
    *,
    no_checks_reported: bool = False,
    merge_state_status: str | None = None,
) -> CICheckClassification:
    """Classify check rollup data into a deterministic automation state."""
    failed_count = 0
    pending_count = 0
    passing_count = 0
    skipped_count = 0
    cancelled_count = 0
    unknown_count = 0

    for check in checks:
        bucket = _classify_check_bucket(check)
        if bucket == "failed":
            failed_count += 1
        elif bucket == "pending":
            pending_count += 1
        elif bucket == "passing":
            passing_count += 1
        elif bucket == "skipped":
            skipped_count += 1
        elif bucket == "cancelled":
            cancelled_count += 1
        else:
            unknown_count += 1

    checks_total = len(checks)

    if no_checks_reported or checks_total == 0:
        state: NormalizedCIState = "no_checks"
    elif failed_count > 0 or cancelled_count > 0:
        state = "failed"
    elif pending_count > 0:
        state = "pending"
    else:
        # Checks are non-blocking; the remaining blocker is repository policy
        # (merge requirements, reviews, queue rules, etc.).
        state = "policy_blocked"

    return {
        "schema_version": 1,
        "state": state,
        "checks_total": checks_total,
        "failed_count": failed_count,
        "pending_count": pending_count,
        "passing_count": passing_count,
        "skipped_count": skipped_count,
        "cancelled_count": cancelled_count,
        "unknown_count": unknown_count,
        "no_checks_reported": no_checks_reported or checks_total == 0,
        "merge_state_status": merge_state_status,
    }


def classify_gh_pr_checks_output(
    raw_output: str, *, merge_state_status: str | None = None
) -> CICheckClassification:
    """Classify raw `gh pr checks` output."""
    checks, no_checks_reported = parse_gh_pr_checks_output(raw_output)
    return classify_ci_checks(
        checks,
        no_checks_reported=no_checks_reported,
        merge_state_status=merge_state_status,
    )
