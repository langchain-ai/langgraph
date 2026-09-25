import os
import shutil
import subprocess

import pytest

from langgraph_cli.ci_checks_classifier import (
    classify_ci_checks,
    parse_gh_pr_checks_output,
)

REAL_PR_ENV_MAP = {
    "failed": "LANGGRAPH_REAL_PR_FAILED",
    "pending": "LANGGRAPH_REAL_PR_PENDING",
    "no_checks": "LANGGRAPH_REAL_PR_NO_CHECKS",
    "policy_blocked": "LANGGRAPH_REAL_PR_POLICY_BLOCKED",
}


def _get_pr_number_for_state(expected_state: str) -> str:
    env_var = REAL_PR_ENV_MAP[expected_state]
    pr_number = os.environ.get(env_var)
    if not pr_number:
        pytest.skip(
            f"Missing {env_var}; set it to enable live PR verification for {expected_state}."
        )
    return pr_number


def _run_gh_pr_checks(pr_number: str) -> tuple[list[dict], bool]:
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "checks",
            pr_number,
            "--repo",
            "langchain-ai/langgraph",
            "--json",
            "name,state,bucket,link",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = proc.stdout.strip() or proc.stderr.strip()
    checks, no_checks_reported = parse_gh_pr_checks_output(payload)
    if proc.returncode != 0 and not no_checks_reported:
        pytest.fail(f"gh pr checks failed: {proc.stderr.strip()}")
    return [dict(check) for check in checks], no_checks_reported


@pytest.mark.skipif(shutil.which("gh") is None, reason="gh CLI is required")
@pytest.mark.parametrize(
    "expected_state", ["failed", "pending", "no_checks", "policy_blocked"]
)
def test_classifier_matches_real_langgraph_pr_state(expected_state: str) -> None:
    pr_number = _get_pr_number_for_state(expected_state)
    checks, no_checks_reported = _run_gh_pr_checks(pr_number)
    result = classify_ci_checks(checks, no_checks_reported=no_checks_reported)
    assert result["state"] == expected_state
