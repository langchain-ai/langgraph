import json
from pathlib import Path

import pytest

from langgraph_cli.ci_checks_classifier import classify_gh_pr_checks_output

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "ci_checks_classifier_cases.json"


def _load_cases() -> list[dict]:
    with FIXTURES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["name"])
def test_classify_gh_pr_checks_output_cases(case: dict) -> None:
    result = classify_gh_pr_checks_output(case["raw_output"])
    assert result["schema_version"] == 1
    assert result["state"] == case["expected_state"]
    assert result["checks_total"] == case["expected"]["checks_total"]
    assert result["failed_count"] == case["expected"]["failed_count"]
    assert result["pending_count"] == case["expected"]["pending_count"]
    assert result["passing_count"] == case["expected"]["passing_count"]
    assert result["skipped_count"] == case["expected"]["skipped_count"]
    assert result["cancelled_count"] == case["expected"]["cancelled_count"]
    assert result["unknown_count"] == case["expected"]["unknown_count"]
    assert result["no_checks_reported"] == case["expected"]["no_checks_reported"]


def test_classify_output_schema_keys_are_stable() -> None:
    result = classify_gh_pr_checks_output("[]")
    assert list(result.keys()) == [
        "schema_version",
        "state",
        "checks_total",
        "failed_count",
        "pending_count",
        "passing_count",
        "skipped_count",
        "cancelled_count",
        "unknown_count",
        "no_checks_reported",
        "merge_state_status",
    ]
