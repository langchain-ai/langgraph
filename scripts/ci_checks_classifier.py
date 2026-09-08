#!/usr/bin/env python3
"""Classify PR checks into deterministic automation states.

Examples:
  python3 scripts/ci_checks_classifier.py --pr 123 --repo langchain-ai/langgraph
  python3 scripts/ci_checks_classifier.py --checks-file ./checks.json
  python3 scripts/ci_checks_classifier.py --gh-output-file ./gh-pr-checks.txt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI_LIB_DIR = REPO_ROOT / "libs" / "cli"
if str(CLI_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_LIB_DIR))

from langgraph_cli.ci_checks_classifier import (  # noqa: E402
    classify_ci_checks,
    classify_gh_pr_checks_output,
    parse_gh_pr_checks_output,
)


def _load_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_gh_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _fetch_merge_state_status(pr: str, repo: str) -> str | None:
    proc = _run_gh_command(
        ["gh", "pr", "view", pr, "--repo", repo, "--json", "mergeStateStatus"]
    )
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    merge_state_status = payload.get("mergeStateStatus")
    return merge_state_status if isinstance(merge_state_status, str) else None


def _fetch_checks_payload(pr: str, repo: str) -> tuple[list[dict[str, Any]], bool]:
    proc = _run_gh_command(
        ["gh", "pr", "checks", pr, "--repo", repo, "--json", "name,state,bucket,link"]
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if stdout:
        try:
            checks, no_checks_reported = parse_gh_pr_checks_output(stdout)
            return [dict(check) for check in checks], no_checks_reported
        except (ValueError, json.JSONDecodeError):
            # Fall through to sentinel/error handling below.
            pass

    if stderr:
        try:
            checks, no_checks_reported = parse_gh_pr_checks_output(stderr)
            if no_checks_reported:
                return [dict(check) for check in checks], no_checks_reported
        except (ValueError, json.JSONDecodeError):
            pass

    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to fetch checks via gh pr checks: "
            f"exit={proc.returncode}; stderr={stderr}"
        )

    if not stdout:
        return [], True

    raise RuntimeError("Unable to parse `gh pr checks` JSON output")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize PR check rollups into a deterministic JSON state for automation."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--pr", help="Pull request number for live `gh` lookup.")
    source_group.add_argument(
        "--checks-file",
        type=pathlib.Path,
        help="Path to JSON output from `gh pr checks --json ...`.",
    )
    source_group.add_argument(
        "--gh-output-file",
        type=pathlib.Path,
        help="Path to raw text output from `gh pr checks`.",
    )
    parser.add_argument(
        "--repo",
        default="langchain-ai/langgraph",
        help="GitHub repository in OWNER/REPO form (used with --pr).",
    )
    parser.add_argument(
        "--merge-state-status",
        default=None,
        help="Optional merge state status (for metadata enrichment).",
    )

    args = parser.parse_args()

    merge_state_status = args.merge_state_status
    if args.pr is not None:
        checks, no_checks_reported = _fetch_checks_payload(args.pr, args.repo)
        if merge_state_status is None:
            merge_state_status = _fetch_merge_state_status(args.pr, args.repo)
        result = classify_ci_checks(
            checks,
            no_checks_reported=no_checks_reported,
            merge_state_status=merge_state_status,
        )
    elif args.checks_file is not None:
        checks_payload = _load_text(args.checks_file)
        checks, no_checks_reported = parse_gh_pr_checks_output(checks_payload)
        result = classify_ci_checks(
            checks,
            no_checks_reported=no_checks_reported,
            merge_state_status=merge_state_status,
        )
    else:
        gh_output = _load_text(args.gh_output_file)
        result = classify_gh_pr_checks_output(
            gh_output, merge_state_status=merge_state_status
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
