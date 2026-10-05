# Scripts

## CI Checks Classifier

Use `scripts/ci_checks_classifier.py` to normalize pull request check rollups into
deterministic states for maintenance automation.

### States

- `failed`
- `pending`
- `no_checks`
- `policy_blocked`

### Usage

Classify live PR checks through `gh`:

```bash
python3 scripts/ci_checks_classifier.py --pr 6927 --repo langchain-ai/langgraph
```

Classify from saved JSON output:

```bash
python3 scripts/ci_checks_classifier.py --checks-file /tmp/checks.json
```

Classify from raw `gh pr checks` text (including `no checks reported` output):

```bash
python3 scripts/ci_checks_classifier.py --gh-output-file /tmp/gh-pr-checks.txt
```

### Example output

```json
{
  "cancelled_count": 0,
  "checks_total": 2,
  "failed_count": 1,
  "merge_state_status": "BLOCKED",
  "no_checks_reported": false,
  "passing_count": 1,
  "pending_count": 0,
  "schema_version": 1,
  "skipped_count": 0,
  "state": "failed",
  "unknown_count": 0
}
```

### Real PR verification (optional)

When real PR samples are available, verify classifier behavior against live data:

```bash
cd libs/cli
LANGGRAPH_REAL_PR_FAILED=<pr-number> \
LANGGRAPH_REAL_PR_PENDING=<pr-number> \
LANGGRAPH_REAL_PR_NO_CHECKS=<pr-number> \
LANGGRAPH_REAL_PR_POLICY_BLOCKED=<pr-number> \
make test-ci-checks-classifier-real-prs
```
