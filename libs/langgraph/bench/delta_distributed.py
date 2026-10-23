"""Plan, execute and merge isolated LangSmith sandbox benchmark shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import shlex
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langsmith.sandbox import SandboxClient

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tests/test_delta_channel_benchmark.py"
REMOTE = "/workspace/langgraph/libs/langgraph"
PROFILES = ("mock700", "tool_calls")
FREQUENCIES = tuple(range(25, 501, 25))
LENGTHS = (50, 200, 1000, 5000, 10000)


def _atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def make_plan(args: argparse.Namespace) -> dict:
    """Bind every shard to source, snapshot, resource and runtime settings."""
    return {
        "schema": 1,
        "snapshot_id": args.snapshot_id,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "vcpus": args.vcpus,
        "mem_bytes": args.memory_gib * 1024**3,
        "fs_capacity_bytes": args.disk_gib * 1024**3,
        "source_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(
            Path(__file__).with_name("delta_shard.py").read_bytes()
        ).hexdigest(),
        "lengths": sorted(set(args.lengths)),
        "shards": [
            {"key": f"{profile}-{freq}", "profile": profile, "frequency": freq}
            for profile, freq in itertools.product(args.profiles, args.frequencies)
        ],
    }


def validate_csv(data: bytes, shard: dict, lengths: list[int]) -> list[dict]:
    """Reject missing, duplicate, foreign or malformed measurement points."""
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    actual = []
    for row in rows:
        if (
            row["backend"] != "postgres"
            or row["profile"] != shard["profile"]
            or int(row["frequency"]) != shard["frequency"]
        ):
            raise ValueError("Foreign shard row")
        actual.append(int(row["updates"]))
        for metric in ("read_p50_ms", "read_p95_ms", "read_max_ms", "storage_bytes"):
            value = float(row[metric])
            if not 0 <= value < float("inf"):
                raise ValueError("Nonfinite or negative metric")
    if sorted(actual) != sorted(lengths):
        raise ValueError("Incomplete or duplicate shard lengths")
    return rows


def run_shard(
    plan: dict, shard: dict, directory: Path, client_factory=SandboxClient
) -> None:
    """Use a disposable sandbox and save validated outputs before deleting it."""
    target = directory / shard["key"]
    target.mkdir(exist_ok=True)
    success = target / "success.json"
    if success.exists():
        validate_csv((target / "results.csv").read_bytes(), shard, plan["lengths"])
        if json.loads(success.read_text())["source_sha256"] != plan["source_sha256"]:
            raise ValueError("Completed shard source differs")
        return
    ledger = target / "attempt.json"
    if ledger.exists() and json.loads(ledger.read_text())["state"] != "deleted":
        raise RuntimeError(
            f"Inspect and clean up the previous sandbox in {ledger} before retrying"
        )
    sandbox_name = f"delta-bench-{uuid.uuid4().hex}"
    _atomic_json(ledger, {"sandbox_name": sandbox_name, "state": "creating"})
    with client_factory() as client:
        sandbox = None
        completed = False
        try:
            sandbox = client.create_sandbox(
                snapshot_id=plan["snapshot_id"],
                name=sandbox_name,
                timeout=300,
                idle_ttl_seconds=600,
                delete_after_stop_seconds=600,
                vcpus=plan["vcpus"],
                mem_bytes=plan["mem_bytes"],
                fs_capacity_bytes=plan["fs_capacity_bytes"],
            )
            _atomic_json(ledger, {"sandbox_name": sandbox_name, "state": "running"})
            sandbox.write(
                f"{REMOTE}/tests/test_delta_channel_benchmark.py",
                BENCHMARK.read_bytes(),
            )
            sandbox.write(
                "/tmp/delta_shard.py",
                Path(__file__).with_name("delta_shard.py").read_bytes(),
            )
            sandbox.write(
                "/tmp/delta-job.json",
                json.dumps(
                    {
                        **shard,
                        "lengths": plan["lengths"],
                        "timeout_seconds": plan["timeout_seconds"],
                    }
                ),
            )
            result = sandbox.run(
                shlex.join([f"{REMOTE}/.venv/bin/python", "/tmp/delta_shard.py"]),
                timeout=plan["timeout_seconds"] + 120,
                cwd=REMOTE,
                kill_on_disconnect=True,
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    "Shard command failed; sandbox logs withheld from shared output"
                )
            data = sandbox.read("/tmp/delta-results.csv")
            metadata = json.loads(sandbox.read("/tmp/delta-results.metadata.json"))
            if metadata["script_sha256"] != plan["source_sha256"]:
                raise ValueError("Remote benchmark source differs")
            validate_csv(data, shard, plan["lengths"])
            (target / "results.csv.tmp").write_bytes(data)
            (target / "results.csv.tmp").replace(target / "results.csv")
            _atomic_json(target / "metadata.json", metadata)
            completed = True
        finally:
            try:
                client.delete_sandbox(sandbox_name)
            except Exception:
                _atomic_json(
                    ledger, {"sandbox_name": sandbox_name, "state": "cleanup_required"}
                )
                raise RuntimeError(f"Sandbox cleanup required; see {ledger}") from None
            _atomic_json(ledger, {"sandbox_name": sandbox_name, "state": "deleted"})
        if completed:
            _atomic_json(
                success,
                {"source_sha256": plan["source_sha256"], "sandbox_name": sandbox_name},
            )


def merge(plan: dict, directory: Path) -> Path:
    """Require the entire planned grid before publishing a combined CSV."""
    rows = []
    for shard in plan["shards"]:
        target = directory / shard["key"]
        if not (target / "success.json").exists():
            raise ValueError(f"Shard incomplete: {shard['key']}")
        rows.extend(
            validate_csv((target / "results.csv").read_bytes(), shard, plan["lengths"])
        )
    output = directory / "combined.csv"
    temp = output.with_suffix(".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output)
    spec = importlib.util.spec_from_file_location("delta_benchmark", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._plot_sweep(output)
    return output


def execute(plan: dict, directory: Path, client_factory=SandboxClient) -> None:
    """Bound active sandboxes and stop admitting shards after a failure."""
    failed = threading.Event()

    def run(shard: dict) -> None:
        if failed.is_set():
            return
        try:
            run_shard(plan, shard, directory, client_factory)
        except Exception as exc:
            failed.set()
            print(f"{shard['key']}: {type(exc).__name__}; see shard ledger", flush=True)
            raise

    with ThreadPoolExecutor(max_workers=plan["concurrency"]) as pool:
        list(pool.map(run, plan["shards"]))
    merge(plan, directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument(
        "--timeout", type=int, required=True, help="Maximum benchmark seconds per shard"
    )
    parser.add_argument("--vcpus", type=int, required=True)
    parser.add_argument("--memory-gib", type=int, required=True)
    parser.add_argument("--disk-gib", type=int, required=True)
    parser.add_argument("--frequencies", nargs="+", type=int, default=list(FREQUENCIES))
    parser.add_argument("--lengths", nargs="+", type=int, default=list(LENGTHS))
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=list(PROFILES)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create billable sandboxes; otherwise only write a plan",
    )
    args = parser.parse_args()
    if any(
        n <= 0
        for n in [
            args.concurrency,
            args.timeout,
            args.vcpus,
            args.memory_gib,
            args.disk_gib,
            *args.frequencies,
            *args.lengths,
        ]
    ):
        parser.error("Resource limits, frequencies and lengths must be positive")
    if args.concurrency > 40 or args.timeout > 86400:
        parser.error("Concurrency must be <=40 and shard timeout <=86400 seconds")
    if len(set(args.frequencies)) != len(args.frequencies) or len(
        set(args.profiles)
    ) != len(args.profiles):
        parser.error("Duplicate shard parameters")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan = make_plan(args)
    plan_path = directory / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        parser.error("Existing plan differs; use a new output directory")
    _atomic_json(plan_path, plan)
    print(
        f"{len(plan['shards'])} shards; concurrency {args.concurrency}; upper bound {len(plan['shards']) * (args.timeout + 420) / 3600:.2f} sandbox-hours excluding deletion; execute={args.execute}"
    )
    if args.execute:
        execute(plan, directory)


if __name__ == "__main__":
    main()
