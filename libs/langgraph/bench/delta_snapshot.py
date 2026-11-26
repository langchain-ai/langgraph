"""Build a dedicated LangSmith benchmark snapshot from a minimal Docker context."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from langsmith.sandbox import SandboxClient

CONTEXT = Path(__file__).resolve().with_name("delta-snapshot")


def snapshot_plan(args: argparse.Namespace) -> dict:
    """Validate resource bounds and pin the public repository to a full SHA."""
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Revision must be a full lowercase 40-character commit SHA")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.name):
        raise ValueError(
            "Name must contain only lowercase letters, numbers and hyphens"
        )
    if not (
        1 <= args.vcpus <= 16
        and 1 <= args.memory_gib <= 64
        and 10 <= args.disk_gib <= 100
        and 60 <= args.timeout <= 7200
    ):
        raise ValueError(
            "Build limits: CPU 1–16, memory 1–64 GiB, disk 10–100 GiB, timeout 60–7200 seconds"
        )
    if sorted(p.name for p in CONTEXT.iterdir()) != ["Dockerfile"]:
        raise ValueError(
            "Build context must contain only Dockerfile; refusing other files"
        )
    return {
        "name": args.name,
        "revision": args.revision,
        "dockerfile_sha256": hashlib.sha256(
            (CONTEXT / "Dockerfile").read_bytes()
        ).hexdigest(),
        "vcpus": args.vcpus,
        "mem_bytes": args.memory_gib * 1024**3,
        "fs_capacity_bytes": args.disk_gib * 1024**3,
        "timeout": args.timeout,
    }


def build_snapshot(plan: dict, client_factory=SandboxClient) -> dict:
    """Create a billable builder, then return the reusable snapshot receipt."""
    with client_factory() as client:
        snapshot = client.create_snapshot_from_dockerfile(
            name=plan["name"],
            dockerfile="Dockerfile",
            context=CONTEXT,
            fs_capacity_bytes=plan["fs_capacity_bytes"],
            build_args={"LANGGRAPH_REVISION": plan["revision"]},
            vcpus=plan["vcpus"],
            mem_bytes=plan["mem_bytes"],
            timeout=plan["timeout"],
        )
    return {**plan, "snapshot_id": snapshot.id, "status": snapshot.status}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--revision",
        default=None,
        help="Full public langgraph commit SHA; defaults to local HEAD",
    )
    parser.add_argument("--vcpus", type=int, default=2)
    parser.add_argument("--memory-gib", type=int, default=8)
    parser.add_argument("--disk-gib", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create a billable builder and persistent snapshot",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Output already exists; use a fresh receipt path to avoid duplicate builds"
        )
    if args.revision is None:
        args.revision = subprocess.check_output(
            ["git", "-C", str(CONTEXT), "rev-parse", "HEAD"], text=True
        ).strip()
    try:
        plan = snapshot_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**plan, "state": "planned"}, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print(
            "Dry run: no resources created. Use --execute and a fresh receipt path to build."
        )
        return
    args.output.write_text(json.dumps({**plan, "state": "building"}, indent=2) + "\n")
    receipt = build_snapshot(plan)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Snapshot ready: {receipt['snapshot_id']}")


if __name__ == "__main__":
    main()
