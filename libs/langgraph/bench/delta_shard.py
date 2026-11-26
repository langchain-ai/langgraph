"""Execute one scripted benchmark with a sandbox-local disposable Postgres."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/langgraph/libs/langgraph")
PGDATA = Path("/tmp/delta-bench-pg")


def checked(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, **kwargs)


def main() -> None:
    job = json.loads(Path("/tmp/delta-job.json").read_text())
    prefix = ["runuser", "-u", "postgres", "--"] if os.getuid() == 0 else []
    for binary in ("initdb", "pg_ctl"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"Snapshot must provide {binary} on PATH")
    PGDATA.mkdir(mode=0o700)
    if prefix:
        checked(["chown", "postgres:postgres", str(PGDATA)])
    checked(
        [
            *prefix,
            "initdb",
            "-D",
            str(PGDATA),
            "--auth-local=trust",
            "--auth-host=trust",
            "-U",
            "postgres",
        ]
    )
    started = False
    try:
        checked(
            [
                *prefix,
                "pg_ctl",
                "-D",
                str(PGDATA),
                "-l",
                "/tmp/delta-postgres.log",
                "-o",
                "-h 127.0.0.1 -p 5432 -k /tmp",
                "-w",
                "start",
            ]
        )
        started = True
        command = [
            sys.executable,
            str(ROOT / "tests/test_delta_channel_benchmark.py"),
            "--snapshot-sweep",
            "--backend",
            "postgres",
            "--frequencies",
            str(job["frequency"]),
            "--profiles",
            job["profile"],
            "--lengths",
            *map(str, job["lengths"]),
            "--output",
            "/tmp/delta-results.csv",
        ]
        env = {
            **os.environ,
            "LANGGRAPH_BENCH_POSTGRES_URI": "postgresql://postgres@127.0.0.1:5432/postgres?sslmode=disable",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "NO_COLOR": "1",
        }
        subprocess.run(
            command, cwd=ROOT, env=env, timeout=job["timeout_seconds"], check=True
        )
    finally:
        if started:
            checked(
                [*prefix, "pg_ctl", "-D", str(PGDATA), "-m", "immediate", "-w", "stop"]
            )


if __name__ == "__main__":
    main()
