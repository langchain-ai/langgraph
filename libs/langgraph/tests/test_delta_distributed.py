import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "delta_distributed",
    Path(__file__).resolve().parents[1] / "bench/delta_distributed.py",
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_plan_grid():
    args = SimpleNamespace(
        snapshot_id="snapshot",
        concurrency=4,
        timeout=3600,
        vcpus=2,
        memory_gib=8,
        disk_gib=30,
        lengths=list(bench.LENGTHS),
        profiles=list(bench.PROFILES),
        frequencies=list(bench.FREQUENCIES),
    )
    plan = bench.make_plan(args)
    assert len(plan["shards"]) == 40
    assert len({s["key"] for s in plan["shards"]}) == 40
    assert plan["mem_bytes"] == 8 * 1024**3


@pytest.mark.parametrize("command_fails", [False, True])
def test_shard_cleanup_resume_and_validation(tmp_path, command_fails):
    shard = {"key": "mock700-25", "profile": "mock700", "frequency": 25}
    plan = {
        "snapshot_id": "test",
        "timeout_seconds": 10,
        "vcpus": 2,
        "mem_bytes": 1024,
        "fs_capacity_bytes": 1024,
        "lengths": [50],
        "source_sha256": "hash",
    }
    data = b"backend,profile,frequency,updates,read_p50_ms,read_p95_ms,read_max_ms,storage_bytes\npostgres,mock700,25,50,1,2,3,400\n"
    deleted = []

    class FakeSandbox:
        def write(self, path, content):
            assert "API_KEY" not in str(content)

        def run(self, *args, **kwargs):
            return SimpleNamespace(exit_code=1 if command_fails else 0)

        def read(self, path):
            return (
                json.dumps({"script_sha256": "hash"}).encode()
                if path.endswith("json")
                else data
            )

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create_sandbox(self, **kwargs):
            return FakeSandbox()

        def delete_sandbox(self, name):
            deleted.append(name)

    if command_fails:
        with pytest.raises(RuntimeError, match="Shard command failed"):
            bench.run_shard(plan, shard, tmp_path, FakeClient)
        assert len(deleted) == 1
        assert not (tmp_path / shard["key"] / "success.json").exists()
        return
    bench.run_shard(plan, shard, tmp_path, FakeClient)
    bench.run_shard(plan, shard, tmp_path, FakeClient)
    assert len(deleted) == 1
    assert (
        json.loads((tmp_path / shard["key"] / "attempt.json").read_text())["state"]
        == "deleted"
    )
    with pytest.raises(ValueError, match="Incomplete"):
        bench.validate_csv(data, shard, [50, 200])
    with pytest.raises(ValueError, match="Foreign"):
        bench.validate_csv(data, {**shard, "frequency": 50}, [50])


def test_failed_shard_is_deleted(tmp_path):
    shard = {"key": "mock700-25", "profile": "mock700", "frequency": 25}
    target = tmp_path / shard["key"]
    target.mkdir()
    (target / "attempt.json").write_text(
        json.dumps({"state": "running", "sandbox_name": "previous"})
    )
    with pytest.raises(RuntimeError, match="previous sandbox"):
        bench.run_shard({}, shard, tmp_path)
    assert not (target / "success.json").exists()
