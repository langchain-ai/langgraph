import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from langsmith.sandbox import Sandbox, SandboxClient

SPEC = importlib.util.spec_from_file_location(
    "delta_snapshot", Path(__file__).resolve().parents[1] / "bench/delta_snapshot.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_snapshot_build_contract():
    args = SimpleNamespace(
        name="delta-bench",
        revision="a" * 40,
        vcpus=2,
        memory_gib=8,
        disk_gib=30,
        timeout=1800,
    )
    plan = builder.snapshot_plan(args)
    client = create_autospec(SandboxClient, instance=True)
    client.__enter__.return_value = client
    client.create_snapshot_from_dockerfile.return_value = SimpleNamespace(
        id="snapshot-id", status="ready"
    )
    receipt = builder.build_snapshot(plan, lambda: client)
    assert receipt["snapshot_id"] == "snapshot-id"
    call = client.create_snapshot_from_dockerfile.call_args.kwargs
    assert call["build_args"] == {"LANGGRAPH_REVISION": "a" * 40}
    assert sorted(p.name for p in call["context"].iterdir()) == ["Dockerfile"]
    assert call["timeout"] == 1800
    client.__exit__.assert_called_once()
    args.revision = "main; echo nope"
    with pytest.raises(ValueError, match="40-character"):
        builder.snapshot_plan(args)


def test_sandbox_sdk_file_api():
    sandbox = create_autospec(Sandbox, instance=True)
    sandbox.write("/tmp/test", b"payload")
    sandbox.read("/tmp/test")
    with pytest.raises(AttributeError):
        sandbox.write_file("/tmp/test", b"payload")
