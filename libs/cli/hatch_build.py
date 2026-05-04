"""Hatch build hook that bundles the platform-specific Go binary into the wheel.

Usage:
  1. Cross-compile: make build-go-target GOOS=linux GOARCH=amd64
  2. Set LANGGRAPH_GO_BINARY to the built binary path
  3. Build wheel: uv build --wheel

The hook copies the binary into langgraph_cli/bin/ so the entrypoint can find it.
If LANGGRAPH_GO_BINARY is not set, the wheel is built without a binary (pure Python
fallback — fine for development and the legacy code path).
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class GoBinaryBuildHook(BuildHookInterface):
    PLUGIN_NAME = "go-binary"

    def initialize(self, version: str, build_data: dict) -> None:
        bin_dir = Path("langgraph_cli/bin")
        if bin_dir.exists():
            shutil.rmtree(bin_dir)

        binary_path = os.environ.get("LANGGRAPH_GO_BINARY")
        if not binary_path:
            return

        source = Path(binary_path)
        if not source.is_file():
            msg = f"LANGGRAPH_GO_BINARY points to missing file: {source}"
            raise FileNotFoundError(msg)

        bin_dir.mkdir(parents=True, exist_ok=True)

        # Determine output name (langgraph or langgraph.exe)
        dest_name = "langgraph.exe" if source.suffix == ".exe" else "langgraph"
        dest = bin_dir / dest_name

        shutil.copy2(str(source), str(dest))
        # Ensure executable permission
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        # Tell hatch to include the binary in the wheel
        build_data["shared_data"] = {}
        build_data["force_include"] = {
            str(dest): f"langgraph_cli/bin/{dest_name}",
        }

        # Set the platform tag so pip installs the right wheel
        platform_tag = os.environ.get("LANGGRAPH_WHEEL_PLAT")
        if platform_tag:
            build_data["tag"] = f"py3-none-{platform_tag}"
