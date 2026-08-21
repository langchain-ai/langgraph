"""Generate a pip constraints file pinning a package's direct runtime
dependencies to the minimum versions declared in its pyproject.toml.

This is used by CI to install a library against the *oldest* versions of
its dependencies that it claims to support (e.g. `langchain-core==0.2.38`
instead of whatever happens to be newest on PyPI), which catches code that
silently relies on APIs added after the declared floor.

Usage:
    python get_min_versions.py path/to/pyproject.toml > min-versions-constraints.txt
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

# Matches a PEP 508 dependency string, e.g.:
#   "langchain-core>=0.2.38"
#   "psycopg[binary]>=3.1,<4"
_DEP_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(\[[^\]]*\])?\s*(.*?)\s*$")

# Matches the lower-bound clause of a version specifier, e.g. ">=1.2.3" or
# "~=1.2.3". We deliberately don't try to satisfy exclusion clauses
# (e.g. "!=1.2.4") here -- the goal is just to find the floor.
_LOWER_BOUND_RE = re.compile(r"(?:>=|~=)\s*([0-9][0-9A-Za-z.\-]*)")


def parse_dependency(dep: str) -> tuple[str, str] | None:
    """Split a PEP 508 dependency string into (name, version_spec).

    Returns None if the string can't be parsed (e.g. it's a URL or a
    local path dependency, which have no version to pin).
    """
    match = _DEP_RE.match(dep)
    if not match:
        return None
    name, _extras, version_spec = match.groups()
    return name, version_spec


def get_min_version(version_spec: str) -> str | None:
    """Return the minimum version declared in a specifier, if any.

    If a dependency has multiple lower bounds (unusual, but technically
    legal), the *highest* of them wins, since that's the effective floor.
    """
    candidates = [m.group(1) for m in _LOWER_BOUND_RE.finditer(version_spec)]
    if not candidates:
        return None
    # Compare as tuples of ints where possible so "0.10.0" > "0.9.0".
    def sort_key(v: str) -> tuple:
        return tuple(int(p) if p.isdigit() else p for p in re.split(r"[.\-]", v))

    return max(candidates, key=sort_key)


def get_min_versions(pyproject_path: Path) -> dict[str, str]:
    """Return {package_name: min_version} for every pinned direct dependency."""
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    dependencies = data.get("project", {}).get("dependencies", [])
    pins: dict[str, str] = {}
    for dep in dependencies:
        parsed = parse_dependency(dep)
        if parsed is None:
            continue
        name, version_spec = parsed
        if not version_spec:
            continue
        min_version = get_min_version(version_spec)
        if min_version:
            pins[name] = min_version
    return pins


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: get_min_versions.py <path-to-pyproject.toml>", file=sys.stderr)
        sys.exit(1)

    pyproject_path = Path(sys.argv[1])
    if not pyproject_path.is_file():
        print(f"No such file: {pyproject_path}", file=sys.stderr)
        sys.exit(1)

    pins = get_min_versions(pyproject_path)
    for name, version in sorted(pins.items()):
        print(f"{name}=={version}")


if __name__ == "__main__":
    main()
