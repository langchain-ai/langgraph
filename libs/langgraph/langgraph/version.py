"""
Exports package version.

This module tries to determine robustly the version of the langgraph package.

1. Primary: `importlib.metadata.version(__package__)` works for normal
   installed environments.
2. Fallback: Parse `pyproject.toml` (PEP 621) that lives two directories
   above this file. This covers editable installs or source checkouts
   where no package metadata has been generated yet.

If both strategies fail we expose `__version__ = ""` (empty string) so
that downstream code can handle the "unknown-version" case explicitly.
"""

from __future__ import annotations

import importlib.metadata as _metadata
from pathlib import Path

try:
    # Normal installation path – package metadata is available.
    __version__ = _metadata.version(__package__)
except _metadata.PackageNotFoundError:  # pragma: no cover – dev env only
    # Fallback for editable installs / CI where metadata isn't generated.
    _pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if _pyproject.is_file():
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover – <3.11 or missing
            try:
                import tomli as tomllib  # type: ignore
            except ModuleNotFoundError:
                tomllib = None  # type: ignore

        if tomllib is not None:
            try:
                with _pyproject.open("rb") as _fp:
                    _project_table = tomllib.load(_fp).get("project", {})
                    __version__ = _project_table.get("version", "") or ""
            except Exception:  # pragma: no cover – parsing errors
                __version__ = ""
        else:
            __version__ = ""
    else:
        __version__ = ""

# Clean-up internals to avoid polluting the public namespace.
del _metadata  # pyright: ignore[reportGeneralTypeIssues]
try:
    del tomllib  # type: ignore  # noqa: F401
except NameError:
    pass
