"""Shared `.dockerignore` / `.gitignore` handling for build-context filtering."""

import pathlib

import pathspec

_ALWAYS_EXCLUDE = [
    "__pycache__/",
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    ".tox/",
    ".mypy_cache/",
]


def _build_ignore_spec(directory: pathlib.Path) -> pathspec.PathSpec:
    """Build a PathSpec combining built-in exclusions with .dockerignore and .gitignore.

    Always excludes common non-source directories (`_ALWAYS_EXCLUDE`).  On top
    of that, patterns from .dockerignore and .gitignore (if present) are merged
    in.
    """
    lines: list[str] = list(_ALWAYS_EXCLUDE)
    for name in (".dockerignore", ".gitignore"):
        ignore_file = directory / name
        if ignore_file.is_file():
            lines.extend(ignore_file.read_text(encoding="utf-8").splitlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)
