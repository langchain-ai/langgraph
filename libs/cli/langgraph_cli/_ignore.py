"""Shared ignore-file handling for local source filtering."""

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


def _build_ignore_spec(
    directory: pathlib.Path, *, include_gitignore: bool = True
) -> pathspec.PathSpec:
    """Build a PathSpec combining built-in exclusions with ignore files.

    Always excludes common non-source directories (`_ALWAYS_EXCLUDE`). On top
    of that, patterns from `.dockerignore` are merged in. `.gitignore` patterns
    are optional because some callers need Docker build-context semantics,
    while archive creation wants both files.
    """
    lines: list[str] = list(_ALWAYS_EXCLUDE)
    ignore_files = [".dockerignore"]
    if include_gitignore:
        ignore_files.append(".gitignore")
    for name in ignore_files:
        ignore_file = directory / name
        if ignore_file.is_file():
            lines.extend(ignore_file.read_text(encoding="utf-8").splitlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _has_negation_patterns(spec: pathspec.PathSpec) -> bool:
    """Whether `spec` has any `!`-negated (re-include) pattern."""
    return any(getattr(p, "include", None) is False for p in spec.patterns)
