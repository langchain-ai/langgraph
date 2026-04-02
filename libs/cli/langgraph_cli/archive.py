"""Create a tarball of project source for remote builds."""

import os
import pathlib
import tarfile
import tempfile
from contextlib import contextmanager

import click
import pathspec

from langgraph_cli.config import Config, _assemble_local_deps

_WARN_SIZE = 50 * 1024 * 1024  # 50 MB
_MAX_SIZE = 200 * 1024 * 1024  # 200 MB

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

    Always excludes common non-source directories (_ALWAYS_EXCLUDE).  On top of
    that, patterns from .dockerignore and .gitignore (if present) are merged in.
    """
    lines: list[str] = list(_ALWAYS_EXCLUDE)
    for name in (".dockerignore", ".gitignore"):
        ignore_file = directory / name
        if ignore_file.is_file():
            lines.extend(ignore_file.read_text(encoding="utf-8").splitlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Strip symlinks, hardlinks, and traversal paths from archive."""
    if tarinfo.issym() or tarinfo.islnk():
        return None
    if ".." in tarinfo.name.split("/"):
        return None
    return tarinfo


def _add_directory(
    tar: tarfile.TarFile,
    source_dir: pathlib.Path,
    arcname_prefix: str | None,
    ignore_spec: pathspec.PathSpec,
) -> None:
    """Recursively add a directory to the tarball under the given prefix.

    If arcname_prefix is None, files are added at the archive root.
    Paths matching ignore_spec are excluded.
    """
    for root, dirs, files in os.walk(source_dir):
        rel_root = os.path.relpath(root, source_dir).replace(os.sep, "/")
        dirs[:] = [
            d
            for d in dirs
            if not ignore_spec.match_file(
                f"{rel_root}/{d}/" if rel_root != "." else f"{d}/"
            )
        ]
        for f in files:
            full_path = os.path.join(root, f)
            rel = os.path.relpath(full_path, source_dir).replace(os.sep, "/")
            if ignore_spec.match_file(rel):
                continue
            arcname = f"{arcname_prefix}/{rel}" if arcname_prefix else rel
            info = tar.gettarinfo(full_path, arcname=arcname)
            filtered = _tar_filter(info)
            if filtered is None:
                continue
            with open(full_path, "rb") as fobj:
                tar.addfile(filtered, fobj)


@contextmanager
def create_archive(
    config_path: pathlib.Path,
    config: Config,
):
    """Context manager that creates a .tar.gz archive of the project source.

    Uses _assemble_local_deps to discover local dependencies referenced in
    langgraph.json, including those outside config.parent (monorepo case).

    The archive preserves the real filesystem layout relative to the common
    ancestor of config.parent and all external dependency directories, so that
    relative references (e.g. `../shared-lib`) resolve correctly after
    extraction.

    Yields (archive_path, file_size, config_relative_path).  The temporary
    directory holding the archive is cleaned up automatically on exit.
    """
    config_path = config_path.resolve()
    context_dir = config_path.parent

    local_deps = _assemble_local_deps(config_path, config)
    extra_contexts = local_deps.additional_contexts or []

    dirs_to_include = [context_dir] + list(extra_contexts)

    common = context_dir
    for d in extra_contexts:
        common = pathlib.Path(os.path.commonpath([common, d]))

    tmp_dir = tempfile.mkdtemp(prefix="langgraph-deploy-")
    try:
        archive_path = os.path.join(tmp_dir, "source.tar.gz")

        added_dirs: set[str] = set()
        with tarfile.open(archive_path, "w:gz") as tar:
            for dir_path in dirs_to_include:
                rel = dir_path.relative_to(common)
                prefix = str(rel).replace(os.sep, "/") if str(rel) != "." else None
                key = prefix or ""
                if key in added_dirs:
                    continue
                added_dirs.add(key)
                ignore_spec = _build_ignore_spec(dir_path)
                _add_directory(
                    tar, dir_path, arcname_prefix=prefix, ignore_spec=ignore_spec
                )

        file_size = os.path.getsize(archive_path)

        config_rel = str(config_path.relative_to(common)).replace(os.sep, "/")

        with tarfile.open(archive_path, "r:gz") as tar:
            names = tar.getnames()
            if config_rel not in names:
                raise click.ClickException(
                    f"Archive validation failed: {config_rel} not found in archive"
                )

        if file_size > _MAX_SIZE:
            raise click.ClickException(
                f"Source archive is {file_size / 1_048_576:.1f} MB, which exceeds the 200 MB limit. "
                "Add large files to .dockerignore or .gitignore (model weights, data sets, etc.)."
            )

        if file_size > _WARN_SIZE:
            click.secho(
                f"   Warning: source archive is {file_size / 1_048_576:.1f} MB. "
                "Consider adding large files to .dockerignore or .gitignore.",
                fg="yellow",
            )

        yield archive_path, file_size, config_rel
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
