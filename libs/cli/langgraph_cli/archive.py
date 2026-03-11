"""Create a tarball of project source for remote builds."""

import os
import pathlib
import tarfile
import tempfile

import click


_WARN_SIZE = 50 * 1024 * 1024  # 50 MB
_MAX_SIZE = 200 * 1024 * 1024  # 200 MB


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Strip symlinks, hardlinks, and traversal paths from archive."""
    if tarinfo.issym() or tarinfo.islnk():
        return None
    if ".." in tarinfo.name.split("/"):
        return None
    return tarinfo


def _read_ignore_patterns(context_dir: pathlib.Path) -> list[str]:
    """Read .dockerignore patterns if present."""
    dockerignore = context_dir / ".dockerignore"
    if dockerignore.is_file():
        patterns = []
        for line in dockerignore.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
        return patterns
    return []


def _should_ignore(rel_path: str, patterns: list[str]) -> bool:
    """Check if a relative path matches any dockerignore pattern."""
    import fnmatch

    rel_path = rel_path.replace(os.sep, "/")

    for pattern in patterns:
        negate = pattern.startswith("!")
        if negate:
            pattern = pattern[1:]

        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(
            rel_path, f"**/{pattern}"
        ):
            if negate:
                return False
            return True

        parts = rel_path.split("/")
        for i in range(len(parts)):
            partial = "/".join(parts[: i + 1])
            if fnmatch.fnmatch(partial, pattern):
                if negate:
                    return False
                return True

    return False


def create_archive(
    config_path: pathlib.Path,
) -> tuple[str, int]:
    """Create a .tar.gz archive of the project source.

    Returns (archive_path, file_size).
    The archive root is config.parent (the directory containing langgraph.json).
    """
    context_dir = config_path.parent.resolve()
    config_filename = config_path.name
    ignore_patterns = _read_ignore_patterns(context_dir)

    tmp_dir = tempfile.mkdtemp(prefix="langgraph-deploy-")
    archive_path = os.path.join(tmp_dir, "source.tar.gz")

    with tarfile.open(archive_path, "w:gz") as tar:
        for root, dirs, files in os.walk(context_dir):
            rel_root = os.path.relpath(root, context_dir)
            if rel_root == ".":
                rel_root = ""

            dirs[:] = [
                d
                for d in dirs
                if not _should_ignore(
                    os.path.join(rel_root, d) if rel_root else d, ignore_patterns
                )
            ]

            for f in files:
                rel_path = os.path.join(rel_root, f) if rel_root else f
                if _should_ignore(rel_path, ignore_patterns):
                    continue
                full_path = os.path.join(root, f)
                arcname = rel_path.replace(os.sep, "/")
                info = tar.gettarinfo(full_path, arcname=arcname)
                filtered = _tar_filter(info)
                if filtered is None:
                    continue
                with open(full_path, "rb") as fobj:
                    tar.addfile(filtered, fobj)

    file_size = os.path.getsize(archive_path)

    # Validate config file is at archive root
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
        if config_filename not in names:
            os.unlink(archive_path)
            raise click.ClickException(
                f"Archive validation failed: {config_filename} not found at archive root"
            )

    if file_size > _MAX_SIZE:
        os.unlink(archive_path)
        raise click.ClickException(
            f"Source archive is {file_size / 1_048_576:.1f} MB, which exceeds the 200 MB limit. "
            "Check your .dockerignore for large files (model weights, data, node_modules, .venv)."
        )

    if file_size > _WARN_SIZE:
        click.secho(
            f"   Warning: source archive is {file_size / 1_048_576:.1f} MB. "
            "Consider adding large files to .dockerignore.",
            fg="yellow",
        )

    return archive_path, file_size
