import os
import tarfile
from unittest.mock import patch

import click
import pytest

from langgraph_cli.archive import (
    _add_directory,
    _build_ignore_spec,
    _tar_filter,
    create_archive,
)

# ---------------------------------------------------------------------------
# _tar_filter
# ---------------------------------------------------------------------------


class TestTarFilter:
    def _make_info(self, name: str, *, type_: int = tarfile.REGTYPE) -> tarfile.TarInfo:
        info = tarfile.TarInfo(name=name)
        info.type = type_
        return info

    def test_regular_file_passes(self):
        info = self._make_info("src/main.py")
        assert _tar_filter(info) is info

    def test_symlink_rejected(self):
        info = self._make_info("link", type_=tarfile.SYMTYPE)
        assert _tar_filter(info) is None

    def test_hardlink_rejected(self):
        info = self._make_info("link", type_=tarfile.LNKTYPE)
        assert _tar_filter(info) is None

    def test_path_traversal_rejected(self):
        info = self._make_info("../../etc/passwd")
        assert _tar_filter(info) is None

    def test_path_traversal_in_middle_rejected(self):
        info = self._make_info("src/../../../etc/passwd")
        assert _tar_filter(info) is None

    def test_dotdot_as_name_component_rejected(self):
        info = self._make_info("foo/../bar")
        assert _tar_filter(info) is None

    def test_dotdot_in_filename_allowed(self):
        """A file literally named 'foo..bar' is not traversal."""
        info = self._make_info("foo..bar")
        assert _tar_filter(info) is info

    def test_directory_passes(self):
        info = self._make_info("src/", type_=tarfile.DIRTYPE)
        assert _tar_filter(info) is info


# ---------------------------------------------------------------------------
# _build_ignore_spec
# ---------------------------------------------------------------------------


class TestBuildIgnoreSpec:
    def test_always_excludes_builtins(self, tmp_path):
        spec = _build_ignore_spec(tmp_path)
        assert spec.match_file("__pycache__/")
        assert spec.match_file(".git/")
        assert spec.match_file(".venv/")
        assert spec.match_file("venv/")
        assert spec.match_file("node_modules/")
        assert spec.match_file(".tox/")
        assert spec.match_file(".mypy_cache/")

    def test_regular_file_not_excluded(self, tmp_path):
        spec = _build_ignore_spec(tmp_path)
        assert not spec.match_file("main.py")
        assert not spec.match_file("src/app.py")

    def test_merges_dockerignore(self, tmp_path):
        (tmp_path / ".dockerignore").write_text("*.log\nbuild/\n")
        spec = _build_ignore_spec(tmp_path)
        assert spec.match_file("server.log")
        assert spec.match_file("build/")
        # builtins still present
        assert spec.match_file("__pycache__/")

    def test_merges_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.pyc\ndist/\n")
        spec = _build_ignore_spec(tmp_path)
        assert spec.match_file("module.pyc")
        assert spec.match_file("dist/")

    def test_merges_both_ignore_files(self, tmp_path):
        (tmp_path / ".dockerignore").write_text("*.log\n")
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        spec = _build_ignore_spec(tmp_path)
        assert spec.match_file("app.log")
        assert spec.match_file("mod.pyc")

    def test_no_ignore_files_only_builtins(self, tmp_path):
        spec = _build_ignore_spec(tmp_path)
        assert spec.match_file("__pycache__/")
        assert not spec.match_file("README.md")


# ---------------------------------------------------------------------------
# _add_directory
# ---------------------------------------------------------------------------


class TestAddDirectory:
    def _create_project(self, tmp_path):
        """Create a small project structure for testing."""
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "util.py").write_text("x = 1")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x00")
        return tmp_path

    def test_adds_files_without_prefix(self, tmp_path):
        project = self._create_project(tmp_path)
        spec = _build_ignore_spec(project)

        archive_path = tmp_path / "out.tar"
        with tarfile.open(archive_path, "w") as tar:
            _add_directory(tar, project, arcname_prefix=None, ignore_spec=spec)

        with tarfile.open(archive_path, "r") as tar:
            names = tar.getnames()
        assert "main.py" in names
        assert "lib/util.py" in names

    def test_excludes_pycache(self, tmp_path):
        project = self._create_project(tmp_path)
        spec = _build_ignore_spec(project)

        archive_path = tmp_path / "out.tar"
        with tarfile.open(archive_path, "w") as tar:
            _add_directory(tar, project, arcname_prefix=None, ignore_spec=spec)

        with tarfile.open(archive_path, "r") as tar:
            names = tar.getnames()
        assert not any("__pycache__" in n for n in names)

    def test_adds_files_with_prefix(self, tmp_path):
        project = self._create_project(tmp_path)
        spec = _build_ignore_spec(project)

        archive_path = tmp_path / "out.tar"
        with tarfile.open(archive_path, "w") as tar:
            _add_directory(tar, project, arcname_prefix="myapp", ignore_spec=spec)

        with tarfile.open(archive_path, "r") as tar:
            names = tar.getnames()
        assert "myapp/main.py" in names
        assert "myapp/lib/util.py" in names

    def test_respects_custom_ignore_patterns(self, tmp_path):
        project = self._create_project(tmp_path)
        (project / ".gitignore").write_text("lib/\n")
        spec = _build_ignore_spec(project)

        archive_path = tmp_path / "out.tar"
        with tarfile.open(archive_path, "w") as tar:
            _add_directory(tar, project, arcname_prefix=None, ignore_spec=spec)

        with tarfile.open(archive_path, "r") as tar:
            names = tar.getnames()
        assert "main.py" in names
        assert "lib/util.py" not in names


# ---------------------------------------------------------------------------
# create_archive (integration)
# ---------------------------------------------------------------------------


class TestCreateArchive:
    def _make_project(self, tmp_path):
        """Set up a minimal project directory with a config file."""
        project = tmp_path / "myproject"
        project.mkdir()
        config_file = project / "langgraph.json"
        config_file.write_text('{"dependencies": ["."]}')
        (project / "app.py").write_text("print('hello')")
        (project / "__pycache__").mkdir()
        (project / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\x00")
        return config_file

    @patch("langgraph_cli.archive._assemble_local_deps")
    def test_yields_archive_with_config(self, mock_deps, tmp_path):
        from langgraph_cli.config import LocalDeps

        config_file = self._make_project(tmp_path)
        mock_deps.return_value = LocalDeps(
            pip_reqs=[], real_pkgs={}, faux_pkgs={}, additional_contexts=None
        )

        with create_archive(config_file, {}) as (archive_path, file_size, config_rel):
            assert os.path.isfile(archive_path)
            assert archive_path.endswith(".tar.gz")
            assert file_size > 0
            assert config_rel == "langgraph.json"

            with tarfile.open(archive_path, "r:gz") as tar:
                names = tar.getnames()
            assert "langgraph.json" in names
            assert "app.py" in names

    @patch("langgraph_cli.archive._assemble_local_deps")
    def test_excludes_pycache(self, mock_deps, tmp_path):
        from langgraph_cli.config import LocalDeps

        config_file = self._make_project(tmp_path)
        mock_deps.return_value = LocalDeps(
            pip_reqs=[], real_pkgs={}, faux_pkgs={}, additional_contexts=None
        )

        with create_archive(config_file, {}) as (archive_path, _size, _rel):
            with tarfile.open(archive_path, "r:gz") as tar:
                names = tar.getnames()
            assert not any("__pycache__" in n for n in names)

    @patch("langgraph_cli.archive._assemble_local_deps")
    def test_cleans_up_tmp_dir_on_normal_exit(self, mock_deps, tmp_path):
        from langgraph_cli.config import LocalDeps

        config_file = self._make_project(tmp_path)
        mock_deps.return_value = LocalDeps(
            pip_reqs=[], real_pkgs={}, faux_pkgs={}, additional_contexts=None
        )

        with create_archive(config_file, {}) as (archive_path, _size, _rel):
            tmp_dir = os.path.dirname(archive_path)
            assert os.path.isdir(tmp_dir)

        assert not os.path.exists(tmp_dir)

    @patch("langgraph_cli.archive._assemble_local_deps")
    def test_cleans_up_tmp_dir_on_exception(self, mock_deps, tmp_path):
        from langgraph_cli.config import LocalDeps

        config_file = self._make_project(tmp_path)
        mock_deps.return_value = LocalDeps(
            pip_reqs=[], real_pkgs={}, faux_pkgs={}, additional_contexts=None
        )

        with pytest.raises(RuntimeError, match="boom"):
            with create_archive(config_file, {}) as (archive_path, _size, _rel):
                tmp_dir = os.path.dirname(archive_path)
                raise RuntimeError("boom")

        assert not os.path.exists(tmp_dir)

    @patch("langgraph_cli.archive._assemble_local_deps")
    @patch("langgraph_cli.archive._MAX_SIZE", 10)
    def test_raises_on_oversized_archive(self, mock_deps, tmp_path):
        from langgraph_cli.config import LocalDeps

        config_file = self._make_project(tmp_path)
        mock_deps.return_value = LocalDeps(
            pip_reqs=[], real_pkgs={}, faux_pkgs={}, additional_contexts=None
        )

        with pytest.raises(click.ClickException, match="exceeds the 200 MB limit"):
            with create_archive(config_file, {}):
                pass

    @patch("langgraph_cli.archive._assemble_local_deps")
    def test_handles_extra_contexts(self, mock_deps, tmp_path):
        """Monorepo case: project + sibling dependency directory."""
        from langgraph_cli.config import LocalDeps

        project = tmp_path / "myproject"
        project.mkdir()
        config_file = project / "langgraph.json"
        config_file.write_text('{"dependencies": [".", "../shared"]}')
        (project / "app.py").write_text("print('hello')")

        shared = tmp_path / "shared"
        shared.mkdir()
        (shared / "lib.py").write_text("y = 2")

        mock_deps.return_value = LocalDeps(
            pip_reqs=[],
            real_pkgs={},
            faux_pkgs={},
            additional_contexts=[shared],
        )

        with create_archive(config_file, {}) as (archive_path, _size, config_rel):
            with tarfile.open(archive_path, "r:gz") as tar:
                names = tar.getnames()
            assert "myproject/app.py" in names
            assert "shared/lib.py" in names
            assert config_rel == "myproject/langgraph.json"
