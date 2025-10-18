"""Unit tests for dependency manager functionality."""

import os
import pathlib
import subprocess
import sys
import tempfile
from unittest.mock import MagicMock, patch

from langgraph_cli.dependency_manager import (
    DependencyManager,
    activate_virtual_environment,
    detect_dependency_manager,
    get_virtual_env_path,
    install_dependencies,
    setup_development_environment,
)


class TestDetectDependencyManager:
    """Test dependency manager detection functionality."""

    def test_detect_pipenv(self):
        """Test detection of Pipenv projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            pipfile_path = temp_path / "Pipfile"
            pipfile_path.write_text('[[source]]\nurl = "https://pypi.org/simple"')

            manager, config_file = detect_dependency_manager(temp_path)
            assert manager == DependencyManager.PIPENV
            assert config_file == pipfile_path

    def test_detect_poetry(self):
        """Test detection of Poetry projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            pyproject_path = temp_path / "pyproject.toml"
            pyproject_path.write_text('[tool.poetry]\nname = "test"')

            manager, config_file = detect_dependency_manager(temp_path)
            assert manager == DependencyManager.POETRY
            assert config_file == pyproject_path

    def test_detect_uv_with_lock(self):
        """Test detection of uv projects with lock file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            uv_lock_path = temp_path / "uv.lock"
            uv_lock_path.write_text("version = 1")

            manager, config_file = detect_dependency_manager(temp_path)
            assert manager == DependencyManager.UV
            assert config_file == uv_lock_path

    def test_detect_uv_with_requirements_lock(self):
        """Test detection of uv projects with requirements.lock."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            requirements_lock_path = temp_path / "requirements.lock"
            requirements_lock_path.write_text("requests==2.31.0")

            manager, config_file = detect_dependency_manager(temp_path)
            assert manager == DependencyManager.UV
            assert config_file == requirements_lock_path

    def test_detect_uv_with_pyproject(self):
        """Test detection of uv projects with pyproject.toml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            pyproject_path = temp_path / "pyproject.toml"
            pyproject_path.write_text("[tool.uv]\ndev-dependencies = []")

            manager, config_file = detect_dependency_manager(temp_path)
            assert manager == DependencyManager.UV
            assert config_file == pyproject_path

    def test_detect_pip_fallback(self):
        """Test fallback to pip when no modern dependency manager is detected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            manager, config_file = detect_dependency_manager(temp_path)
            assert manager == DependencyManager.PIP
            assert config_file is None

    def test_uv_priority_over_poetry(self):
        """Test that uv is detected over poetry when both pyproject.toml sections exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            pyproject_path = temp_path / "pyproject.toml"
            pyproject_path.write_text(
                '[tool.poetry]\nname = "test"\n\n[tool.uv]\ndev-dependencies = []'
            )

            manager, config_file = detect_dependency_manager(temp_path)
            assert manager == DependencyManager.UV
            assert config_file == pyproject_path


class TestGetVirtualEnvPath:
    """Test virtual environment path detection."""

    @patch("subprocess.run")
    def test_get_pipenv_venv_path(self, mock_run):
        """Test getting pipenv virtual environment path."""
        mock_result = MagicMock()
        mock_result.stdout = "/path/to/venv\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Mock pathlib.Path.exists to return True for the test path
        with patch("pathlib.Path.exists", return_value=True):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = pathlib.Path(temp_dir)
                venv_path = get_virtual_env_path(DependencyManager.PIPENV, temp_path)

                assert venv_path == pathlib.Path("/path/to/venv")
                mock_run.assert_called_once_with(
                    ["pipenv", "--venv"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )

    @patch("subprocess.run")
    def test_get_poetry_venv_path(self, mock_run):
        """Test getting poetry virtual environment path."""
        mock_result = MagicMock()
        mock_result.stdout = "/path/to/poetry-venv\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Mock pathlib.Path.exists to return True for the test path
        with patch("pathlib.Path.exists", return_value=True):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = pathlib.Path(temp_dir)
                venv_path = get_virtual_env_path(DependencyManager.POETRY, temp_path)

                assert venv_path == pathlib.Path("/path/to/poetry-venv")
                mock_run.assert_called_once_with(
                    ["poetry", "env", "info", "--path"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )

    @patch("subprocess.run")
    def test_get_uv_venv_path(self, mock_run):
        """Test getting uv virtual environment path."""
        mock_run.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            venv_path = temp_path / ".venv"
            venv_path.mkdir()

            result = get_virtual_env_path(DependencyManager.UV, temp_path)

            assert result == venv_path

    @patch("subprocess.run")
    def test_get_venv_path_failure(self, mock_run):
        """Test handling of subprocess failures."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "command")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            venv_path = get_virtual_env_path(DependencyManager.PIPENV, temp_path)

            assert venv_path is None


class TestActivateVirtualEnvironment:
    """Test virtual environment activation."""

    def test_activate_venv_success(self):
        """Test successful virtual environment activation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = pathlib.Path(temp_dir)

            # Create site-packages directory
            if sys.platform == "win32":
                site_packages = venv_path / "Lib" / "site-packages"
            else:
                site_packages = (
                    venv_path
                    / "lib"
                    / f"python{sys.version_info.major}.{sys.version_info.minor}"
                    / "site-packages"
                )
            site_packages.mkdir(parents=True)

            # Create bin directory
            if sys.platform == "win32":
                bin_path = venv_path / "Scripts"
            else:
                bin_path = venv_path / "bin"
            bin_path.mkdir()

            result = activate_virtual_environment(venv_path)

            assert result is True
            assert str(site_packages) in sys.path
            assert os.environ.get("VIRTUAL_ENV") == str(venv_path)

    def test_activate_venv_nonexistent(self):
        """Test activation of non-existent virtual environment."""
        venv_path = pathlib.Path("/nonexistent/path")

        result = activate_virtual_environment(venv_path)

        assert result is False

    def test_activate_venv_no_site_packages(self):
        """Test activation when site-packages doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = pathlib.Path(temp_dir)

            result = activate_virtual_environment(venv_path)

            assert result is False


class TestInstallDependencies:
    """Test dependency installation functionality."""

    @patch("subprocess.run")
    def test_install_pipenv_dependencies(self, mock_run):
        """Test installing dependencies with pipenv."""
        mock_run.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            result = install_dependencies(DependencyManager.PIPENV, temp_path)

            assert result is True
            mock_run.assert_called_once_with(
                ["pipenv", "install"],
                cwd=temp_path,
                check=True,
            )

    @patch("subprocess.run")
    def test_install_poetry_dependencies(self, mock_run):
        """Test installing dependencies with poetry."""
        mock_run.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            result = install_dependencies(DependencyManager.POETRY, temp_path)

            assert result is True
            mock_run.assert_called_once_with(
                ["poetry", "install"],
                cwd=temp_path,
                check=True,
            )

    @patch("subprocess.run")
    def test_install_uv_dependencies(self, mock_run):
        """Test installing dependencies with uv."""
        mock_run.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            result = install_dependencies(DependencyManager.UV, temp_path)

            assert result is True
            mock_run.assert_called_once_with(
                ["uv", "pip", "install", "-e", "."],
                cwd=temp_path,
                check=True,
            )

    @patch("subprocess.run")
    def test_install_dependencies_failure(self, mock_run):
        """Test handling of installation failures."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "command")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            result = install_dependencies(DependencyManager.PIPENV, temp_path)

            assert result is False


class TestSetupDevelopmentEnvironment:
    """Test development environment setup."""

    @patch("langgraph_cli.dependency_manager.get_virtual_env_path")
    @patch("langgraph_cli.dependency_manager.activate_virtual_environment")
    def test_setup_pipenv_success(self, mock_activate, mock_get_venv):
        """Test successful pipenv environment setup."""
        mock_get_venv.return_value = pathlib.Path("/test/venv")
        mock_activate.return_value = True

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            pipfile_path = temp_path / "Pipfile"
            pipfile_path.write_text('[[source]]\nurl = "https://pypi.org/simple"')

            manager, success = setup_development_environment(temp_path)

            assert manager == DependencyManager.PIPENV
            assert success is True
            mock_activate.assert_called_once_with(pathlib.Path("/test/venv"))

    @patch("langgraph_cli.dependency_manager.get_virtual_env_path")
    @patch("langgraph_cli.dependency_manager.activate_virtual_environment")
    def test_setup_pipenv_activation_failure(self, mock_activate, mock_get_venv):
        """Test pipenv setup when activation fails."""
        mock_get_venv.return_value = pathlib.Path("/test/venv")
        mock_activate.return_value = False

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            pipfile_path = temp_path / "Pipfile"
            pipfile_path.write_text('[[source]]\nurl = "https://pypi.org/simple"')

            manager, success = setup_development_environment(temp_path)

            assert manager == DependencyManager.PIP
            assert success is False

    def test_setup_pip_fallback(self):
        """Test setup with pip fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            manager, success = setup_development_environment(temp_path)

            assert manager == DependencyManager.PIP
            assert success is True

    @patch("subprocess.run")
    @patch("langgraph_cli.dependency_manager.get_virtual_env_path")
    @patch("langgraph_cli.dependency_manager.activate_virtual_environment")
    def test_setup_uv_create_venv(self, mock_activate, mock_get_venv, mock_run):
        """Test uv setup when virtual environment needs to be created."""
        mock_get_venv.return_value = None
        mock_run.return_value = MagicMock()
        mock_activate.return_value = True

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            uv_lock_path = temp_path / "uv.lock"
            uv_lock_path.write_text("version = 1")

            manager, success = setup_development_environment(temp_path)

            assert manager == DependencyManager.UV
            assert success is True
            mock_run.assert_called_once_with(["uv", "venv"], cwd=temp_path, check=True)
