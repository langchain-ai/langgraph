"""Dependency manager detection and virtual environment activation for LangGraph CLI.

This module provides functionality to detect modern Python dependency managers
(pipenv, uv, poetry) and activate their virtual environments when running
`langgraph dev`.
"""

import os
import pathlib
import subprocess
import sys
from enum import Enum
from typing import Optional


class DependencyManager(Enum):
    """Supported dependency managers."""

    PIPENV = "pipenv"
    UV = "uv"
    POETRY = "poetry"
    PIP = "pip"  # fallback


def detect_dependency_manager(
    project_dir: pathlib.Path,
) -> tuple[DependencyManager, Optional[pathlib.Path]]:
    """Detect the dependency manager used in the project.

    Args:
        project_dir: The project directory to check for dependency manager files.

    Returns:
        A tuple of (dependency_manager, config_file_path).
        config_file_path is None for pip (no specific config file).
    """
    # Check for uv first (LangGraph's chosen tool, fastest, most modern)
    uv_lock_path = project_dir / "uv.lock"
    if uv_lock_path.exists():
        return DependencyManager.UV, uv_lock_path

    # Check for requirements.lock (uv alternative)
    requirements_lock_path = project_dir / "requirements.lock"
    if requirements_lock_path.exists():
        return DependencyManager.UV, requirements_lock_path

    # Check for uv project without lock file
    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path, encoding="utf-8") as f:
                content = f.read()
                if "[tool.uv]" in content:
                    return DependencyManager.UV, pyproject_path
        except (OSError, UnicodeDecodeError):
            pass

    # Check for Poetry (mature, widely adopted)
    if pyproject_path.exists():
        try:
            with open(pyproject_path, encoding="utf-8") as f:
                content = f.read()
                if "[tool.poetry]" in content:
                    return DependencyManager.POETRY, pyproject_path
        except (OSError, UnicodeDecodeError):
            pass

    # Check for Pipenv (legacy support)
    pipfile_path = project_dir / "Pipfile"
    if pipfile_path.exists():
        return DependencyManager.PIPENV, pipfile_path

    # Fallback to pip
    return DependencyManager.PIP, None


def get_virtual_env_path(
    dependency_manager: DependencyManager, project_dir: pathlib.Path
) -> Optional[pathlib.Path]:
    """Get the virtual environment path for the given dependency manager.

    Args:
        dependency_manager: The detected dependency manager.
        project_dir: The project directory.

    Returns:
        Path to the virtual environment, or None if not found.
    """
    if dependency_manager == DependencyManager.PIPENV:
        try:
            # Get pipenv virtual environment path
            result = subprocess.run(
                ["pipenv", "--venv"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            venv_path = pathlib.Path(result.stdout.strip())
            return venv_path if venv_path.exists() else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    elif dependency_manager == DependencyManager.POETRY:
        try:
            # Get poetry virtual environment path
            result = subprocess.run(
                ["poetry", "env", "info", "--path"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            venv_path = pathlib.Path(result.stdout.strip())
            return venv_path if venv_path.exists() else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    elif dependency_manager == DependencyManager.UV:
        try:
            # Get uv virtual environment path
            result = subprocess.run(
                ["uv", "venv", "--python", "python"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=False,  # Don't fail if venv doesn't exist
            )
            # uv venv creates .venv by default
            venv_path = project_dir / ".venv"
            return venv_path if venv_path.exists() else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return None


def activate_virtual_environment(venv_path: pathlib.Path) -> bool:
    """Activate a virtual environment by modifying sys.path and environment variables.

    Args:
        venv_path: Path to the virtual environment.

    Returns:
        True if activation was successful, False otherwise.
    """
    if not venv_path.exists():
        return False

    # Find the correct Python version directory by scanning the lib directory
    lib_dir = venv_path / "lib"
    if not lib_dir.exists():
        return False

    # Look for pythonX.Y directories
    python_dirs = [
        d for d in lib_dir.iterdir() if d.is_dir() and d.name.startswith("python")
    ]

    if not python_dirs:
        return False

    # Use the first Python directory found (most virtual environments have only one)
    python_dir = python_dirs[0]
    site_packages = python_dir / "site-packages"

    if site_packages.exists():
        # Insert at the beginning to prioritize virtual environment packages
        sys.path.insert(0, str(site_packages))

        # Set VIRTUAL_ENV environment variable
        os.environ["VIRTUAL_ENV"] = str(venv_path)

        # Update PATH to prioritize virtual environment's bin directory
        if sys.platform == "win32":
            bin_path = venv_path / "Scripts"
        else:
            bin_path = venv_path / "bin"

        if bin_path.exists():
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_path}{os.pathsep}{current_path}"

        return True

    return False


def install_dependencies(
    dependency_manager: DependencyManager, project_dir: pathlib.Path
) -> bool:
    """Install dependencies using the detected dependency manager.

    Args:
        dependency_manager: The detected dependency manager.
        project_dir: The project directory.

    Returns:
        True if installation was successful, False otherwise.
    """
    try:
        if dependency_manager == DependencyManager.PIPENV:
            subprocess.run(["pipenv", "install"], cwd=project_dir, check=True)
            return True

        elif dependency_manager == DependencyManager.POETRY:
            subprocess.run(["poetry", "install"], cwd=project_dir, check=True)
            return True

        elif dependency_manager == DependencyManager.UV:
            # Try to install with uv
            subprocess.run(
                ["uv", "pip", "install", "-e", "."], cwd=project_dir, check=True
            )
            return True

        # For pip, we don't automatically install dependencies
        # as the user should handle this manually
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def setup_development_environment(
    project_dir: pathlib.Path,
) -> tuple[DependencyManager, bool]:
    """Set up the development environment by detecting and activating the appropriate dependency manager.

    Args:
        project_dir: The project directory to set up.

    Returns:
        A tuple of (detected_dependency_manager, success).
    """
    # Store the original working directory
    original_cwd = os.getcwd()

    try:
        # Change to project directory for dependency manager commands
        os.chdir(project_dir)

        # Detect dependency manager
        dependency_manager, config_file = detect_dependency_manager(project_dir)

        if dependency_manager == DependencyManager.PIP:
            # No special setup needed for pip
            return dependency_manager, True

        # Try to get virtual environment path
        venv_path = get_virtual_env_path(dependency_manager, project_dir)

        if venv_path is None:
            # Try to create virtual environment if it doesn't exist
            if dependency_manager == DependencyManager.UV:
                try:
                    subprocess.run(["uv", "venv"], cwd=project_dir, check=True)
                    venv_path = project_dir / ".venv"
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

        # Activate virtual environment if found
        if venv_path and activate_virtual_environment(venv_path):
            return dependency_manager, True

        # If virtual environment activation failed, fall back to pip
        return DependencyManager.PIP, False

    finally:
        # Always restore the original working directory
        os.chdir(original_cwd)
