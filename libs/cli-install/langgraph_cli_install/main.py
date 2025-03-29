"""LangGraph CLI Installation Script.

Main entry point for installing langgraph-cli in an isolated environment.
This script uses uv to create an isolated installation of langgraph-cli.
"""

import platform
import subprocess
import sys

import uv


def main():
    """Install langgraph-cli using uv in an isolated environment."""
    print("Installing LangGraph CLI...")

    try:
        uv_bin = uv.find_uv_bin()

        # Get best Python version for installation (prefer 3.12 if available)
        python_version = get_latest_python_version()

        # Create an isolated environment with langgraph-cli
        print(f"Creating isolated environment using {python_version}...")
        subprocess.check_call(
            [
                uv_bin,
                "tool",
                "install",
                "--force",
                "--python",
                python_version,
                "langgraph-cli@latest",
            ]
        )

        # Update PATH so the tool is available
        subprocess.check_call([uv_bin, "tool", "update-shell"])

        # Show install location and help
        show_success_message(uv_bin)

    except subprocess.CalledProcessError as e:
        print(f"\nFailed to install langgraph-cli: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


def get_latest_python_version() -> str:
    """Get the latest compatible Python version for installation."""
    # Try to use Python 3.13 if possible, otherwise fall back to the current version
    target_version = "3.13"
    try:
        # Check if this version is available through uv
        uv_bin = uv.find_uv_bin()
        result = subprocess.run(
            [uv_bin, "python", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        if target_version in result.stdout:
            return f"python{target_version}"
    except Exception:
        pass

    # Fall back to current version
    major, minor = sys.version_info.major, sys.version_info.minor
    return f"python{major}.{minor}"


def show_success_message(uv_bin):
    """Show success message and installation details."""
    # Get installation path
    result = subprocess.run(
        [uv_bin, "tool", "list"],
        capture_output=True,
        text=True,
        check=True,
    )

    install_path = None
    for line in result.stdout.splitlines():
        if "langgraph-cli" in line:
            parts = line.strip().split()
            if len(parts) >= 2:
                install_path = parts[1]
                break

    # Success message
    print("\n🎉 LangGraph CLI has been successfully installed!\n")
    print("You can now use it by running:")
    print("  langgraph --help")

    if install_path:
        print(f"\nInstalled at: {install_path}")

    # Provide hint about shell restart if needed
    if platform.system() != "Windows":
        print("\nNote: You may need to restart your terminal or run:")
        print("  source ~/.bashrc  # or ~/.zshrc depending on your shell")
        print("to ensure the langgraph command is available in your PATH.")


if __name__ == "__main__":
    main()
