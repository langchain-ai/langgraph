#!/usr/bin/env python3
"""
Quick setup script for LangGraph Kusto Checkpointer tutorials.

This script checks your environment and helps you get set up.
"""

import subprocess
import sys
import os


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def check_python_version():
    """Check if Python version is 3.10+."""
    print_section("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required")
        print("   Please upgrade Python")
        return False
    
    print("✓ Python version is compatible")
    return True


def check_pip():
    """Check if pip is available."""
    print_section("Checking pip")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✓ pip is available: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        print("   Install pip: python -m ensurepip --upgrade")
        return False


def install_dependencies():
    """Install package dependencies."""
    print_section("Installing Dependencies")
    
    print("Installing langgraph-checkpoint-kusto...")
    print("This may take a few minutes...\n")
    
    try:
        # Try to install in editable mode from current directory
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Installation failed")
        print(f"   Error: {e.stderr}")
        print("\nTry manual installation:")
        print("   pip install langgraph")
        print("   pip install azure-kusto-data azure-kusto-ingest azure-identity")
        return False


def verify_installation():
    """Verify that the package can be imported."""
    print_section("Verifying Installation")
    
    try:
        from langgraph.checkpoint.kusto.aio import AsyncKustoSaver
        print("✓ langgraph.checkpoint.kusto can be imported")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\nTroubleshooting:")
        print("   1. Make sure you're in the libs/checkpoint-kusto directory")
        print("   2. Try: pip install -e .")
        print("   3. Check for errors in the pip install output")
        return False


def check_environment_variables():
    """Check if environment variables are set."""
    print_section("Checking Environment Variables")
    
    cluster_uri = os.getenv("KUSTO_CLUSTER_URI")
    database = os.getenv("KUSTO_DATABASE")
    
    all_set = True
    
    if cluster_uri:
        print(f"✓ KUSTO_CLUSTER_URI is set: {cluster_uri}")
    else:
        print("⚠  KUSTO_CLUSTER_URI is not set")
        print("   Set it with:")
        print('   PowerShell: $env:KUSTO_CLUSTER_URI = "https://your-cluster.region.kusto.windows.net"')
        print('   Bash: export KUSTO_CLUSTER_URI="https://your-cluster.region.kusto.windows.net"')
        all_set = False
    
    if database:
        print(f"✓ KUSTO_DATABASE is set: {database}")
    else:
        print("⚠  KUSTO_DATABASE is not set (will use 'langgraph' as default)")
        print("   Set it with:")
        print('   PowerShell: $env:KUSTO_DATABASE = "langgraph"')
        print('   Bash: export KUSTO_DATABASE="langgraph"')
    
    return all_set


def check_azure_cli():
    """Check if Azure CLI is installed and logged in."""
    print_section("Checking Azure CLI")
    
    try:
        result = subprocess.run(
            ["az", "account", "show"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Azure CLI is installed and logged in")
        import json
        account = json.loads(result.stdout)
        print(f"   Account: {account.get('name', 'Unknown')}")
        return True
    except FileNotFoundError:
        print("⚠  Azure CLI is not installed")
        print("   Install from: https://aka.ms/installazurecliwindows")
        return False
    except subprocess.CalledProcessError:
        print("⚠  Azure CLI is installed but not logged in")
        print("   Run: az login")
        return False


def main():
    """Run all setup checks."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║   LangGraph Kusto Checkpointer - Setup Assistant         ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    checks = []
    
    # Check Python version
    checks.append(("Python 3.10+", check_python_version()))
    
    # Check pip
    checks.append(("pip installed", check_pip()))
    
    # Install dependencies
    if checks[-1][1]:  # Only if pip is available
        checks.append(("Dependencies installed", install_dependencies()))
    
    # Verify installation
    if checks[-1][1]:  # Only if installation succeeded
        checks.append(("Import successful", verify_installation()))
    
    # Check environment variables
    checks.append(("Environment variables", check_environment_variables()))
    
    # Check Azure CLI
    checks.append(("Azure CLI", check_azure_cli()))
    
    # Summary
    print_section("Setup Summary")
    
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"{status} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print("\n🎉 Setup complete! You're ready to run the tutorials.")
        print("\nNext steps:")
        print("   1. Make sure you've run provision.kql in your Kusto cluster")
        print("   2. cd examples")
        print("   3. python tutorial_01_first_checkpoint.py")
        print("\nFor detailed setup, see: SETUP.md")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("   See SETUP.md for detailed instructions")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
