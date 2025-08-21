#!/usr/bin/env python3
"""
Comparison script for importlib.metadata vs dynamic versioning.
"""

import time


def old_approach():
    """Simulate old importlib.metadata approach."""
    print("Old approach (importlib.metadata):")
    start = time.perf_counter()

    # Simulate the old code
    try:
        from importlib.metadata import version as _version

        version = _version("langgraph")
    except ImportError:
        version = "unknown"

    end = time.perf_counter()
    print(f"  Time: {end - start:.6f}s")
    print(f"  Version: {version}")
    print("  Dependencies: importlib.metadata")

    return end - start


def new_approach():
    """Show new dynamic versioning approach."""
    print("\nNew approach (dynamic versioning):")
    start = time.perf_counter()

    # Simulate the new code
    try:
        from .about import __version__

        version = __version__
    except ImportError:
        version = "0.0.0"

    end = time.perf_counter()
    print(f"  Time: {end - start:.6f}s")
    print(f"  Version: {version}")
    print("  Dependencies: None")

    return end - start


def main():
    print("LangGraph Import Method Comparison")
    print("=" * 40)

    old_time = old_approach()
    new_time = new_approach()

    if old_time and new_time:
        improvement = ((old_time - new_time) / old_time) * 100
        print(f"\nImprovement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
