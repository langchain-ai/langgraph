#!/usr/bin/env python3
"""Test script to verify the _parse_version fix works correctly."""

from langgraph_cli.docker import _parse_version

def test_version_parsing():
    """Test various version formats to ensure the fix works."""
    test_cases = [
        ('28.1.1', (28, 1, 1)),
        ('28.1.1+1', (28, 1, 1)),  # This was the failing case
        ('1.2.3-alpha', (1, 2, 3)),
        ('1.2.3+build', (1, 2, 3)),
        ('1.2.3-alpha+build', (1, 2, 3)),
        ('v1.2.3', (1, 2, 3)),
        ('1.2', (1, 2, 0)),
        ('1', (1, 0, 0)),
    ]
    
    print("Testing _parse_version function:")
    for version_str, expected in test_cases:
        try:
            result = _parse_version(version_str)
            actual = (result.major, result.minor, result.patch)
            status = "✓" if actual == expected else "✗"
            print(f"{status} {version_str:15} -> {actual} (expected {expected})")
        except Exception as e:
            print(f"✗ {version_str:15} -> ERROR: {e}")

if __name__ == "__main__":
    test_version_parsing()
