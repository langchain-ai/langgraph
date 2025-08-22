#!/usr/bin/env python
"""Verify that the reserved keyword tests work correctly."""

import sys
sys.path.insert(0, 'libs/prebuilt')

from tests.test_react_agent import test_tool_node_inject_state_reserved_keyword

print("Running test_tool_node_inject_state_reserved_keyword...")
try:
    test_tool_node_inject_state_reserved_keyword()
    print("✓ State reserved keyword test passed!")
except Exception as e:
    print(f"✗ State reserved keyword test failed: {e}")
    sys.exit(1)

print("\nAll existing tests passed!")
