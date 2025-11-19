#!/usr/bin/env python3
"""Test to verify ToolCallRequest raises deprecation warning on attribute modification."""

import sys
import warnings
from dataclasses import dataclass, replace
from typing import Any

# Minimal inline definition for testing
@dataclass
class ToolCallRequest:
    """Tool execution request passed to tool call interceptors."""
    tool_call: dict
    tool: Any
    state: Any
    runtime: Any

    def __setattr__(self, name: str, value: Any) -> None:
        """Raise deprecation warning when setting attributes directly.

        Direct attribute assignment is deprecated. Use the `override()` method instead.
        """
        # Allow setting attributes during initialization
        if not hasattr(self, '__dict__') or name not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            warnings.warn(
                f"Setting attribute '{name}' on ToolCallRequest is deprecated. "
                "Use the override() method instead to create a new instance with modified values.",
                DeprecationWarning,
                stacklevel=2
            )
            object.__setattr__(self, name, value)

    def override(self, **overrides: Any) -> "ToolCallRequest":
        """Replace the request with a new request with the given overrides."""
        return replace(self, **overrides)


# Create a mock ToolCall
tool_call = {"name": "test", "args": {"a": 1}, "id": "call_1", "type": "tool_call"}

# Create a ToolCallRequest
request = ToolCallRequest(
    tool_call=tool_call,
    tool=None,
    state={"messages": []},
    runtime=None
)

print("✓ Created ToolCallRequest successfully")

# Test 1: Try direct attribute assignment (should work but raise deprecation warning)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    request.tool_call = {"name": "other", "args": {}, "id": "call_2"}

    # Verify warning was raised
    assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
    assert issubclass(w[0].category, DeprecationWarning)
    assert "deprecated" in str(w[0].message).lower()
    assert "override()" in str(w[0].message)
    print("✓ Direct attribute assignment raises DeprecationWarning")
    print(f"  Warning message: {w[0].message}")

# Verify the attribute was actually modified
assert request.tool_call == {"name": "other", "args": {}, "id": "call_2"}
print("✓ Attribute was successfully modified (despite deprecation)")

# Reset for further tests
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    request.tool_call = tool_call

# Test 2: Test override method works (no warning)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    new_tool_call = {"name": "new_tool", "args": {"b": 2}, "id": "call_3", "type": "tool_call"}
    new_request = request.override(tool_call=new_tool_call)

    # Verify no warning was raised
    assert len(w) == 0, f"Expected no warnings, got {len(w)}"
    print("✓ Override method works without warnings")

# Test 3: Verify original is unchanged
assert request.tool_call == tool_call
assert request.tool_call["name"] == "test"
print("✓ Original request unchanged after override")

# Test 4: Verify new request has updated values
assert new_request.tool_call == new_tool_call
assert new_request.tool_call["name"] == "new_tool"
print("✓ New request has updated values")

# Test 5: Verify other fields remain the same
assert new_request.tool is None
assert new_request.state == {"messages": []}
assert new_request.runtime is None
print("✓ Other fields preserved in override")

# Test 6: Verify initialization doesn't trigger warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    request2 = ToolCallRequest(
        tool_call=tool_call,
        tool=None,
        state={"messages": []},
        runtime=None
    )
    # Verify no warning was raised during initialization
    assert len(w) == 0, f"Expected no warnings during initialization, got {len(w)}"
    print("✓ No warning during initialization")

print("\n✅ All tests passed! ToolCallRequest raises deprecation warnings on direct attribute modification.")
