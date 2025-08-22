#!/usr/bin/env python
"""Test that deprecation warnings are properly emitted for InjectedState and InjectedStore."""

import warnings
from typing import Annotated
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode, InjectedState, InjectedStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore


def test_deprecation_warnings():
    """Test that deprecation warnings are emitted for annotation-based injection."""
    
    # Define tools using deprecated annotations
    def tool_with_injected_state(x: int, state: Annotated[dict, InjectedState]) -> str:
        """Tool using deprecated InjectedState annotation."""
        return f"state: {state.get('foo', 'none')}"
    
    def tool_with_injected_store(x: int, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """Tool using deprecated InjectedStore annotation."""
        return "has store"
    
    # Define tools using new reserved keywords (should not trigger warnings)
    def tool_with_reserved_state(x: int, state) -> str:
        """Tool using reserved keyword 'state'."""
        return f"state: {state.get('foo', 'none')}"
    
    def tool_with_reserved_runtime(x: int, runtime) -> str:
        """Tool using reserved keyword 'runtime'."""
        return "has runtime"
    
    print("Testing deprecation warnings...")
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Create ToolNode with deprecated annotation tools
        print("\n1. Creating ToolNode with deprecated annotation tools...")
        node1 = ToolNode([tool_with_injected_state, tool_with_injected_store])
        
        # Check that warnings were emitted
        assert len(w) == 2, f"Expected 2 warnings, got {len(w)}"
        
        # Check warning messages
        warning_messages = [str(warning.message) for warning in w]
        assert any("InjectedState" in msg for msg in warning_messages), "Missing InjectedState warning"
        assert any("InjectedStore" in msg for msg in warning_messages), "Missing InjectedStore warning"
        
        print(f"   ✓ Emitted {len(w)} deprecation warnings for annotation-based tools")
        for warning in w:
            print(f"     - {warning.message}")
    
    # Test that reserved keywords don't trigger warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        print("\n2. Creating ToolNode with reserved keyword tools...")
        node2 = ToolNode([tool_with_reserved_state, tool_with_reserved_runtime])
        
        # Check that no warnings were emitted
        assert len(w) == 0, f"Expected 0 warnings for reserved keywords, got {len(w)}"
        print(f"   ✓ No warnings emitted for reserved keyword tools")
    
    # Test mixed usage
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        print("\n3. Creating ToolNode with mixed tools...")
        node3 = ToolNode([
            tool_with_injected_state,  # Should warn
            tool_with_reserved_state,  # Should not warn
            tool_with_injected_store,  # Should warn
            tool_with_reserved_runtime  # Should not warn
        ])
        
        # Check that only 2 warnings were emitted (for the annotation-based tools)
        assert len(w) == 2, f"Expected 2 warnings for mixed tools, got {len(w)}"
        print(f"   ✓ Emitted {len(w)} warnings for annotation-based tools only")
    
    print("\n✅ All deprecation warning tests passed!")


if __name__ == "__main__":
    test_deprecation_warnings()
