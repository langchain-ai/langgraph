#!/usr/bin/env python
"""Debug reserved keyword detection for annotated tools."""

from typing import Annotated
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.prebuilt.tool_node import _get_reserved_keyword_args
from langchain_core.tools import StructuredTool
from langgraph.store.base import BaseStore


def tool_with_injected_state(x: int, state: Annotated[dict, InjectedState]) -> str:
    """Tool using deprecated InjectedState annotation."""
    return f"state: {state.get('foo', 'none')}"

def tool_with_injected_store(x: int, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Tool using deprecated InjectedStore annotation."""
    return "has store"

def tool_with_reserved_state(x: int, state) -> str:
    """Tool using reserved keyword 'state'."""
    return f"state: {state.get('foo', 'none')}"

# Convert to tools
tool1 = StructuredTool.from_function(tool_with_injected_state)
tool2 = StructuredTool.from_function(tool_with_injected_store)
tool3 = StructuredTool.from_function(tool_with_reserved_state)

print("Debugging reserved keyword detection:")

print(f"\nTool 1 (Annotated[dict, InjectedState]):")
reserved1 = _get_reserved_keyword_args(tool1)
print(f"  Reserved args: {reserved1}")
print(f"  Has 'state' as reserved? {'state' in reserved1}")

print(f"\nTool 2 (Annotated[BaseStore, InjectedStore()]):")
reserved2 = _get_reserved_keyword_args(tool2)
print(f"  Reserved args: {reserved2}")
print(f"  Has 'runtime' as reserved? {'runtime' in reserved2}")

print(f"\nTool 3 (plain 'state' parameter):")
reserved3 = _get_reserved_keyword_args(tool3)
print(f"  Reserved args: {reserved3}")
print(f"  Has 'state' as reserved? {'state' in reserved3}")

# The issue might be that 'state' is being detected as a reserved keyword
# even when it has an annotation
print("\nConclusion:")
print("If tool1 shows 'state' as reserved, that's the bug - it shouldn't be")
print("considered reserved when it has an InjectedState annotation.")
