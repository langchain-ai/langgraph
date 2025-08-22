#!/usr/bin/env python
"""Debug why InjectedState deprecation warning is not being emitted."""

from typing import Annotated
from langgraph.prebuilt import ToolNode, InjectedState, InjectedStore
from langgraph.prebuilt.tool_node import _get_state_args, _get_store_arg, _get_reserved_keyword_args
from langgraph.store.base import BaseStore
from langchain_core.tools import create_tool


def tool_with_injected_state(x: int, state: Annotated[dict, InjectedState]) -> str:
    """Tool using deprecated InjectedState annotation."""
    return f"state: {state.get('foo', 'none')}"

def tool_with_injected_store(x: int, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Tool using deprecated InjectedStore annotation."""
    return "has store"

# Convert to tools
tool1 = create_tool(tool_with_injected_state)
tool2 = create_tool(tool_with_injected_store)

print("Debugging annotation detection:")
print(f"\nTool 1 (InjectedState):")
print(f"  _get_state_args: {_get_state_args(tool1)}")
print(f"  _get_reserved_keyword_args: {_get_reserved_keyword_args(tool1)}")

print(f"\nTool 2 (InjectedStore):")
print(f"  _get_store_arg: {_get_store_arg(tool2)}")
print(f"  _get_reserved_keyword_args: {_get_reserved_keyword_args(tool2)}")

# Check the condition for warnings
state_args = _get_state_args(tool1)
reserved_args = _get_reserved_keyword_args(tool1)
print(f"\nTool 1 warning condition:")
print(f"  state_args: {state_args}")
print(f"  reserved_args.get('state'): {reserved_args.get('state')}")
print(f"  Should warn: {bool(state_args and not reserved_args.get('state'))}")

store_arg = _get_store_arg(tool2)
reserved_args2 = _get_reserved_keyword_args(tool2)
print(f"\nTool 2 warning condition:")
print(f"  store_arg: {store_arg}")
print(f"  reserved_args.get('runtime'): {reserved_args2.get('runtime')}")
print(f"  Should warn: {bool(store_arg and not reserved_args2.get('runtime'))}")
