#!/usr/bin/env python
"""Debug why InjectedState warning is not being emitted."""

from typing import Annotated, get_args
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.prebuilt.tool_node import get_all_basemodel_annotations, _is_injection
from langchain_core.tools import StructuredTool
from langgraph.store.base import BaseStore


def tool_with_injected_state(x: int, state: Annotated[dict, InjectedState]) -> str:
    """Tool using deprecated InjectedState annotation."""
    return f"state: {state.get('foo', 'none')}"

def tool_with_injected_store(x: int, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Tool using deprecated InjectedStore annotation."""
    return "has store"

# Convert to tools
tool1 = StructuredTool.from_function(tool_with_injected_state)
tool2 = StructuredTool.from_function(tool_with_injected_store)

print("Debugging annotation detection:")

print(f"\nTool 1 (InjectedState):")
schema1 = tool1.get_input_schema()
print(f"  Schema fields: {list(schema1.__fields__.keys()) if hasattr(schema1, '__fields__') else list(schema1.model_fields.keys())}")

for name, type_ in get_all_basemodel_annotations(schema1).items():
    print(f"  Field '{name}': type={type_}")
    args = get_args(type_)
    print(f"    Type args: {args}")
    for arg in args:
        print(f"      Is InjectedState? {_is_injection(arg, InjectedState)}")
        print(f"      Type of arg: {type(arg)}")

print(f"\nTool 2 (InjectedStore):")
schema2 = tool2.get_input_schema()
print(f"  Schema fields: {list(schema2.__fields__.keys()) if hasattr(schema2, '__fields__') else list(schema2.model_fields.keys())}")

for name, type_ in get_all_basemodel_annotations(schema2).items():
    print(f"  Field '{name}': type={type_}")
    args = get_args(type_)
    print(f"    Type args: {args}")
    for arg in args:
        print(f"      Is InjectedStore? {_is_injection(arg, InjectedStore)}")
        print(f"      Type of arg: {type(arg)}")
