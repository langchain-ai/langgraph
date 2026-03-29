"""
Standard state reducers for LangGraph.
These functions are designed to be used with Annotated[...] in State definitions
to handle complex state merging logic automatically.
"""
from typing import Any, TypeVar

T = TypeVar("T")

def smart_merge_dict(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """
    A deep merge reducer that intelligently handles conflicts.
    
    Strategy:
    1. If key is new: Add it.
    2. If key exists and both are dicts: Recurse (Deep Merge).
    3. If key exists and types match (list/list): Extend.
    4. If key exists and types conflict or are scalar: Turn into list (Preserve History).
    """
    if not current:
        return update
    if not update:
        return current
        
    # Always copy to enforce immutability principles in Graph state
    new_state = current.copy()
    
    for key, value in update.items():
        if key not in new_state:
            new_state[key] = value
            continue
            
        current_val = new_state[key]
        
        # Scenario A: Deep Merge for nested dictionaries
        if isinstance(current_val, dict) and isinstance(value, dict):
            new_state[key] = smart_merge_dict(current_val, value)
            
        # Scenario B: List extension
        elif isinstance(current_val, list) and isinstance(value, list):
            new_state[key] = current_val + value
            
        # Scenario C: Conflict resolution -> Upgrade to List
        else:
            # Normalize current value to list
            left = current_val if isinstance(current_val, list) else [current_val]
            # Normalize new value to list
            right = value if isinstance(value, list) else [value]
            new_state[key] = left + right
            
    return new_state

def combine_distinct(current: list[T], update: list[T]) -> list[T]:
    """
    Merges two lists while removing duplicates, preserving order.
    Useful for accumulating tags, tool names, or unique references.
    """
    if not current:
        return update
    if not update:
        return current
    
    # Use dict.fromkeys to preserve order while deduping (Python 3.7+)
    merged = list(dict.fromkeys(current + update))
    return merged

def first_wins(current: T | None, update: T) -> T:
    """
    A reducer where the existing state is preserved, ignoring updates.
    Opposite of the default 'Last-Write-Wins'.
    """
    if current is not None:
        return current
    return update