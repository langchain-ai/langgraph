"""
Standard state reducers for LangGraph.

These functions are designed to be used with `Annotated[...]` in State
definitions to handle complex state-merging logic automatically, e.g. when
multiple nodes write to the same channel within a single superstep.
"""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


def smart_merge_dict(
    current: dict[str, Any] | None,
    update: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Deep-merge reducer that resolves conflicts instead of overwriting.

    Strategy per key:
      1. New key            -> add it.
      2. dict   + dict      -> recurse (deep merge).
      3. list   + list      -> concatenate.
      4. scalar/type clash  -> upgrade to a list (preserve both values).

    Note: this is a *shallow* copy at each level. Touched branches are rebuilt
    into fresh objects, but keys present only in `current` keep their original
    references. Reducers must never mutate state in place (already a LangGraph
    requirement), so this is safe and avoids a full deepcopy on large states.
    """
    if current is None:
        return dict(update) if update is not None else {}
    if update is None:
        return dict(current)

    new_state = current.copy()

    for key, value in update.items():
        if key not in new_state:
            new_state[key] = value
            continue

        current_val = new_state[key]

        # Deep-merge nested dicts.
        if isinstance(current_val, dict) and isinstance(value, dict):
            new_state[key] = smart_merge_dict(current_val, value)
        # Concatenate lists.
        elif isinstance(current_val, list) and isinstance(value, list):
            new_state[key] = current_val + value
        # Conflict / scalar -> upgrade to list (preserve history).
        else:
            left = current_val if isinstance(current_val, list) else [current_val]
            right = value if isinstance(value, list) else [value]
            new_state[key] = left + right

    return new_state


def combine_distinct(current: list[T] | None, update: list[T] | None) -> list[T]:
    """
    Merge two lists, dropping duplicates while preserving first-seen order.

    Unlike a naive `set`-based dedupe, this also works when elements are
    unhashable (e.g. dicts in RAG document lists): hashable elements use a fast
    `dict.fromkeys` path, and the presence of any unhashable element falls back
    to an order-preserving equality scan.
    """
    if current is None:
        return list(update) if update is not None else []
    if update is None:
        return list(current)

    combined = [*current, *update]
    try:
        # Fast path: every element is hashable.
        return list(dict.fromkeys(combined))
    except TypeError:
        # At least one unhashable element (dict, list, ...): O(n^2) fallback.
        merged: list[T] = []
        for item in combined:
            if not any(item == seen for seen in merged):
                merged.append(item)
        return merged


def first_wins(current: T | None, update: T) -> T:
    """
    Keep the first value written and ignore later updates
    (the inverse of LangGraph's default last-write-wins).
    """
    if current is not None:
        return current
    return update
