"""Tracing annotations for controlling field visibility in LangSmith traces.

This module provides annotations that can be used with Python's `Annotated` type
to control how state fields are traced in LangSmith.

Example:
    ```python
    from typing import Annotated
    from langgraph.tracing import NotTraced, AsAttachment

    class State(TypedDict):
        query: str  # Traced normally in all nodes
        large_pdf: Annotated[bytes, NotTraced()]  # Never traced
        results: Annotated[list, AsAttachment()]  # Traced in root, masked in child runs
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class TraceLevel(str, Enum):
    """Enum defining trace visibility levels."""

    TRACED = "traced"  # Normal tracing (default)
    NOT_TRACED = "not_traced"  # Never traced
    ROOT_ONLY = "root_only"  # Only traced in root run
    AS_ATTACHMENT = "as_attachment"  # Traced in root, masked in children


@dataclass(frozen=True)
class TraceAnnotation:
    """Base class for tracing annotations.

    This annotation can be added to state field type hints using Python's
    `Annotated` type to control how that field is traced in LangSmith.
    """

    level: TraceLevel
    mask_value: Any = "[MASKED]"

    def should_trace(self, is_root: bool = False) -> bool:
        """Determine if the field should be traced in the current context.

        Args:
            is_root: Whether this is the root run (not a child node).

        Returns:
            True if the field should be included in the trace.
        """
        if self.level == TraceLevel.TRACED:
            return True
        elif self.level == TraceLevel.NOT_TRACED:
            return False
        elif self.level == TraceLevel.ROOT_ONLY:
            return is_root
        elif self.level == TraceLevel.AS_ATTACHMENT:
            return is_root
        return True

    def get_trace_value(self, value: Any, is_root: bool = False) -> Any:
        """Get the value to use in the trace.

        Args:
            value: The actual field value.
            is_root: Whether this is the root run.

        Returns:
            The value to include in the trace (or mask value if masked).
        """
        if self.should_trace(is_root):
            return value
        return self.mask_value


class Traced(TraceAnnotation):
    """Annotation to explicitly mark a field as traced (default behavior).

    Example:
        ```python
        from typing import Annotated
        from langgraph.tracing import Traced

        class State(TypedDict):
            query: Annotated[str, Traced()]  # Explicitly traced
        ```
    """

    def __init__(self, mask_value: Any = "[MASKED]"):
        super().__init__(level=TraceLevel.TRACED, mask_value=mask_value)


class NotTraced(TraceAnnotation):
    """Annotation to exclude a field from all tracing.

    Use this for large data (PDFs, images, large dataframes) that should
    never be sent to LangSmith.

    Example:
        ```python
        from typing import Annotated
        from langgraph.tracing import NotTraced

        class State(TypedDict):
            large_pdf: Annotated[bytes, NotTraced()]
            dataframe_dicts: Annotated[dict, NotTraced(mask_value="<dataframe>")]
        ```
    """

    def __init__(self, mask_value: Any = "[NOT TRACED]"):
        super().__init__(level=TraceLevel.NOT_TRACED, mask_value=mask_value)


class RootOnly(TraceAnnotation):
    """Annotation to trace a field only in the root run.

    The field will be traced when the graph is invoked, but masked in all
    child node executions.

    Example:
        ```python
        from typing import Annotated
        from langgraph.tracing import RootOnly

        class State(TypedDict):
            input_document: Annotated[str, RootOnly()]
        ```
    """

    def __init__(self, mask_value: Any = "[MASKED IN CHILD RUN]"):
        super().__init__(level=TraceLevel.ROOT_ONLY, mask_value=mask_value)


class AsAttachment(TraceAnnotation):
    """Annotation to trace a field in root run, mask in child runs.

    Similar to RootOnly but semantically indicates the value is an "attachment"
    that should be visible at the top level but not repeated in every child trace.

    Example:
        ```python
        from typing import Annotated
        from langgraph.tracing import AsAttachment

        class State(TypedDict):
            pdf_content: Annotated[bytes, AsAttachment(mask_value="<PDF attachment>")]
        ```
    """

    def __init__(self, mask_value: Any = "[SEE ROOT TRACE]"):
        super().__init__(level=TraceLevel.AS_ATTACHMENT, mask_value=mask_value)


def get_trace_annotation(typ: type[Any]) -> TraceAnnotation | None:
    """Extract tracing annotation from a type hint.

    Args:
        typ: A type hint, possibly Annotated with a TraceAnnotation.

    Returns:
        The TraceAnnotation instance if present, otherwise None.
    """
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        # Search through all annotations to find tracing annotations
        for item in meta:
            if isinstance(item, TraceAnnotation):
                return item
    return None


def filter_state_for_tracing(
    state: dict[str, Any],
    type_hints: dict[str, type],
    is_root: bool = False,
) -> dict[str, Any]:
    """Filter state dictionary based on tracing annotations.

    Args:
        state: The state dictionary to filter.
        type_hints: Type hints for the state fields (from get_type_hints).
        is_root: Whether this is the root run (not a child node).

    Returns:
        A filtered copy of the state with masked values where appropriate.
    """
    if not state or not type_hints:
        return state

    filtered = {}
    for key, value in state.items():
        if key not in type_hints:
            # No type hint, include as-is
            filtered[key] = value
            continue

        annotation = get_trace_annotation(type_hints[key])
        if annotation is None:
            # No tracing annotation, include as-is
            filtered[key] = value
        else:
            # Apply tracing annotation
            filtered[key] = annotation.get_trace_value(value, is_root)

    return filtered


__all__ = [
    "TraceLevel",
    "TraceAnnotation",
    "Traced",
    "NotTraced",
    "RootOnly",
    "AsAttachment",
    "get_trace_annotation",
    "filter_state_for_tracing",
]
