"""Namespace prefix filtering for subscription matching.

Direct port of `libs/sdk/src/client/stream/subscription.ts` from the JS SDK.
Channel inference and full filter matching land in Task 3.
"""

from __future__ import annotations

from langchain_protocol import Channel, Event, Namespace, SubscribeParams  # noqa: F401


def normalize_segment(segment: str) -> str:
    """Strip the dynamic suffix after `:` from a namespace segment."""
    idx = segment.find(":")
    return segment if idx == -1 else segment[:idx]


def is_prefix_match(event_namespace: Namespace, prefix: Namespace) -> bool:
    """Whether `event_namespace` starts with `prefix`.

    Segments compare literally first; if the prefix segment contains no `:`,
    the candidate is also compared after its dynamic suffix is stripped.
    Mirrors `is_prefix_match` in `api/langgraph_api/protocol/namespace.py`.
    """
    if len(prefix) > len(event_namespace):
        return False
    for seg, candidate in zip(prefix, event_namespace, strict=False):
        if candidate == seg:
            continue
        if ":" in seg:
            return False
        if normalize_segment(candidate) == seg:
            continue
        return False
    return True
