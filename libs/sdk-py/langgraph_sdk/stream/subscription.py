"""Subscription matching: channel inference + namespace prefix filtering.

Direct port of `libs/sdk/src/client/stream/subscription.ts` from the JS SDK.
"""

from __future__ import annotations

from langchain_protocol import Channel, Event, Namespace, SubscribeParams


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


def namespace_matches(
    event_namespace: Namespace,
    prefixes: list[Namespace] | None,
    depth: int | None,
) -> bool:
    """Whether `event_namespace` matches any of `prefixes` within `depth`."""
    if not prefixes:
        return True
    for prefix in prefixes:
        if not is_prefix_match(event_namespace, prefix):
            continue
        if depth is None:
            return True
        if len(event_namespace) - len(prefix) <= depth:
            return True
    return False


_DIRECT_METHODS = {
    "values",
    "checkpoints",
    "updates",
    "messages",
    "tools",
    "lifecycle",
    "tasks",
}


def infer_channel(event: Event) -> Channel | None:
    """Map a protocol event's `method` to its subscription channel.

    Returns `None` for unrecognized methods so new server-side channels (e.g.
    from extension transformers) don't break existing clients.
    """
    method = event.get("method")
    if method in _DIRECT_METHODS:
        return method  # type: ignore[return-value]
    if method == "custom":
        params = event.get("params") or {}
        data = params.get("data") if isinstance(params, dict) else None
        name = data.get("name") if isinstance(data, dict) else None
        # JS uses != null; truthiness here treats name="" the same as missing.
        return f"custom:{name}" if name else "custom"
    if method == "input.requested":
        return "input"
    return None


def matches_subscription(event: Event, definition: SubscribeParams) -> bool:
    """Whether `event` should be delivered for `definition`."""
    channel = infer_channel(event)
    if channel is None:
        return False
    channels = definition.get("channels", [])
    if channel not in channels and not (
        channel.startswith("custom:") and "custom" in channels
    ):
        return False
    params = event.get("params") or {}
    namespace = params.get("namespace", []) if isinstance(params, dict) else []
    return namespace_matches(
        namespace,
        definition.get("namespaces"),
        definition.get("depth"),
    )
