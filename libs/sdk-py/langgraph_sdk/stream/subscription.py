"""Subscription matching: channel inference + namespace prefix filtering.

Direct port of `libs/sdk/src/client/stream/subscription.ts` from the JS SDK.
"""

from __future__ import annotations

from typing import Any

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


def compute_union_filter(
    subscriptions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate a set of subscription filters into one covering filter.

    Direct port of `client/stream/index.ts:#computeUnionFilter`.

    - Channels are unioned.
    - Namespaces: if any subscription omits `namespaces` (wildcard), the union
      is unscoped (omits the key). Otherwise, deduplicated union.
    - Depth: if any subscription omits `depth` (unbounded), the union is
      unbounded (omits the key). Otherwise, take the max. `depth=0` is a
      valid bounded value — never omit when all subscriptions provide it.

    Args:
        subscriptions: list of `SubscribeParams`-shaped dicts.

    Returns:
        A `SubscribeParams`-shaped dict covering every input.
    """
    if not subscriptions:
        return {"channels": []}

    channels: set[str] = set()
    wildcard_namespaces = False
    namespace_map: dict[tuple[str, ...], list[str]] = {}
    unbounded_depth = False
    max_depth = 0

    for sub in subscriptions:
        for ch in sub.get("channels", []):
            channels.add(ch)

        sub_namespaces = sub.get("namespaces")
        if sub_namespaces is None:
            wildcard_namespaces = True
        elif not wildcard_namespaces:
            for ns in sub_namespaces:
                namespace_map[tuple(ns)] = ns

        sub_depth = sub.get("depth")
        if sub_depth is None:
            unbounded_depth = True
        elif not unbounded_depth and sub_depth > max_depth:
            max_depth = sub_depth

    result: dict[str, Any] = {"channels": sorted(channels)}
    if not wildcard_namespaces and namespace_map:
        result["namespaces"] = list(namespace_map.values())
    if not unbounded_depth:
        result["depth"] = max_depth
    return result


def filter_covers(coverer: dict[str, Any], target: dict[str, Any]) -> bool:
    """Whether `coverer` is a superset of `target`.

    Direct port of `client/stream/index.ts:filterCovers`. Depth coverage
    accounts for namespace-prefix offset: a scoped coverer needs enough depth
    to absorb the extra levels of any deeper target namespace prefix.
    """
    coverer_channels = set(coverer.get("channels", []))
    for ch in target.get("channels", []):
        if ch not in coverer_channels:
            return False

    coverer_depth = coverer.get("depth")
    target_depth = target.get("depth")
    coverer_namespaces = coverer.get("namespaces")
    target_namespaces = target.get("namespaces")

    # Unscoped coverer covers any namespace; depth is a simple scalar check.
    if coverer_namespaces is None:
        if coverer_depth is None:
            return True
        if target_depth is None:
            return False
        return target_depth <= coverer_depth

    # Scoped coverer cannot cover an unscoped target.
    if target_namespaces is None:
        return False

    # Each target namespace must be covered by SOME coverer namespace,
    # AND the depth-with-offset must fit.
    for tp in target_namespaces:
        covered = False
        for cp in coverer_namespaces:
            if not is_prefix_match(tp, cp):
                continue
            if coverer_depth is None:
                covered = True
                break
            if target_depth is None:
                # target wants unbounded depth — coverer bounded can't cover.
                continue
            if len(tp) - len(cp) + target_depth <= coverer_depth:
                covered = True
                break
        if not covered:
            return False
    return True
