from __future__ import annotations

import pytest
from langchain_protocol import (
    Channel as ProtocolChannel,
)
from langchain_protocol import (
    Event as ProtocolEvent,
)
from langchain_protocol import (
    Namespace as ProtocolNamespace,
)
from langchain_protocol import (
    SubscribeParams as ProtocolSubscribeParams,
)

from langgraph_sdk.stream import (
    Channel,
    Event,
    Namespace,
    SubscribeParams,
)
from langgraph_sdk.stream.subscription import (
    compute_union_filter,
    filter_covers,
    infer_channel,
    is_prefix_match,
    matches_subscription,
    namespace_matches,
    normalize_segment,
)
from streaming._events import (
    custom_event,
    lifecycle_event,
    values_event,
)


def test_normalize_segment_strips_suffix_after_colon():
    assert normalize_segment("fetcher:abc-uuid") == "fetcher"


def test_normalize_segment_passes_through_when_no_colon():
    assert normalize_segment("fetcher") == "fetcher"


def test_is_prefix_match_empty_prefix_matches_anything():
    assert is_prefix_match(["a", "b"], []) is True


def test_is_prefix_match_exact_literal_match():
    assert is_prefix_match(["fetcher", "inner"], ["fetcher"]) is True


def test_is_prefix_match_strips_runtime_suffix_when_prefix_is_static():
    assert is_prefix_match(["fetcher:abc-uuid", "inner"], ["fetcher"]) is True


def test_is_prefix_match_keeps_colons_in_prefix_literal():
    # When the prefix itself contains a colon it is treated as exact.
    assert is_prefix_match(["fetcher:abc"], ["fetcher:abc"]) is True
    assert is_prefix_match(["fetcher:xyz"], ["fetcher:abc"]) is False


def test_is_prefix_match_returns_false_when_prefix_longer():
    assert is_prefix_match(["a"], ["a", "b"]) is False


def test_namespace_matches_no_prefixes_matches_anything():
    assert namespace_matches(["any", "ns"], None, None) is True
    assert namespace_matches(["any", "ns"], [], None) is True


def test_namespace_matches_with_depth_limit():
    assert namespace_matches(["a", "b"], [["a"]], 1) is True
    assert namespace_matches(["a", "b", "c"], [["a"]], 1) is False


@pytest.mark.parametrize(
    ("method", "expected_channel"),
    [
        ("values", "values"),
        ("checkpoints", "checkpoints"),
        ("updates", "updates"),
        ("messages", "messages"),
        ("tools", "tools"),
        ("lifecycle", "lifecycle"),
        ("tasks", "tasks"),
        ("input.requested", "input"),
    ],
)
def test_infer_channel_for_each_method(method, expected_channel):
    event = {
        "type": "event",
        "method": method,
        "params": {"namespace": [], "data": {}},
        "seq": 0,
        "event_id": "e",
    }
    assert infer_channel(event) == expected_channel  # ty: ignore[invalid-argument-type]


def test_infer_channel_custom_with_name_produces_namespaced_channel():
    assert infer_channel(custom_event(name="my_ext")) == "custom:my_ext"  # ty:ignore[invalid-argument-type]


def test_infer_channel_custom_without_name_falls_back_to_bare_custom():
    assert infer_channel(custom_event(name="")) == "custom"  # ty:ignore[invalid-argument-type]


def test_infer_channel_unknown_method_returns_none():
    assert (
        infer_channel(
            {
                "type": "event",
                "method": "unknown",
                "params": {"namespace": [], "data": {}},
                "seq": 0,
                "event_id": "e",
            }  # ty:ignore[invalid-argument-type]
        )
        is None
    )


def test_matches_subscription_channel_in_filter():
    sub = {"channels": ["values"]}
    assert matches_subscription(values_event(), sub) is True  # ty:ignore[invalid-argument-type]
    assert matches_subscription(lifecycle_event(), sub) is False  # ty:ignore[invalid-argument-type]


def test_matches_subscription_bare_custom_covers_namespaced_custom():
    sub = {"channels": ["custom"]}
    assert matches_subscription(custom_event(name="my_ext"), sub) is True  # ty:ignore[invalid-argument-type]


def test_matches_subscription_namespace_filter_applied():
    sub = {"channels": ["values"], "namespaces": [["fetcher"]]}
    assert (
        matches_subscription(values_event(namespace=["fetcher", "inner"]), sub) is True  # ty:ignore[invalid-argument-type]
    )
    assert matches_subscription(values_event(namespace=["other"]), sub) is False  # ty:ignore[invalid-argument-type]


def test_matches_subscription_bare_custom_event_matches_bare_custom_filter():
    sub = {"channels": ["custom"]}
    assert matches_subscription(custom_event(name=""), sub) is True  # ty: ignore[invalid-argument-type]


def test_compute_union_filter_merges_channels():
    a = {"channels": ["values"]}
    b = {"channels": ["messages", "lifecycle"]}
    result = compute_union_filter([a, b])
    assert set(result["channels"]) == {"values", "messages", "lifecycle"}


def test_compute_union_filter_drops_namespaces_when_any_subscription_is_unscoped():
    a = {"channels": ["values"], "namespaces": [["fetcher"]]}
    b = {"channels": ["messages"]}  # no namespaces == wildcard
    result = compute_union_filter([a, b])
    # If any subscription is unscoped, the union must be unscoped.
    assert "namespaces" not in result or result.get("namespaces") is None


def test_compute_union_filter_unions_namespaces_when_all_scoped():
    a = {"channels": ["values"], "namespaces": [["fetcher"]]}
    b = {"channels": ["messages"], "namespaces": [["scorer"]]}
    result = compute_union_filter([a, b])
    assert sorted(result["namespaces"]) == [["fetcher"], ["scorer"]]


def test_compute_union_filter_takes_max_depth():
    a = {"channels": ["values"], "depth": 1}
    b = {"channels": ["messages"], "depth": 3}
    result = compute_union_filter([a, b])
    assert result["depth"] == 3


def test_compute_union_filter_omits_depth_when_any_subscription_omits():
    a = {"channels": ["values"], "depth": 2}
    b = {"channels": ["messages"]}  # no depth == unbounded
    result = compute_union_filter([a, b])
    assert "depth" not in result or result.get("depth") is None


def test_compute_union_filter_empty_input_returns_empty_channel_filter():
    result = compute_union_filter([])
    assert result == {"channels": []}


def test_filter_covers_same_filter():
    f = {"channels": ["values", "messages"]}
    assert filter_covers(f, f) is True


def test_filter_covers_superset_channels():
    coverer = {"channels": ["values", "messages", "lifecycle"]}
    target = {"channels": ["values", "messages"]}
    assert filter_covers(coverer, target) is True


def test_filter_covers_missing_channel():
    coverer = {"channels": ["values"]}
    target = {"channels": ["values", "messages"]}
    assert filter_covers(coverer, target) is False


def test_filter_covers_unscoped_covers_scoped():
    coverer = {"channels": ["values"]}  # wildcard namespaces
    target = {"channels": ["values"], "namespaces": [["fetcher"]]}
    assert filter_covers(coverer, target) is True


def test_filter_covers_scoped_does_not_cover_unscoped():
    coverer = {"channels": ["values"], "namespaces": [["fetcher"]]}
    target = {"channels": ["values"]}  # wildcard
    assert filter_covers(coverer, target) is False


def test_filter_covers_depth_with_namespace_offset():
    # Coverer at depth 1 from ["agent"] reaches ["agent", X].
    # Target needs depth 1 from ["agent", "tool"] — i.e., ["agent", "tool", X].
    # That's 2 levels past coverer's prefix, but coverer only allows 1.
    coverer = {"channels": ["values"], "namespaces": [["agent"]], "depth": 1}
    target = {
        "channels": ["values"],
        "namespaces": [["agent", "tool"]],
        "depth": 1,
    }
    assert filter_covers(coverer, target) is False


def test_filter_covers_depth_with_offset_enough_depth():
    # Same setup but coverer depth=2 absorbs the offset.
    coverer = {"channels": ["values"], "namespaces": [["agent"]], "depth": 2}
    target = {
        "channels": ["values"],
        "namespaces": [["agent", "tool"]],
        "depth": 1,
    }
    assert filter_covers(coverer, target) is True


def test_filter_covers_unscoped_coverer_with_bounded_depth():
    # Coverer is unscoped; depth comparison is the simple scalar form.
    coverer = {"channels": ["values"], "depth": 2}
    target = {"channels": ["values"], "depth": 1}
    assert filter_covers(coverer, target) is True
    target_too_deep = {"channels": ["values"], "depth": 3}
    assert filter_covers(coverer, target_too_deep) is False


def test_filter_covers_bounded_coverer_does_not_cover_unbounded_target():
    coverer = {"channels": ["values"], "depth": 2}
    target = {"channels": ["values"]}  # unbounded
    assert filter_covers(coverer, target) is False


def test_protocol_types_are_importable_from_stream_module():
    """Test that v3 protocol types are re-exported from langgraph_sdk.stream."""
    assert Channel is ProtocolChannel
    assert Event is ProtocolEvent
    assert Namespace is ProtocolNamespace
    assert SubscribeParams is ProtocolSubscribeParams
