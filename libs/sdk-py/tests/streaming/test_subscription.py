from __future__ import annotations

import pytest

from langgraph_sdk.stream.subscription import (
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
        "method": method,
        "params": {"namespace": [], "data": {}},
        "seq": 0,
        "id": "e",
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
                "method": "unknown",
                "params": {"namespace": [], "data": {}},
                "seq": 0,
                "id": "e",
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
