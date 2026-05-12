from __future__ import annotations

from langgraph_sdk.stream.subscription import is_prefix_match, normalize_segment


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
