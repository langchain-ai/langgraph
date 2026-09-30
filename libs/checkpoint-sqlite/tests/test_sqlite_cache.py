"""Regression tests for SqliteCache.

Covers the namespace encoding fix (#8660): node/checkpoint names may legally
contain the delimiter that older versions used to join namespace parts, which
broke the ns round-trip and surfaced as a KeyError on cache lookups.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from langgraph.cache.sqlite import SqliteCache, _decode_ns, _encode_ns


@pytest.fixture()
def cache() -> SqliteCache[object]:
    path = os.path.join(tempfile.mkdtemp(), "cache.db")
    return SqliteCache(path=path)


def test_ns_round_trip_with_delimiter_in_part() -> None:
    """A namespace part containing a comma must round-trip losslessly.

    Regression for #8660: `",".join`/`split(",")` turned ("foo,bar",) into
    ("foo", "bar"), so the cache key reconstructed after a read never matched
    the key that was written.
    """
    ns = ("__pregel_ns_writes", "__main__.node", "foo,bar")
    assert _decode_ns(_encode_ns(ns)) == ns


def test_ns_round_trip_multipart_and_special_chars() -> None:
    ns = ("a", "b,c", "d\ne", 'q"x', "\\r", "日本語", "")
    assert _decode_ns(_encode_ns(ns)) == ns


def test_decode_falls_back_to_legacy_comma_format() -> None:
    # Rows written by pre-#8660 versions are comma-joined and cannot be
    # distinguished from JSON; they must still decode (best effort).
    assert _decode_ns("a,b") == ("a", "b")


def test_set_get_round_trip_with_comma_in_namespace(cache: SqliteCache[object]) -> None:
    ns = ("__pregel_ns_writes", "node,with,commas")
    key = (ns, "some-checkpoint-id")
    cache.set({key: ({"value": 1}, None)})
    assert cache.get([key]) == {key: {"value": 1}}


def test_set_get_round_trip_plain_namespace(cache: SqliteCache[object]) -> None:
    ns = ("__pregel_ns_writes", "plain-node")
    key = (ns, "cid")
    cache.set({key: ({"value": 2}, None)})
    assert cache.get([key]) == {key: {"value": 2}}


def test_clear_namespace_with_comma(cache: SqliteCache[object]) -> None:
    ns = ("__pregel_ns_writes", "node,with,commas")
    key = (ns, "cid")
    cache.set({key: ({"value": 3}, None)})
    cache.clear([ns])
    assert cache.get([key]) == {}
