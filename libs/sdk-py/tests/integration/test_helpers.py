"""`thread.agent.get_tree` and `thread.extensions` cache identity."""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID

pytestmark = pytest.mark.integration


async def test_get_tree_and_extensions_cache_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        tree = await thread.agent.get_tree()
        assert tree, "expected non-empty tree"
        node_ids = [n["id"] for n in tree.get("nodes", [])]
        assert "stream_message" in node_ids
        assert "ask_human" in node_ids

        tree_xray = await thread.agent.get_tree(xray=True)
        assert set(tree_xray) >= {"nodes", "edges"}

        a = thread.extensions["progress"]
        b = thread.extensions["progress"]
        assert a is b, "expected cached projection instance on repeated access"


def test_get_tree_and_extensions_cache_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        tree = thread.agent.get_tree()
        assert tree, "expected non-empty tree"
        node_ids = [n["id"] for n in tree.get("nodes", [])]
        assert "stream_message" in node_ids
        assert "ask_human" in node_ids

        tree_xray = thread.agent.get_tree(xray=True)
        assert set(tree_xray) >= {"nodes", "edges"}

        a = thread.extensions["progress"]
        b = thread.extensions["progress"]
        assert a is b, "expected cached projection instance on repeated access"
