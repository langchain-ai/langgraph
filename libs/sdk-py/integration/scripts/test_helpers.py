"""Exercise helper methods on `thread` against the integration API.

Covers `thread.agent.get_tree(xray=...)` and the extensions cache.
"""

from __future__ import annotations

import asyncio

from _common import (
    ASSISTANT_ID,
    check_api_reachable,
    header,
    make_async_client,
    make_sync_client,
)


async def run_async() -> None:
    header("async helpers (agent.get_tree, extensions cache)")
    threads, raw = make_async_client()
    try:
        async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            tree = await thread.agent.get_tree()
            print(f"  get_tree() returned: nodes={list(tree.get('nodes', []))[:5]} ...")
            assert tree, "expected non-empty tree"

            tree_xray = await thread.agent.get_tree(xray=True)
            print(f"  get_tree(xray=True) returned keys: {list(tree_xray)[:5]}")

            # Extensions cache: same name returns same projection instance.
            a = thread.extensions["progress"]
            b = thread.extensions["progress"]
            assert a is b, "expected cached _ExtensionProjection on repeated access"
            print("  extensions cache: OK (same projection instance reused)")
    finally:
        await raw.aclose()


def run_sync() -> None:
    header("sync helpers (agent.get_tree, extensions cache)")
    threads, raw = make_sync_client()
    try:
        with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            tree = thread.agent.get_tree()
            print(f"  get_tree() returned: nodes={list(tree.get('nodes', []))[:5]} ...")
            assert tree, "expected non-empty tree"

            tree_xray = thread.agent.get_tree(xray=True)
            print(f"  get_tree(xray=True) returned keys: {list(tree_xray)[:5]}")

            a = thread.extensions["progress"]
            b = thread.extensions["progress"]
            assert a is b, "expected cached _ExtensionProjection on repeated access"
            print("  extensions cache: OK (same projection instance reused)")
    finally:
        raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
