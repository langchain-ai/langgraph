"""Exercise `thread.subgraphs` against both example graphs.

Two passes:

1. `agent` (plain `StateGraph`): the parent calls a nested subgraph via
   `subgraph.invoke(...)` from a node. This may or may not surface as a
   v3 scoped child handle depending on how the server emits namespaces
   for nested invokes — included so we can compare behavior.

2. `deep_agent`: built with `create_deep_agent` + one `SubAgent`. The
   supervisor's model is scripted to issue a `task(researcher, ...)`
   call, which IS the path that produces a proper scoped child handle
   on `thread.subgraphs`. This is the canonical exercise for the v3
   scoped-subgraph surface.
"""

from __future__ import annotations

import asyncio

from _common import check_api_reachable, header, make_async_client, make_sync_client


async def _drain_subgraphs(thread) -> list:
    """Drain ``thread.subgraphs`` to a list of {path, messages} dicts.

    The outer subgraphs iterator must complete before we deep-iterate
    each handle's ``messages`` projection -- the same nested-iteration
    deadlock pattern as ``thread.tool_calls`` (see
    ``test_tools.py``). So we first collect handles, then drain
    each handle's messages serially.
    """
    handles: list = [h async for h in thread.subgraphs]
    children: list = []
    for child in handles:
        print(f"  child handle path={child.path}")
        # Just count handle paths; deep message iteration on scoped
        # handles has its own draining pattern and isn't the goal of
        # this test (which exercises subgraph discovery via
        # child-namespace ``lifecycle: started``).
        children.append({"path": child.path})
    return children


def _drain_subgraphs_sync(thread) -> list:
    handles = list(thread.subgraphs)
    children: list = []
    for child in handles:
        print(f"  child handle path={child.path}")
        children.append({"path": child.path})
    return children


async def run_async() -> None:
    threads, raw = make_async_client()
    try:
        header("async subgraphs / agent (plain StateGraph)")
        async with threads.stream(assistant_id="agent") as thread:
            await thread.run.start(input={"messages": [], "value": "init", "items": []})
            children = await _drain_subgraphs(thread)
        print(f"  agent: total subgraph handles: {len(children)}")

        header("async subgraphs / deep_agent (create_deep_agent + SubAgent)")
        async with threads.stream(assistant_id="deep_agent") as thread:
            await thread.run.start(
                input={
                    "messages": [{"role": "user", "content": "research the v3 spec"}]
                },
            )
            children = await _drain_subgraphs(thread)
        print(f"  deep_agent: total subgraph handles: {len(children)}")
        assert children, "deep_agent should produce at least one direct-child handle"
    finally:
        await raw.aclose()


def run_sync() -> None:
    threads, raw = make_sync_client()
    try:
        header("sync subgraphs / agent")
        with threads.stream(assistant_id="agent") as thread:
            thread.run.start(input={"messages": [], "value": "init", "items": []})
            children = _drain_subgraphs_sync(thread)
        print(f"  agent: total subgraph handles: {len(children)}")

        header("sync subgraphs / deep_agent")
        with threads.stream(assistant_id="deep_agent") as thread:
            thread.run.start(
                input={
                    "messages": [{"role": "user", "content": "research the v3 spec"}]
                },
            )
            children = _drain_subgraphs_sync(thread)
        print(f"  deep_agent: total subgraph handles: {len(children)}")
        assert children, "deep_agent should produce at least one direct-child handle"
    finally:
        raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
