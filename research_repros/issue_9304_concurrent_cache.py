"""Independent stress reproducer for upstream LangGraph issue #9304.

This branch is verification-only, not a competing fix or an upstream PR.
Requires an installed development checkout of LangGraph.
"""

from __future__ import annotations

import sys

from langgraph.cache.memory import InMemoryCache
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy


@task(cache_policy=CachePolicy())
def leaf(i: int) -> int:
    return i


@task(cache_policy=CachePolicy())
def parent(i: int) -> int:
    return sum(f.result() for f in [leaf(i * 1000 + j) for j in range(25)])


def run_once() -> int:
    @entrypoint(cache=InMemoryCache())
    def workflow(count: int) -> int:
        return sum(f.result() for f in [parent(i) for i in range(count)])

    return workflow.invoke(4)


def main() -> None:
    expected = sum(i * 1000 + j for i in range(4) for j in range(25))
    previous_interval = sys.getswitchinterval()
    failures: list[str] = []
    try:
        sys.setswitchinterval(1e-6)
        for trial in range(20):
            try:
                actual = run_once()
                assert actual == expected, f"expected {expected}, got {actual}"
            except Exception as exc:
                failures.append(f"run {trial + 1}: {type(exc).__name__}: {exc}")
    finally:
        sys.setswitchinterval(previous_interval)
    print(f"Failures: {len(failures)} / 20")
    for failure in failures:
        print(failure)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
