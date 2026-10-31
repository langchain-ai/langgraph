"""A task that calls other tasks pushes them from its own worker thread, so
several threads can register tasks and match them against the node cache at
the same time. That used to fail with "dictionary changed size during
iteration" whenever one thread added to the loop's task map while another was
iterating it to collect cache keys."""

import sys

from langgraph.cache.memory import InMemoryCache

from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy

PARENTS = 4
LEAVES = 25
RUNS = 10


def test_parallel_tasks_pushing_cached_tasks() -> None:
    @task(cache_policy=CachePolicy())
    def leaf(i: int) -> int:
        return i

    @task(cache_policy=CachePolicy())
    def parent(i: int) -> int:
        return sum(f.result() for f in [leaf(i * 1000 + j) for j in range(LEAVES)])

    @entrypoint(cache=InMemoryCache())  # A fresh cache per run, so every task runs.
    def main(n: int) -> int:
        return sum(f.result() for f in [parent(i) for i in range(n)])

    expected = sum(i * 1000 + j for i in range(PARENTS) for j in range(LEAVES))
    # Hand off between threads as soon as possible, so the pushes overlap.
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(RUNS):
            assert main.invoke(PARENTS) == expected
    finally:
        sys.setswitchinterval(switch_interval)
