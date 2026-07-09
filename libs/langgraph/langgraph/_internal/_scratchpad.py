import dataclasses
from collections.abc import Callable
from typing import Any

from langgraph.types import _DC_KWARGS


@dataclasses.dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int
    stop: int
    # call
    call_counter: Callable[[], int]
    # interrupt
    interrupt_counter: Callable[[], int]
    get_null_resume: Callable[[bool], Any]
    resume: list[Any]
    # fetch: content-addressed id -> terminal FetchResult persisted from prior runs.
    fetch_results: dict[str, Any]
    # fetch: content-addressed id -> value/FetchResult delivered this run via
    # Command(fetch=...). Kept separate from fetch_results so a deadline can be enforced
    # on fresh deliveries (a late fulfillment is rejected) but not on already-recorded
    # terminal results.
    fetch_delivered: dict[str, Any]
    # fetch: content-addressed id -> the FetchRequest persisted at first suspension, so
    # re-execution reuses the original deadline/created_at instead of recomputing them.
    fetch_pending: dict[str, Any]
    # subgraph
    subgraph_counter: Callable[[], int]
