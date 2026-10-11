import dataclasses
import threading
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
    # subgraph
    subgraph_counter: Callable[[], int]
    subgraph_invocation_counters: dict[int, int] = dataclasses.field(
        default_factory=dict
    )
    subgraph_namespace_bindings: dict[tuple[int, int], str] = dataclasses.field(
        default_factory=dict
    )
    subgraph_lock: Any = dataclasses.field(default_factory=threading.Lock)

    def next_subgraph_invocation(self, subgraph_id: int) -> int:
        with self.subgraph_lock:
            invocation = self.subgraph_invocation_counters.get(subgraph_id, 0)
            self.subgraph_invocation_counters[subgraph_id] = invocation + 1
            return invocation

    def reset_subgraph_invocations(self) -> None:
        with self.subgraph_lock:
            self.subgraph_invocation_counters.clear()
