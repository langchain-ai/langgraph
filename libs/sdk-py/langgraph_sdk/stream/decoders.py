"""Per-channel event → items state machines.

Used both by the projection iterators (`_ValuesProjection`,
`_MessagesProjection`, `_ToolCallsProjection`, `_SubgraphsProjection`) on
`AsyncThreadStream` / `SyncThreadStream`, and by `interleave_projections`,
which drives multiple decoders from one shared subscription.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class Decoder(Protocol):
    def feed(self, event: dict[str, Any]) -> Iterable[Any]: ...


class ValuesDecoder:
    """Yields snapshot dicts from `values` method events.

    Mirrors the per-event body of `_ValuesProjection._values_iter` in
    `langgraph_sdk/_async/stream.py` (the REST-state seeding stays at the
    projection layer; it is a one-shot pre-stream fetch, not part of the
    event state machine).
    """

    def feed(self, event: dict[str, Any]) -> Iterable[Any]:
        if event.get("method") == "values":
            params = event.get("params") or {}
            data = params.get("data")
            if data is not None:
                yield data
