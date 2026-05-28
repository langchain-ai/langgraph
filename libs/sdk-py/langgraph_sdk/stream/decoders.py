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
