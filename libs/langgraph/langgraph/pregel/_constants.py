from __future__ import annotations

import asyncio
import concurrent.futures
import weakref

SKIP_RERAISE_SET: weakref.WeakSet[concurrent.futures.Future | asyncio.Future] = (
    weakref.WeakSet()
)
