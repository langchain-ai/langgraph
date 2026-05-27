"""`CronClient` against the integration API.

Covers create / search / delete (no `update` since the surface accepts a
sparse update and the round-trip is implicitly exercised by the others).
The schedule fires in the future so the cron is never executed during
the test; we tear it down before any tick can land.
"""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID

pytestmark = pytest.mark.integration


def _async_crons(raw):
    from langgraph_sdk._async.cron import CronClient
    from langgraph_sdk._async.http import HttpClient

    return CronClient(HttpClient(raw))


def _sync_crons(raw):
    from langgraph_sdk._sync.cron import SyncCronClient
    from langgraph_sdk._sync.http import SyncHttpClient

    return SyncCronClient(SyncHttpClient(raw))


# Once a year, on Jan 1 at 00:00 UTC. Deterministic and well past any
# test runtime.
_DISTANT_SCHEDULE = "0 0 1 1 *"


async def test_crons_create_search_delete_async(async_threads) -> None:
    _, raw = async_threads
    crons = _async_crons(raw)
    created = await crons.create(
        ASSISTANT_ID,
        schedule=_DISTANT_SCHEDULE,
        input={"messages": [], "value": "init", "items": []},
        metadata={"suite": "integration", "label": "crons-async"},
    )
    cron_id = created["cron_id"]
    try:
        results = await crons.search(limit=20)
        assert any(c["cron_id"] == cron_id for c in results)
    finally:
        await crons.delete(cron_id)


def test_crons_create_search_delete_sync(sync_threads) -> None:
    _, raw = sync_threads
    crons = _sync_crons(raw)
    created = crons.create(
        ASSISTANT_ID,
        schedule=_DISTANT_SCHEDULE,
        input={"messages": [], "value": "init", "items": []},
        metadata={"suite": "integration", "label": "crons-sync"},
    )
    cron_id = created["cron_id"]
    try:
        results = crons.search(limit=20)
        assert any(c["cron_id"] == cron_id for c in results)
    finally:
        crons.delete(cron_id)
