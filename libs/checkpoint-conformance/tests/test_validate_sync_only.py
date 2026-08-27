"""Tests for sync-only checkpointer detection."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.capabilities import DetectedCapabilities


class SyncOnlySaver(InMemorySaver):
    """In-memory saver with only the sync API exposed as implemented."""

    aget_tuple = BaseCheckpointSaver.aget_tuple
    alist = BaseCheckpointSaver.alist
    aput = BaseCheckpointSaver.aput
    aput_writes = BaseCheckpointSaver.aput_writes
    adelete_thread = BaseCheckpointSaver.adelete_thread


class SignpostedSyncOnlySaver(SyncOnlySaver):
    """Sync-only saver with async methods that explicitly refuse calls."""

    async def aget_tuple(self, *args, **kwargs):
        raise NotImplementedError("use an async saver")

    async def alist(self, *args, **kwargs):
        raise NotImplementedError("use an async saver")
        yield

    async def aput(self, *args, **kwargs):
        raise NotImplementedError("use an async saver")


@checkpointer_test(name="SignpostedSyncOnlySaver")
async def signposted_sync_only_checkpointer():
    yield SignpostedSyncOnlySaver()


@checkpointer_test(name="SyncOnlySaver")
async def sync_only_checkpointer():
    yield SyncOnlySaver()


def test_detects_sync_only_savers_with_or_without_async_signposts() -> None:
    assert DetectedCapabilities.from_instance(SyncOnlySaver()).sync_only
    assert DetectedCapabilities.from_instance(SignpostedSyncOnlySaver()).sync_only
    assert not DetectedCapabilities.from_instance(InMemorySaver()).sync_only


async def test_validate_skips_sync_only_saver(capsys) -> None:
    for registered in (sync_only_checkpointer, signposted_sync_only_checkpointer):
        report = await validate(registered)

        assert report.conformance_level() == "SKIPPED"
        assert report.skip_reason == "sync-only saver"
        assert not report.passed_all()
        assert not report.passed_all_base()
        assert all(
            result.skip_reason == "sync-only saver"
            for result in report.results.values()
        )

    report.print_report()
    assert "Result: SKIPPED (sync-only saver)" in capsys.readouterr().out
