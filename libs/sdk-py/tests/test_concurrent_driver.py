from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from integration.test_concurrent import _drive_one_async, _drive_one_sync


def _thread():
    thread = MagicMock(thread_id="thread")
    type(thread).interrupted = PropertyMock(side_effect=[False, False, True])
    thread.values.__iter__.return_value = []
    thread.values.__aiter__.return_value = []
    return thread


def test_sync_driver_waits_for_delayed_interrupt():
    thread = _thread()
    thread.output = {"items": ["done"]}
    threads = MagicMock()
    threads.stream.return_value.__enter__.return_value = thread
    results = {}

    _drive_one_sync(threads, "A", results)

    thread.run.respond.assert_called_once_with("yes")
    assert results["A"] == {"thread_id": "thread", "items": ["done"]}


async def test_async_driver_waits_for_delayed_interrupt():
    thread = _thread()
    thread.run.start = AsyncMock()
    thread.run.respond = AsyncMock()
    thread.output = AsyncMock(return_value={"items": ["done"]})()
    threads = MagicMock()
    threads.stream.return_value.__aenter__.return_value = thread

    result = await _drive_one_async(threads)

    thread.run.respond.assert_awaited_once_with("yes")
    assert result == {"thread_id": "thread", "items": ["done"]}


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_driver_fails_if_interrupt_never_arrives(monkeypatch, asynchronous):
    monkeypatch.setattr("integration.test_concurrent._INTERRUPT_TIMEOUT", 0)
    thread = _thread()
    threads = MagicMock()
    threads.stream.return_value.__enter__.return_value = thread
    threads.stream.return_value.__aenter__.return_value = thread
    if asynchronous:
        thread.run.start = AsyncMock()
    with pytest.raises(AssertionError, match="expected an interrupt before output"):
        if asynchronous:
            await _drive_one_async(threads)
        else:
            _drive_one_sync(threads, "A", {})
    thread.run.respond.assert_not_called()
