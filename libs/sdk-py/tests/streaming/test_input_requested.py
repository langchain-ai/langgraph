from __future__ import annotations

import warnings
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_protocol import Event

from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk._sync.stream import SyncThreadStream


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("data", "expected", "legacy"),
    [
        ({"payload": {"question": "Continue?"}}, {"question": "Continue?"}, False),
        ({"payload": "canonical", "value": "legacy"}, "canonical", False),
        ({"payload": None, "value": "legacy"}, None, False),
        ({"payload": False, "value": "legacy"}, False, False),
        ({"payload": 0, "value": "legacy"}, 0, False),
        ({"payload": "", "value": "legacy"}, "", False),
        ({"payload": {}, "value": "legacy"}, {}, False),
        ({"payload": [], "value": "legacy"}, [], False),
        ({"value": {"question": "Continue?"}}, {"question": "Continue?"}, True),
        ({"value": None}, None, True),
        ({}, None, False),
    ],
)
async def test_input_requested_preserves_content(
    mode: str, data: dict[str, Any], expected: Any, legacy: bool
) -> None:
    event = cast(
        Event,
        {
            "method": "input.requested",
            "params": {
                "namespace": ["subgraph"],
                "data": {"interrupt_id": "int-1", **data},
            },
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mode == "sync":
            stream = SyncThreadStream(
                http=MagicMock(), thread_id="thread-1", assistant_id="agent"
            )
            stream._apply_lifecycle_event(event)
            interrupts = stream.interrupts
            interrupted = stream.interrupted
        else:
            async_stream = AsyncThreadStream(
                http=MagicMock(), thread_id="thread-1", assistant_id="agent"
            )
            await async_stream._apply_lifecycle_event(event)
            interrupts = async_stream.interrupts
            interrupted = async_stream.interrupted

    assert interrupted
    assert interrupts == [
        {"interrupt_id": "int-1", "value": expected, "namespace": ["subgraph"]}
    ]
    if legacy:
        assert len(caught) == 1
        assert caught[0].category is DeprecationWarning
        assert "payload" in str(caught[0].message)
    else:
        assert not caught
