from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.pregel.remote import RemoteGraph
from langgraph.types import All, StreamMode


class _FakeRemoteGraph(RemoteGraph):
    def __init__(self) -> None:
        super().__init__("agent", url="http://unused")
        self.last_stream_modes: list[StreamMode] | None = None

    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v2"],
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        assert version == "v2"
        assert subgraphs is True
        self.last_stream_modes = (
            [stream_mode] if isinstance(stream_mode, str) else list(stream_mode or [])
        )
        yield {
            "type": "values",
            "ns": (),
            "data": {"value": input["value"] + "A"},
            "interrupts": (),
        }
        yield {
            "type": "values",
            "ns": (),
            "data": {"value": input["value"] + "AB"},
            "interrupts": (),
        }

    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v2"],
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        assert version == "v2"
        assert subgraphs is True
        self.last_stream_modes = (
            [stream_mode] if isinstance(stream_mode, str) else list(stream_mode or [])
        )
        yield {
            "type": "values",
            "ns": (),
            "data": {"value": input["value"] + "A"},
            "interrupts": (),
        }
        yield {
            "type": "values",
            "ns": (),
            "data": {"value": input["value"] + "AB"},
            "interrupts": (),
        }


def test_remote_stream_v2_values_and_output() -> None:
    remote = _FakeRemoteGraph()
    run = remote.stream_v2({"value": "x"})

    assert list(run.values) == [{"value": "xA"}, {"value": "xAB"}]
    assert "values" in (remote.last_stream_modes or [])


def test_remote_stream_v2_output_drains_remote_events() -> None:
    remote = _FakeRemoteGraph()
    run = remote.stream_v2({"value": "x"})

    assert run.output == {"value": "xAB"}


def test_remote_stream_v2_raw_events() -> None:
    remote = _FakeRemoteGraph()
    run = remote.stream_v2({"value": "x"})

    events = list(run)
    assert [event["method"] for event in events] == ["values", "values"]
    assert [event["seq"] for event in events] == [1, 2]


@pytest.mark.anyio
async def test_remote_astream_v2_values_and_output() -> None:
    remote = _FakeRemoteGraph()
    run = await remote.astream_v2({"value": "x"})

    assert [item async for item in run.values] == [
        {"value": "xA"},
        {"value": "xAB"},
    ]
    assert "values" in (remote.last_stream_modes or [])


@pytest.mark.anyio
async def test_remote_astream_v2_output_drains_remote_events() -> None:
    remote = _FakeRemoteGraph()
    run = await remote.astream_v2({"value": "x"})

    assert await run.output() == {"value": "xAB"}

