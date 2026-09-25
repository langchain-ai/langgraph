"""Queued protocol frames must retain the contents observed at dispatch."""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import GenericFakeChatModel

from langgraph.graph import END, START, MessagesState, StateGraph


class ReusingFieldsModel(GenericFakeChatModel):
    fragments: list[str]

    async def _astream_chat_model_events(self, messages, **kwargs):
        yield {"event": "message-start", "role": "ai", "message_id": "message"}
        yield {
            "event": "content-block-start",
            "index": 0,
            "content": {
                "type": "tool_call_chunk",
                "id": "call",
                "name": "save",
                "args": "",
            },
        }
        fields: dict[str, Any] = {
            "type": "tool_call_chunk",
            "id": "call",
            "name": "save",
            "args": "",
        }
        for fragment in self.fragments:
            fields["args"] += fragment
            # Native OpenAI-style producers reuse this cumulative fields object.
            yield {
                "event": "content-block-delta",
                "index": 0,
                "delta": {"type": "block-delta", "fields": fields},
            }
        yield {
            "event": "content-block-finish",
            "index": 0,
            "content": {
                "type": "tool_call",
                "id": "call",
                "name": "save",
                "args": json.loads(fields["args"]),
            },
        }
        yield {"event": "message-finish", "reason": "tool_calls"}


class ProtocolObserver(BaseCallbackHandler):
    run_inline = True

    def __init__(self):
        self.arguments = []

    def on_stream_event(self, event, **kwargs):
        if event.get("event") == "content-block-delta":
            self.arguments.append(event["delta"]["fields"]["args"])


def graph_for(fragments, observer):
    model = ReusingFieldsModel(messages=iter([]), fragments=fragments)

    async def call_model(state):
        return {"messages": [await model.ainvoke(state["messages"])]}

    return (
        StateGraph(MessagesState)
        .add_node("model", call_model)
        .add_edge(START, "model")
        .add_edge("model", END)
        .compile()
        .with_config(callbacks=[observer])
    )


async def capture(fragments):
    observer = ProtocolObserver()
    run = await graph_for(fragments, observer).astream_events(
        {"messages": [{"role": "user", "content": "save a value"}]}, version="v3"
    )
    events = [event async for event in run]
    arguments = [
        event["params"]["data"][0]["delta"]["fields"]["args"]
        for event in events
        if event["method"] == "messages"
        and event["params"]["data"][0].get("event") == "content-block-delta"
    ]
    return arguments, observer.arguments


class StreamFrameSnapshotsTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_reused_fields_retain_every_original_fragment(self):
        fragments = ['{"value":', '"a', "b", 'c"}']
        expected = ["".join(fragments[:i]) for i in range(1, len(fragments) + 1)]
        arguments, observed = await capture(fragments)
        # The public callback observes originals before the graph stream queue.
        self.assertEqual(observed, expected)
        self.assertEqual(arguments, expected)
        self.assertEqual(
            [
                value[len(previous) :]
                for previous, value in zip([""] + arguments, arguments)
            ],
            fragments,
        )

    async def test_concurrent_streams_keep_independent_snapshots(self):
        groups = [["{", '"first":', "1}"], ["{", '"second":', "2}"]]
        results = await asyncio.gather(*(capture(fragments) for fragments in groups))
        for fragments, (arguments, observed) in zip(groups, results):
            expected = ["".join(fragments[:i]) for i in range(1, len(fragments) + 1)]
            self.assertEqual(arguments, expected)
            self.assertEqual(observed, expected)
