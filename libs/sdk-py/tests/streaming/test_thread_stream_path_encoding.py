"""Thread stream REST helpers must percent-encode `thread_id` and `assistant_id`
so a value containing reserved characters or dot-segments stays an opaque path
segment instead of being normalized into a different path by the HTTP stack.

Each test drives a public entry point (`threads.stream(...)` followed by
`thread.output` or `thread.agent.get_tree()`) with an identifier containing
`..`, `/`, and `#`, and asserts the request path keeps it as one segment.
"""

from __future__ import annotations

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.threads import SyncThreadsClient

# Interpolated raw, httpx would collapse the `..` segments and treat `#` as a
# fragment, producing `/assistants/a/graph` and `/other/path`.
THREAD_ID = "../assistants/a/graph#"
ASSISTANT_ID = "../../other/path#"
ENCODED_STATE_PATH = "/threads/..%2Fassistants%2Fa%2Fgraph%23/state"
ENCODED_GRAPH_PATH = "/assistants/..%2F..%2Fother%2Fpath%23/graph"

TERMINAL_STATE = {"values": {"ok": True}, "next": [], "tasks": []}


def _recording_handler(gets: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            path = request.url.raw_path.decode("ascii").split("?", 1)[0]
            gets.append(path)
            return httpx.Response(200, json=TERMINAL_STATE)
        # Lifecycle watcher subscription: an empty event stream.
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b""
        )

    return handler


async def test_async_output_encodes_thread_id():
    gets: list[str] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_recording_handler(gets)),
        base_url="https://example.com",
    ) as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id=THREAD_ID, assistant_id="agent") as thread:
            assert await thread.output == {"ok": True}

    assert gets == [ENCODED_STATE_PATH]


async def test_async_agent_get_tree_encodes_assistant_id():
    gets: list[str] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_recording_handler(gets)),
        base_url="https://example.com",
    ) as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id=ASSISTANT_ID) as thread:
            await thread.agent.get_tree()

    assert gets == [ENCODED_GRAPH_PATH]


def test_sync_output_encodes_thread_id():
    gets: list[str] = []
    with httpx.Client(
        transport=httpx.MockTransport(_recording_handler(gets)),
        base_url="https://example.com",
    ) as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id=THREAD_ID, assistant_id="agent") as thread:
            assert thread.output == {"ok": True}

    assert gets == [ENCODED_STATE_PATH]


def test_sync_agent_get_tree_encodes_assistant_id():
    gets: list[str] = []
    with httpx.Client(
        transport=httpx.MockTransport(_recording_handler(gets)),
        base_url="https://example.com",
    ) as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id=ASSISTANT_ID) as thread:
            thread.agent.get_tree()

    assert gets == [ENCODED_GRAPH_PATH]
