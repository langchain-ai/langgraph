"""Regression tests for path-segment encoding of caller-supplied identifiers.

Covers GHSA-w39p-vh2g-g8g5: identifier values interpolated into request paths
are encoded so the resulting request addresses the resource the SDK method
indicates, even if the identifier contains characters with special meaning in
URL paths.
"""

from __future__ import annotations

import httpx
import pytest

from langgraph_sdk._shared.utilities import _quote_path_param
from langgraph_sdk.client import (
    AssistantsClient,
    CronClient,
    HttpClient,
    RunsClient,
    SyncAssistantsClient,
    SyncCronClient,
    SyncHttpClient,
    SyncRunsClient,
    SyncThreadsClient,
    ThreadsClient,
)


class TestQuotePathParam:
    """Unit tests for the encoding helper itself."""

    def test_uuid_round_trips_unchanged(self) -> None:
        uuid_value = "550e8400-e29b-41d4-a716-446655440000"
        assert _quote_path_param(uuid_value) == uuid_value

    def test_simple_opaque_id_round_trips_unchanged(self) -> None:
        assert _quote_path_param("thread_123") == "thread_123"
        assert _quote_path_param("asst_abc") == "asst_abc"

    def test_slash_is_encoded(self) -> None:
        assert _quote_path_param("foo/bar") == "foo%2Fbar"

    def test_bare_dot_segments_are_encoded(self) -> None:
        # All-dot strings are encoded to make them opaque to HTTP stacks that
        # collapse "./.." path segments client-side.
        assert _quote_path_param(".") == "%2E"
        assert _quote_path_param("..") == "%2E%2E"
        assert _quote_path_param("...") == "%2E%2E%2E"
        # Mixed values that happen to contain dots are not affected.
        assert _quote_path_param("agent.v1") == "agent.v1"
        # Subsequent ``/`` characters are encoded regardless.
        assert _quote_path_param("../bar") == "..%2Fbar"

    def test_full_pivot_payload_is_encoded(self) -> None:
        # A caller-supplied identifier that, if interpolated raw, would route
        # the request to a different resource type.
        payload = "../assistants/abc-123"
        encoded = _quote_path_param(payload)
        assert encoded == "..%2Fassistants%2Fabc-123"
        assert "/" not in encoded

    def test_non_string_values_are_coerced_to_str(self) -> None:
        import uuid

        uid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        assert _quote_path_param(uid) == str(uid)
        assert _quote_path_param(42) == "42"

    def test_none_value_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must not be None"):
            _quote_path_param(None)

    def test_bytes_value_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must not be bytes"):
            _quote_path_param(b"bytes")
        with pytest.raises(TypeError, match="must not be bytes"):
            _quote_path_param(bytearray(b"bytes"))


def _wire_path(request: httpx.Request) -> str:
    """Return the path as it goes on the wire (preserves percent-encoding)."""
    return request.url.raw_path.decode("ascii")


@pytest.mark.asyncio
class TestAsyncPathEncoding:
    """Async-client tests that verify the encoded path actually lands on the wire.

    Note: ``request.url.path`` is the percent-decoded display form. The bytes
    that actually go on the wire are in ``request.url.raw_path``; that is what
    the server's router sees and what these tests inspect.
    """

    async def test_threads_get_with_pivot_payload_stays_on_threads(self) -> None:
        captured: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={"thread_id": "anything"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            threads_client = ThreadsClient(HttpClient(client))
            await threads_client.get("../assistants/abc-123")

        assert len(captured) == 1
        wire = captured[0]
        # The identifier is encoded so the wire path stays inside `/threads/...`.
        # The encoded segment must not contain literal slashes that could let
        # the server re-route to a different resource type.
        assert wire.startswith("/threads/")
        segment = wire[len("/threads/") :]
        assert "/" not in segment
        assert "%2F" in segment
        assert segment == "..%2Fassistants%2Fabc-123"

    async def test_threads_update_with_pivot_payload_stays_on_threads(self) -> None:
        captured: list[tuple[str, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append((request.method, _wire_path(request)))
            return httpx.Response(200, json={"thread_id": "anything"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            threads_client = ThreadsClient(HttpClient(client))
            await threads_client.update("../assistants/abc-123", metadata={"x": 1})

        assert len(captured) == 1
        method, wire = captured[0]
        assert method == "PATCH"
        assert wire.startswith("/threads/")
        segment = wire[len("/threads/") :]
        assert "/" not in segment

    async def test_threads_delete_with_pivot_payload_stays_on_threads(self) -> None:
        captured: list[tuple[str, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append((request.method, _wire_path(request)))
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            threads_client = ThreadsClient(HttpClient(client))
            await threads_client.delete("../runs/crons/some-cron-id")

        assert len(captured) == 1
        method, wire = captured[0]
        assert method == "DELETE"
        assert wire.startswith("/threads/")
        segment = wire[len("/threads/") :]
        assert "/" not in segment

    async def test_assistants_get_with_pivot_payload_stays_on_assistants(
        self,
    ) -> None:
        captured: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={"assistant_id": "anything"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            assistants_client = AssistantsClient(HttpClient(client))
            await assistants_client.get("../threads/abc-123")

        assert len(captured) == 1
        wire = captured[0]
        assert wire.startswith("/assistants/")
        segment = wire[len("/assistants/") :]
        assert "/" not in segment

    async def test_runs_delete_double_id_pivot_stays_on_threads_runs(self) -> None:
        captured: list[tuple[str, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append((request.method, _wire_path(request)))
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs_client = RunsClient(HttpClient(client))
            # Both identifier values supplied as path-traversal payloads.
            await runs_client.delete("..", "../runs/crons/cron-id")

        assert len(captured) == 1
        method, wire = captured[0]
        assert method == "DELETE"
        # The path should match `/threads/{quoted_thread}/runs/{quoted_run}`
        # exactly. Neither segment should contain literal slashes.
        assert wire.startswith("/threads/")
        assert "/runs/crons/" not in wire
        parts = wire.split("/")
        # Expected shape: ['', 'threads', '<encoded ..>', 'runs', '<encoded ..>']
        assert len(parts) == 5
        assert parts[1] == "threads"
        assert parts[3] == "runs"
        # Encoded thread_id and run_id are between literal slashes.
        assert parts[2] == "%2E%2E"
        assert parts[4] == "..%2Fruns%2Fcrons%2Fcron-id"

    async def test_crons_delete_with_pivot_payload_stays_on_crons(self) -> None:
        captured: list[tuple[str, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append((request.method, _wire_path(request)))
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            crons_client = CronClient(HttpClient(client))
            await crons_client.delete("../../assistants/abc-123")

        assert len(captured) == 1
        method, wire = captured[0]
        assert method == "DELETE"
        assert wire.startswith("/runs/crons/")
        segment = wire[len("/runs/crons/") :]
        assert "/" not in segment

    async def test_threads_get_state_with_pivot_checkpoint_id_stays_on_state(
        self,
    ) -> None:
        captured: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            threads_client = ThreadsClient(HttpClient(client))
            await threads_client.get_state(thread_id="tid-1", checkpoint_id="../runs")

        assert len(captured) == 1
        wire = captured[0]
        # Wire path must stay on `/threads/{tid}/state/...`, not pivot to
        # `/threads/tid-1/runs`.
        assert wire.startswith("/threads/tid-1/state/")
        # Strip query string before checking the checkpoint segment.
        path_only = wire.split("?", 1)[0]
        segment = path_only[len("/threads/tid-1/state/") :]
        assert "/" not in segment
        assert segment == "..%2Fruns"

    async def test_assistants_get_subgraphs_with_pivot_namespace_stays_on_subgraphs(
        self,
    ) -> None:
        captured: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            assistants_client = AssistantsClient(HttpClient(client))
            await assistants_client.get_subgraphs("aid-1", namespace="../foo")

        assert len(captured) == 1
        wire = captured[0]
        # Wire path must stay on `/assistants/{aid}/subgraphs/...`.
        assert wire.startswith("/assistants/aid-1/subgraphs/")
        # Strip query string before checking the namespace segment.
        path_only = wire.split("?", 1)[0]
        segment = path_only[len("/assistants/aid-1/subgraphs/") :]
        assert "/" not in segment
        assert segment == "..%2Ffoo"

    async def test_bare_double_dot_thread_id_survives_to_wire(self) -> None:
        """The all-dot encoding branch must survive httpx's relative-path collapse."""
        captured: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={"thread_id": "anything"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            threads_client = ThreadsClient(HttpClient(client))
            await threads_client.get("..")

        assert len(captured) == 1
        # The all-dot identifier is fully percent-encoded so httpx does NOT
        # collapse it client-side as a relative-path traversal.
        assert captured[0].endswith("/threads/%2E%2E")

    async def test_bare_single_dot_thread_id_survives_to_wire(self) -> None:
        captured: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={"thread_id": "anything"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            threads_client = ThreadsClient(HttpClient(client))
            await threads_client.get(".")

        assert len(captured) == 1
        assert captured[0].endswith("/threads/%2E")

    async def test_uuid_identifier_lands_on_intended_path(self) -> None:
        """Legitimate UUID identifiers round-trip without encoding artifacts."""
        captured: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={"thread_id": "anything"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            threads_client = ThreadsClient(HttpClient(client))
            await threads_client.get("550e8400-e29b-41d4-a716-446655440000")

        assert captured == ["/threads/550e8400-e29b-41d4-a716-446655440000"]


class TestSyncPathEncoding:
    """Sync-client tests that mirror the async coverage on a representative subset."""

    def test_threads_get_with_pivot_payload_stays_on_threads(self) -> None:
        captured: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={"thread_id": "anything"})

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            threads_client = SyncThreadsClient(SyncHttpClient(client))
            threads_client.get("../assistants/abc-123")

        assert len(captured) == 1
        wire = captured[0]
        assert wire.startswith("/threads/")
        segment = wire[len("/threads/") :]
        assert "/" not in segment
        assert segment == "..%2Fassistants%2Fabc-123"

    def test_assistants_get_with_pivot_payload_stays_on_assistants(self) -> None:
        captured: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={"assistant_id": "anything"})

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            assistants_client = SyncAssistantsClient(SyncHttpClient(client))
            assistants_client.get("../threads/abc-123")

        assert len(captured) == 1
        wire = captured[0]
        assert wire.startswith("/assistants/")
        segment = wire[len("/assistants/") :]
        assert "/" not in segment

    def test_runs_delete_double_id_pivot_stays_on_threads_runs(self) -> None:
        captured: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append((request.method, _wire_path(request)))
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs_client = SyncRunsClient(SyncHttpClient(client))
            runs_client.delete("..", "../runs/crons/cron-id")

        assert len(captured) == 1
        method, wire = captured[0]
        assert method == "DELETE"
        assert wire.startswith("/threads/")
        assert "/runs/crons/" not in wire
        parts = wire.split("/")
        assert len(parts) == 5
        assert parts[1] == "threads"
        assert parts[3] == "runs"
        assert parts[2] == "%2E%2E"
        assert parts[4] == "..%2Fruns%2Fcrons%2Fcron-id"

    def test_crons_delete_with_pivot_payload_stays_on_crons(self) -> None:
        captured: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append((request.method, _wire_path(request)))
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            crons_client = SyncCronClient(SyncHttpClient(client))
            crons_client.delete("../../assistants/abc-123")

        assert len(captured) == 1
        method, wire = captured[0]
        assert method == "DELETE"
        assert wire.startswith("/runs/crons/")
        segment = wire[len("/runs/crons/") :]
        assert "/" not in segment

    def test_uuid_identifier_lands_on_intended_path(self) -> None:
        captured: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_wire_path(request))
            return httpx.Response(200, json={"thread_id": "anything"})

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            threads_client = SyncThreadsClient(SyncHttpClient(client))
            threads_client.get("550e8400-e29b-41d4-a716-446655440000")

        assert captured == ["/threads/550e8400-e29b-41d4-a716-446655440000"]
