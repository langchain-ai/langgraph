"""Async client for managing recurrent runs (cron jobs) in LangGraph."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk.schema import (
    All,
    Config,
    Context,
    Cron,
    CronSelectField,
    CronSortBy,
    Durability,
    Input,
    OnCompletionBehavior,
    QueryParamTypes,
    Run,
    SortOrder,
    StreamMode,
)


class CronClient:
    """Client for managing recurrent runs (cron jobs) in LangGraph.

    A run is a single invocation of an assistant with optional input, config, and context.
    This client allows scheduling recurring runs to occur automatically.

    ???+ example "Example Usage"

        ```python
        client = get_client(url="http://localhost:2024"))
        cron_job = await client.crons.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_456",
            schedule="0 9 * * *",
            input={"message": "Daily update"}
        )
        ```

    !!! note "Feature Availability"

        The crons client functionality is not supported on all licenses.
        Please check the relevant license documentation for the most up-to-date
        details on feature availability.
    """

    def __init__(self, http_client: HttpClient) -> None:
        self.http = http_client

    async def create_for_thread(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        schedule: str,
        input: Input | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint_during: bool | None = None,  # deprecated
        interrupt_before: All | list[str] | None = None,
        interrupt_after: All | list[str] | None = None,
        webhook: str | None = None,
        multitask_strategy: str | None = None,
        end_time: datetime | None = None,
        enabled: bool | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        stream_subgraphs: bool | None = None,
        stream_resumable: bool | None = None,
        durability: Durability | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Run:
        """Create a cron job for a thread.

        Args:
            thread_id: the thread ID to run the cron job on.
            assistant_id: The assistant ID or graph name to use for the cron job.
                If using graph name, will default to first assistant created from that graph.
            schedule: The cron schedule to execute this job on.
                Schedules are interpreted in UTC.
            input: The input to the graph.
            metadata: Metadata to assign to the cron job runs.
            config: The configuration for the assistant.
            context: Static context to add to the assistant.
                !!! version-added "Added in version 0.6.0"
            checkpoint_during: (deprecated) Whether to checkpoint during the run (or only at the end/interruption).
            interrupt_before: Nodes to interrupt immediately before they get executed.

            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.

            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            end_time: The time to stop running the cron job. If not provided, the cron job will run indefinitely.
            enabled: Whether the cron job is enabled or not.
            stream_mode: The stream mode(s) to use.
            stream_subgraphs: Whether to stream output from subgraphs.
            stream_resumable: Whether to persist the stream chunks in order to resume the stream later.
            durability: Durability level for the run. Must be one of 'sync', 'async', or 'exit'.
                "async" means checkpoints are persisted async while next graph step executes, replaces checkpoint_during=True
                "sync" means checkpoints are persisted sync after graph step executes, replaces checkpoint_during=False
                "exit" means checkpoints are only persisted when the run exits, does not save intermediate steps
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            The cron run.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            cron_run = await client.crons.create_for_thread(
                thread_id="my-thread-id",
                assistant_id="agent",
                schedule="27 15 * * *",
                input={"messages": [{"role": "user", "content": "hello!"}]},
                metadata={"name":"my_run"},
                context={"model_name": "openai"},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt",
                enabled=True,
            )
            ```
        """
        if checkpoint_during is not None:
            warnings.warn(
                "`checkpoint_during` is deprecated and will be removed in a future version. Use `durability` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        payload = {
            "schedule": schedule,
            "input": input,
            "config": config,
            "metadata": metadata,
            "context": context,
            "assistant_id": assistant_id,
            "checkpoint_during": checkpoint_during,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "end_time": end_time.isoformat() if end_time else None,
            "enabled": enabled,
            "stream_mode": stream_mode,
            "stream_subgraphs": stream_subgraphs,
            "stream_resumable": stream_resumable,
            "durability": durability,
        }
        if multitask_strategy:
            payload["multitask_strategy"] = multitask_strategy
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self.http.post(
            f"/threads/{thread_id}/runs/crons",
            json=payload,
            headers=headers,
            params=params,
        )

    async def create(
        self,
        assistant_id: str,
        *,
        schedule: str,
        input: Input | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint_during: bool | None = None,  # deprecated
        interrupt_before: All | list[str] | None = None,
        interrupt_after: All | list[str] | None = None,
        webhook: str | None = None,
        on_run_completed: OnCompletionBehavior | None = None,
        multitask_strategy: str | None = None,
        end_time: datetime | None = None,
        enabled: bool | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        stream_subgraphs: bool | None = None,
        stream_resumable: bool | None = None,
        durability: Durability | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Run:
        """Create a cron run.

        Args:
            assistant_id: The assistant ID or graph name to use for the cron job.
                If using graph name, will default to first assistant created from that graph.
            schedule: The cron schedule to execute this job on.
                Schedules are interpreted in UTC.
            input: The input to the graph.
            metadata: Metadata to assign to the cron job runs.
            config: The configuration for the assistant.
            context: Static context to add to the assistant.
                !!! version-added "Added in version 0.6.0"
            checkpoint_during: (deprecated) Whether to checkpoint during the run (or only at the end/interruption).
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            webhook: Webhook to call after LangGraph API call is done.
            on_run_completed: What to do with the thread after the run completes.
                Must be one of 'delete' (default) or 'keep'. 'delete' removes the thread
                after execution. 'keep' creates a new thread for each execution but does not
                clean them up. Clients are responsible for cleaning up kept threads.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            end_time: The time to stop running the cron job. If not provided, the cron job will run indefinitely.
            enabled: Whether the cron job is enabled or not.
            stream_mode: The stream mode(s) to use.
            stream_subgraphs: Whether to stream output from subgraphs.
            stream_resumable: Whether to persist the stream chunks in order to resume the stream later.
            durability: Durability level for the run. Must be one of 'sync', 'async', or 'exit'.
                "async" means checkpoints are persisted async while next graph step executes, replaces checkpoint_during=True
                "sync" means checkpoints are persisted sync after graph step executes, replaces checkpoint_during=False
                "exit" means checkpoints are only persisted when the run exits, does not save intermediate steps
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            The cron run.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            cron_run = client.crons.create(
                assistant_id="agent",
                schedule="27 15 * * *",
                input={"messages": [{"role": "user", "content": "hello!"}]},
                metadata={"name":"my_run"},
                context={"model_name": "openai"},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt",
                enabled=True,
            )
            ```

        """
        if checkpoint_during is not None:
            warnings.warn(
                "`checkpoint_during` is deprecated and will be removed in a future version. Use `durability` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        payload = {
            "schedule": schedule,
            "input": input,
            "config": config,
            "metadata": metadata,
            "context": context,
            "assistant_id": assistant_id,
            "checkpoint_during": checkpoint_during,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "on_run_completed": on_run_completed,
            "end_time": end_time.isoformat() if end_time else None,
            "enabled": enabled,
            "stream_mode": stream_mode,
            "stream_subgraphs": stream_subgraphs,
            "stream_resumable": stream_resumable,
            "durability": durability,
        }
        if multitask_strategy:
            payload["multitask_strategy"] = multitask_strategy
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self.http.post(
            "/runs/crons", json=payload, headers=headers, params=params
        )

    async def delete(
        self,
        cron_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        """Delete a cron.

        Args:
            cron_id: The cron ID to delete.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            await client.crons.delete(
                cron_id="cron_to_delete"
            )
            ```

        """
        await self.http.delete(f"/runs/crons/{cron_id}", headers=headers, params=params)

    async def update(
        self,
        cron_id: str,
        *,
        schedule: str | None = None,
        end_time: datetime | None = None,
        input: Input | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        webhook: str | None = None,
        interrupt_before: All | list[str] | None = None,
        interrupt_after: All | list[str] | None = None,
        on_run_completed: OnCompletionBehavior | None = None,
        enabled: bool | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        stream_subgraphs: bool | None = None,
        stream_resumable: bool | None = None,
        durability: Durability | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Cron:
        """Update a cron job by ID.

        Args:
            cron_id: The cron ID to update.
            schedule: The cron schedule to execute this job on.
                Schedules are interpreted in UTC.
            end_time: The end date to stop running the cron.
            input: The input to the graph.
            metadata: Metadata to assign to the cron job runs.
            config: The configuration for the assistant.
            context: Static context added to the assistant.
            webhook: Webhook to call after LangGraph API call is done.
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to interrupt immediately after they get executed.
            on_run_completed: What to do with the thread after the run completes.
                Must be one of 'delete' or 'keep'. 'delete' removes the thread
                after execution. 'keep' creates a new thread for each execution but does not
                clean them up.
            enabled: Enable or disable the cron job.
            stream_mode: The stream mode(s) to use.
            stream_subgraphs: Whether to stream output from subgraphs.
            stream_resumable: Whether to persist the stream chunks in order to resume the stream later.
            durability: Durability level for the run. Must be one of 'sync', 'async', or 'exit'.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            The updated cron job.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            updated_cron = await client.crons.update(
                cron_id="1ef3cefa-4c09-6926-96d0-3dc97fd5e39b",
                schedule="0 10 * * *",
                enabled=False,
            )
            ```

        """
        payload = {
            "schedule": schedule,
            "end_time": end_time.isoformat() if end_time else None,
            "input": input,
            "metadata": metadata,
            "config": config,
            "context": context,
            "webhook": webhook,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "on_run_completed": on_run_completed,
            "enabled": enabled,
            "stream_mode": stream_mode,
            "stream_subgraphs": stream_subgraphs,
            "stream_resumable": stream_resumable,
            "durability": durability,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self.http.patch(
            f"/runs/crons/{cron_id}",
            json=payload,
            headers=headers,
            params=params,
        )

    async def search(
        self,
        *,
        assistant_id: str | None = None,
        thread_id: str | None = None,
        enabled: bool | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: CronSortBy | None = None,
        sort_order: SortOrder | None = None,
        select: list[CronSelectField] | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> list[Cron]:
        """Get a list of cron jobs.

        Args:
            assistant_id: The assistant ID or graph name to search for.
            thread_id: the thread ID to search for.
            enabled: The enabled status to search for.
            limit: The maximum number of results to return.
            offset: The number of results to skip.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            The list of cron jobs returned by the search,

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            cron_jobs = await client.crons.search(
                assistant_id="my_assistant_id",
                thread_id="my_thread_id",
                enabled=True,
                limit=5,
                offset=5,
            )
            print(cron_jobs)
            ```
            ```shell

            ----------------------------------------------------------

            [
                {
                    'cron_id': '1ef3cefa-4c09-6926-96d0-3dc97fd5e39b',
                    'assistant_id': 'my_assistant_id',
                    'thread_id': 'my_thread_id',
                    'user_id': None,
                    'payload':
                        {
                            'input': {'start_time': ''},
                            'schedule': '4 * * * *',
                            'assistant_id': 'my_assistant_id'
                        },
                    'schedule': '4 * * * *',
                    'next_run_date': '2024-07-25T17:04:00+00:00',
                    'end_time': None,
                    'created_at': '2024-07-08T06:02:23.073257+00:00',
                    'updated_at': '2024-07-08T06:02:23.073257+00:00'
                }
            ]
            ```

        """
        payload = {
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "enabled": enabled,
            "limit": limit,
            "offset": offset,
        }
        if sort_by:
            payload["sort_by"] = sort_by
        if sort_order:
            payload["sort_order"] = sort_order
        if select:
            payload["select"] = select
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self.http.post(
            "/runs/crons/search", json=payload, headers=headers, params=params
        )

    async def count(
        self,
        *,
        assistant_id: str | None = None,
        thread_id: str | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> int:
        """Count cron jobs matching filters.

        Args:
            assistant_id: Assistant ID to filter by.
            thread_id: Thread ID to filter by.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            int: Number of crons matching the criteria.
        """
        payload: dict[str, Any] = {}
        if assistant_id:
            payload["assistant_id"] = assistant_id
        if thread_id:
            payload["thread_id"] = thread_id
        return await self.http.post(
            "/runs/crons/count", json=payload, headers=headers, params=params
        )
