"""Synchronous client for managing runs in LangGraph."""

from __future__ import annotations

import builtins
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, overload

import httpx

from langgraph_sdk._shared.utilities import _get_run_metadata_from_response
from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk.schema import (
    All,
    BulkCancelRunsStatus,
    CancelAction,
    Checkpoint,
    Command,
    Config,
    Context,
    DisconnectMode,
    Durability,
    IfNotExists,
    Input,
    MultitaskStrategy,
    OnCompletionBehavior,
    QueryParamTypes,
    Run,
    RunCreate,
    RunCreateMetadata,
    RunSelectField,
    RunStatus,
    StreamMode,
    StreamPart,
)


class SyncRunsClient:
    """Synchronous client for managing runs in LangGraph.

    This class provides methods to create, retrieve, and manage runs, which represent
    individual executions of graphs.

    ???+ example "Example"

        ```python
        client = get_sync_client(url="http://localhost:2024")
        run = client.runs.create(thread_id="thread_123", assistant_id="asst_456")
        ```
    """

    def __init__(self, http: SyncHttpClient) -> None:
        self.http = http

    @overload
    def stream(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Input | None = None,
        command: Command | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] = "values",
        stream_subgraphs: bool = False,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint: Checkpoint | None = None,
        checkpoint_id: str | None = None,
        checkpoint_during: bool | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        feedback_keys: Sequence[str] | None = None,
        on_disconnect: DisconnectMode | None = None,
        webhook: str | None = None,
        multitask_strategy: MultitaskStrategy | None = None,
        if_not_exists: IfNotExists | None = None,
        after_seconds: int | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        on_run_created: Callable[[RunCreateMetadata], None] | None = None,
    ) -> Iterator[StreamPart]: ...

    @overload
    def stream(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Input | None = None,
        command: Command | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] = "values",
        stream_subgraphs: bool = False,
        stream_resumable: bool = False,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint_during: bool | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        feedback_keys: Sequence[str] | None = None,
        on_disconnect: DisconnectMode | None = None,
        on_completion: OnCompletionBehavior | None = None,
        if_not_exists: IfNotExists | None = None,
        webhook: str | None = None,
        after_seconds: int | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        on_run_created: Callable[[RunCreateMetadata], None] | None = None,
    ) -> Iterator[StreamPart]: ...

    def stream(
        self,
        thread_id: str | None,
        assistant_id: str,
        *,
        input: Input | None = None,
        command: Command | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] = "values",
        stream_subgraphs: bool = False,
        stream_resumable: bool = False,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint: Checkpoint | None = None,
        checkpoint_id: str | None = None,
        checkpoint_during: bool | None = None,  # deprecated
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        feedback_keys: Sequence[str] | None = None,
        on_disconnect: DisconnectMode | None = None,
        on_completion: OnCompletionBehavior | None = None,
        webhook: str | None = None,
        multitask_strategy: MultitaskStrategy | None = None,
        if_not_exists: IfNotExists | None = None,
        after_seconds: int | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        on_run_created: Callable[[RunCreateMetadata], None] | None = None,
        durability: Durability | None = None,
    ) -> Iterator[StreamPart]:
        """Create a run and stream the results.

        Args:
            thread_id: the thread ID to assign to the thread.
                If `None` will create a stateless run.
            assistant_id: The assistant ID or graph name to stream from.
                If using graph name, will default to first assistant created from that graph.
            input: The input to the graph.
            command: The command to execute.
            stream_mode: The stream mode(s) to use.
            stream_subgraphs: Whether to stream output from subgraphs.
            stream_resumable: Whether the stream is considered resumable.
                If true, the stream can be resumed and replayed in its entirety even after disconnection.
            metadata: Metadata to assign to the run.
            config: The configuration for the assistant.
            context: Static context to add to the assistant.
                !!! version-added "Added in version 0.6.0"
            checkpoint: The checkpoint to resume from.
            checkpoint_during: (deprecated) Whether to checkpoint during the run (or only at the end/interruption).
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            feedback_keys: Feedback keys to assign to run.
            on_disconnect: The disconnect mode to use.
                Must be one of 'cancel' or 'continue'.
            on_completion: Whether to delete or keep the thread created for a stateless run.
                Must be one of 'delete' or 'keep'.
            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            if_not_exists: How to handle missing thread. Defaults to 'reject'.
                Must be either 'reject' (raise error if missing), or 'create' (create new thread).
            after_seconds: The number of seconds to wait before starting the run.
                Use to schedule future runs.
            headers: Optional custom headers to include with the request.
            on_run_created: Optional callback to call when a run is created.
            durability: The durability to use for the run. Values are "sync", "async", or "exit".
                "async" means checkpoints are persisted async while next graph step executes, replaces checkpoint_during=True
                "sync" means checkpoints are persisted sync after graph step executes, replaces checkpoint_during=False
                "exit" means checkpoints are only persisted when the run exits, does not save intermediate steps


        Returns:
            Iterator of stream results.

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            async for chunk in client.runs.stream(
                thread_id=None,
                assistant_id="agent",
                input={"messages": [{"role": "user", "content": "how are you?"}]},
                stream_mode=["values","debug"],
                metadata={"name":"my_run"},
                context={"model_name": "anthropic"},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                feedback_keys=["my_feedback_key_1","my_feedback_key_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            ):
                print(chunk)
            ```
            ```shell
            ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            StreamPart(event='metadata', data={'run_id': '1ef4a9b8-d7da-679a-a45a-872054341df2'})
            StreamPart(event='values', data={'messages': [{'content': 'how are you?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'fe0a5778-cfe9-42ee-b807-0adaa1873c10', 'example': False}]})
            StreamPart(event='values', data={'messages': [{'content': 'how are you?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'fe0a5778-cfe9-42ee-b807-0adaa1873c10', 'example': False}, {'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-159b782c-b679-4830-83c6-cef87798fe8b', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]})
            StreamPart(event='end', data=None)
            ```
        """
        if checkpoint_during is not None:
            warnings.warn(
                "`checkpoint_during` is deprecated and will be removed in a future version. Use `durability` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        payload = {
            "input": input,
            "command": (
                {k: v for k, v in command.items() if v is not None} if command else None
            ),
            "config": config,
            "context": context,
            "metadata": metadata,
            "stream_mode": stream_mode,
            "stream_subgraphs": stream_subgraphs,
            "stream_resumable": stream_resumable,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "feedback_keys": feedback_keys,
            "webhook": webhook,
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "checkpoint_during": checkpoint_during,
            "multitask_strategy": multitask_strategy,
            "if_not_exists": if_not_exists,
            "on_disconnect": on_disconnect,
            "on_completion": on_completion,
            "after_seconds": after_seconds,
            "durability": durability,
        }
        endpoint = (
            f"/threads/{thread_id}/runs/stream"
            if thread_id is not None
            else "/runs/stream"
        )

        def on_response(res: httpx.Response):
            """Callback function to handle the response."""
            if on_run_created and (metadata := _get_run_metadata_from_response(res)):
                on_run_created(metadata)

        return self.http.stream(
            endpoint,
            "POST",
            json={k: v for k, v in payload.items() if v is not None},
            params=params,
            headers=headers,
            on_response=on_response if on_run_created else None,
        )

    @overload
    def create(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Input | None = None,
        command: Command | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] = "values",
        stream_subgraphs: bool = False,
        stream_resumable: bool = False,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint_during: bool | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        webhook: str | None = None,
        on_completion: OnCompletionBehavior | None = None,
        if_not_exists: IfNotExists | None = None,
        after_seconds: int | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        on_run_created: Callable[[RunCreateMetadata], None] | None = None,
    ) -> Run: ...

    @overload
    def create(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Input | None = None,
        command: Command | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] = "values",
        stream_subgraphs: bool = False,
        stream_resumable: bool = False,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint: Checkpoint | None = None,
        checkpoint_id: str | None = None,
        checkpoint_during: bool | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        webhook: str | None = None,
        multitask_strategy: MultitaskStrategy | None = None,
        if_not_exists: IfNotExists | None = None,
        after_seconds: int | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        on_run_created: Callable[[RunCreateMetadata], None] | None = None,
    ) -> Run: ...

    def create(
        self,
        thread_id: str | None,
        assistant_id: str,
        *,
        input: Input | None = None,
        command: Command | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] = "values",
        stream_subgraphs: bool = False,
        stream_resumable: bool = False,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint: Checkpoint | None = None,
        checkpoint_id: str | None = None,
        checkpoint_during: bool | None = None,  # deprecated
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        webhook: str | None = None,
        multitask_strategy: MultitaskStrategy | None = None,
        if_not_exists: IfNotExists | None = None,
        on_completion: OnCompletionBehavior | None = None,
        after_seconds: int | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        on_run_created: Callable[[RunCreateMetadata], None] | None = None,
        durability: Durability | None = None,
    ) -> Run:
        """Create a background run.

        Args:
            thread_id: the thread ID to assign to the thread.
                If `None` will create a stateless run.
            assistant_id: The assistant ID or graph name to stream from.
                If using graph name, will default to first assistant created from that graph.
            input: The input to the graph.
            command: The command to execute.
            stream_mode: The stream mode(s) to use.
            stream_subgraphs: Whether to stream output from subgraphs.
            stream_resumable: Whether the stream is considered resumable.
                If true, the stream can be resumed and replayed in its entirety even after disconnection.
            metadata: Metadata to assign to the run.
            config: The configuration for the assistant.
            context: Static context to add to the assistant.
                !!! version-added "Added in version 0.6.0"
            checkpoint: The checkpoint to resume from.
            checkpoint_during: (deprecated) Whether to checkpoint during the run (or only at the end/interruption).
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            on_completion: Whether to delete or keep the thread created for a stateless run.
                Must be one of 'delete' or 'keep'.
            if_not_exists: How to handle missing thread. Defaults to 'reject'.
                Must be either 'reject' (raise error if missing), or 'create' (create new thread).
            after_seconds: The number of seconds to wait before starting the run.
                Use to schedule future runs.
            headers: Optional custom headers to include with the request.
            on_run_created: Optional callback to call when a run is created.
            durability: The durability to use for the run. Values are "sync", "async", or "exit".
                "async" means checkpoints are persisted async while next graph step executes, replaces checkpoint_during=True
                "sync" means checkpoints are persisted sync after graph step executes, replaces checkpoint_during=False
                "exit" means checkpoints are only persisted when the run exits, does not save intermediate steps

        Returns:
            The created background `Run`.

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            background_run = client.runs.create(
                thread_id="my_thread_id",
                assistant_id="my_assistant_id",
                input={"messages": [{"role": "user", "content": "hello!"}]},
                metadata={"name":"my_run"},
                context={"model_name": "openai"},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )
            print(background_run)
            ```

            ```shell
            --------------------------------------------------------------------------------

            {
                'run_id': 'my_run_id',
                'thread_id': 'my_thread_id',
                'assistant_id': 'my_assistant_id',
                'created_at': '2024-07-25T15:35:42.598503+00:00',
                'updated_at': '2024-07-25T15:35:42.598503+00:00',
                'metadata': {},
                'status': 'pending',
                'kwargs':
                    {
                        'input':
                            {
                                'messages': [
                                    {
                                        'role': 'user',
                                        'content': 'how are you?'
                                    }
                                ]
                            },
                        'config':
                            {
                                'metadata':
                                    {
                                        'created_by': 'system'
                                    },
                                'configurable':
                                    {
                                        'run_id': 'my_run_id',
                                        'user_id': None,
                                        'graph_id': 'agent',
                                        'thread_id': 'my_thread_id',
                                        'checkpoint_id': None,
                                        'assistant_id': 'my_assistant_id'
                                    }
                            },
                        'context':
                            {
                                'model_name': 'openai'
                            },
                        'webhook': "https://my.fake.webhook.com",
                        'temporary': False,
                        'stream_mode': ['values'],
                        'feedback_keys': None,
                        'interrupt_after': ["node_to_stop_after_1","node_to_stop_after_2"],
                        'interrupt_before': ["node_to_stop_before_1","node_to_stop_before_2"]
                    },
                'multitask_strategy': 'interrupt'
            }
            ```
        """
        if checkpoint_during is not None:
            warnings.warn(
                "`checkpoint_during` is deprecated and will be removed in a future version. Use `durability` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        payload = {
            "input": input,
            "command": (
                {k: v for k, v in command.items() if v is not None} if command else None
            ),
            "stream_mode": stream_mode,
            "stream_subgraphs": stream_subgraphs,
            "stream_resumable": stream_resumable,
            "config": config,
            "context": context,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "checkpoint_during": checkpoint_during,
            "multitask_strategy": multitask_strategy,
            "if_not_exists": if_not_exists,
            "on_completion": on_completion,
            "after_seconds": after_seconds,
            "durability": durability,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        def on_response(res: httpx.Response):
            """Callback function to handle the response."""
            if on_run_created and (metadata := _get_run_metadata_from_response(res)):
                on_run_created(metadata)

        return self.http.post(
            f"/threads/{thread_id}/runs" if thread_id else "/runs",
            json=payload,
            params=params,
            headers=headers,
            on_response=on_response if on_run_created else None,
        )

    def create_batch(
        self,
        payloads: builtins.list[RunCreate],
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> builtins.list[Run]:
        """Create a batch of stateless background runs."""

        def filter_payload(payload: RunCreate):
            return {k: v for k, v in payload.items() if v is not None}

        filtered = [filter_payload(payload) for payload in payloads]
        return self.http.post(
            "/runs/batch", json=filtered, headers=headers, params=params
        )

    @overload
    def wait(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Input | None = None,
        command: Command | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint: Checkpoint | None = None,
        checkpoint_id: str | None = None,
        checkpoint_during: bool | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        webhook: str | None = None,
        on_disconnect: DisconnectMode | None = None,
        multitask_strategy: MultitaskStrategy | None = None,
        if_not_exists: IfNotExists | None = None,
        after_seconds: int | None = None,
        raise_error: bool = True,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        on_run_created: Callable[[RunCreateMetadata], None] | None = None,
    ) -> builtins.list[dict] | dict[str, Any]: ...

    @overload
    def wait(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Input | None = None,
        command: Command | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint_during: bool | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        webhook: str | None = None,
        on_disconnect: DisconnectMode | None = None,
        on_completion: OnCompletionBehavior | None = None,
        if_not_exists: IfNotExists | None = None,
        after_seconds: int | None = None,
        raise_error: bool = True,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        on_run_created: Callable[[RunCreateMetadata], None] | None = None,
    ) -> builtins.list[dict] | dict[str, Any]: ...

    def wait(
        self,
        thread_id: str | None,
        assistant_id: str,
        *,
        input: Input | None = None,
        command: Command | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: Config | None = None,
        context: Context | None = None,
        checkpoint_during: bool | None = None,  # deprecated
        checkpoint: Checkpoint | None = None,
        checkpoint_id: str | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        webhook: str | None = None,
        on_disconnect: DisconnectMode | None = None,
        on_completion: OnCompletionBehavior | None = None,
        multitask_strategy: MultitaskStrategy | None = None,
        if_not_exists: IfNotExists | None = None,
        after_seconds: int | None = None,
        raise_error: bool = True,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        on_run_created: Callable[[RunCreateMetadata], None] | None = None,
        durability: Durability | None = None,
    ) -> builtins.list[dict] | dict[str, Any]:
        """Create a run, wait until it finishes and return the final state.

        Args:
            thread_id: the thread ID to create the run on.
                If `None` will create a stateless run.
            assistant_id: The assistant ID or graph name to run.
                If using graph name, will default to first assistant created from that graph.
            input: The input to the graph.
            command: The command to execute.
            metadata: Metadata to assign to the run.
            config: The configuration for the assistant.
            context: Static context to add to the assistant.
                !!! version-added "Added in version 0.6.0"
            checkpoint: The checkpoint to resume from.
            checkpoint_during: (deprecated) Whether to checkpoint during the run (or only at the end/interruption).
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            webhook: Webhook to call after LangGraph API call is done.
            on_disconnect: The disconnect mode to use.
                Must be one of 'cancel' or 'continue'.
            on_completion: Whether to delete or keep the thread created for a stateless run.
                Must be one of 'delete' or 'keep'.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            if_not_exists: How to handle missing thread. Defaults to 'reject'.
                Must be either 'reject' (raise error if missing), or 'create' (create new thread).
            after_seconds: The number of seconds to wait before starting the run.
                Use to schedule future runs.
            raise_error: Whether to raise an error if the run fails.
            headers: Optional custom headers to include with the request.
            on_run_created: Optional callback to call when a run is created.
            durability: The durability to use for the run. Values are "sync", "async", or "exit".
                "async" means checkpoints are persisted async while next graph step executes, replaces checkpoint_during=True
                "sync" means checkpoints are persisted sync after graph step executes, replaces checkpoint_during=False
                "exit" means checkpoints are only persisted when the run exits, does not save intermediate steps

        Returns:
            The output of the `Run`.

        ???+ example "Example Usage"

            ```python

            final_state_of_run = client.runs.wait(
                thread_id=None,
                assistant_id="agent",
                input={"messages": [{"role": "user", "content": "how are you?"}]},
                metadata={"name":"my_run"},
                context={"model_name": "anthropic"},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )
            print(final_state_of_run)
            ```

            ```shell

            -------------------------------------------------------------------------------------------------------------------------------------------

            {
                'messages': [
                    {
                        'content': 'how are you?',
                        'additional_kwargs': {},
                        'response_metadata': {},
                        'type': 'human',
                        'name': None,
                        'id': 'f51a862c-62fe-4866-863b-b0863e8ad78a',
                        'example': False
                    },
                    {
                        'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.",
                        'additional_kwargs': {},
                        'response_metadata': {},
                        'type': 'ai',
                        'name': None,
                        'id': 'run-bf1cd3c6-768f-4c16-b62d-ba6f17ad8b36',
                        'example': False,
                        'tool_calls': [],
                        'invalid_tool_calls': [],
                        'usage_metadata': None
                    }
                ]
            }
            ```

        """
        if checkpoint_during is not None:
            warnings.warn(
                "`checkpoint_during` is deprecated and will be removed in a future version. Use `durability` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        payload = {
            "input": input,
            "command": (
                {k: v for k, v in command.items() if v is not None} if command else None
            ),
            "config": config,
            "context": context,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
            "if_not_exists": if_not_exists,
            "on_disconnect": on_disconnect,
            "checkpoint_during": checkpoint_during,
            "on_completion": on_completion,
            "after_seconds": after_seconds,
            "raise_error": raise_error,
            "durability": durability,
        }

        def on_response(res: httpx.Response):
            """Callback function to handle the response."""
            if on_run_created and (metadata := _get_run_metadata_from_response(res)):
                on_run_created(metadata)

        endpoint = (
            f"/threads/{thread_id}/runs/wait" if thread_id is not None else "/runs/wait"
        )
        return self.http.request_reconnect(
            endpoint,
            "POST",
            json={k: v for k, v in payload.items() if v is not None},
            params=params,
            headers=headers,
            on_response=on_response if on_run_created else None,
        )

    def list(
        self,
        thread_id: str,
        *,
        limit: int = 10,
        offset: int = 0,
        status: RunStatus | None = None,
        select: builtins.list[RunSelectField] | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> builtins.list[Run]:
        """List runs.

        Args:
            thread_id: The thread ID to list runs for.
            limit: The maximum number of results to return.
            offset: The number of results to skip.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            The runs for the thread.

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            client.runs.list(
                thread_id="thread_id",
                limit=5,
                offset=5,
            )
            ```

        """
        query_params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            query_params["status"] = status
        if select:
            query_params["select"] = select
        if params:
            query_params.update(params)
        return self.http.get(
            f"/threads/{thread_id}/runs", params=query_params, headers=headers
        )

    def get(
        self,
        thread_id: str,
        run_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Run:
        """Get a run.

        Args:
            thread_id: The thread ID to get.
            run_id: The run ID to get.
            headers: Optional custom headers to include with the request.

        Returns:
            `Run` object.

        ???+ example "Example Usage"

            ```python

            run = client.runs.get(
                thread_id="thread_id_to_delete",
                run_id="run_id_to_delete",
            )
            ```
        """

        return self.http.get(
            f"/threads/{thread_id}/runs/{run_id}", headers=headers, params=params
        )

    def cancel(
        self,
        thread_id: str,
        run_id: str,
        *,
        wait: bool = False,
        action: CancelAction = "interrupt",
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        """Get a run.

        Args:
            thread_id: The thread ID to cancel.
            run_id: The run ID to cancel.
            wait: Whether to wait until run has completed.
            action: Action to take when cancelling the run. Possible values
                are `interrupt` or `rollback`. Default is `interrupt`.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            client.runs.cancel(
                thread_id="thread_id_to_cancel",
                run_id="run_id_to_cancel",
                wait=True,
                action="interrupt"
            )
            ```

        """
        query_params = {
            "wait": 1 if wait else 0,
            "action": action,
        }
        if params:
            query_params.update(params)
        if wait:
            return self.http.request_reconnect(
                f"/threads/{thread_id}/runs/{run_id}/cancel",
                "POST",
                json=None,
                params=query_params,
                headers=headers,
            )
        return self.http.post(
            f"/threads/{thread_id}/runs/{run_id}/cancel",
            json=None,
            params=query_params,
            headers=headers,
        )

    def cancel_many(
        self,
        *,
        thread_id: str | None = None,
        run_ids: Sequence[str] | None = None,
        status: BulkCancelRunsStatus | None = None,
        action: CancelAction = "interrupt",
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        """Cancel one or more runs.

        Can cancel runs by thread ID and run IDs, or by status filter.

        Args:
            thread_id: The ID of the thread containing runs to cancel.
            run_ids: List of run IDs to cancel.
            status: Filter runs by status to cancel. Must be one of
                `"pending"`, `"running"`, or `"all"`.
            action: Action to take when cancelling the run. Possible values
                are `"interrupt"` or `"rollback"`. Default is `"interrupt"`.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            # Cancel all pending runs
            client.runs.cancel_many(status="pending")
            # Cancel specific runs on a thread
            client.runs.cancel_many(
                thread_id="my_thread_id",
                run_ids=["run_1", "run_2"],
                action="rollback",
            )
            ```

        """
        payload: dict[str, Any] = {}
        if thread_id:
            payload["thread_id"] = thread_id
        if run_ids:
            payload["run_ids"] = run_ids
        if status:
            payload["status"] = status
        query_params: dict[str, Any] = {"action": action}
        if params:
            query_params.update(params)
        self.http.post(
            "/runs/cancel",
            json=payload,
            headers=headers,
            params=query_params,
        )

    def join(
        self,
        thread_id: str,
        run_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> dict:
        """Block until a run is done. Returns the final state of the thread.

        Args:
            thread_id: The thread ID to join.
            run_id: The run ID to join.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            client.runs.join(
                thread_id="thread_id_to_join",
                run_id="run_id_to_join"
            )
            ```

        """
        return self.http.request_reconnect(
            f"/threads/{thread_id}/runs/{run_id}/join",
            "GET",
            headers=headers,
            params=params,
        )

    def join_stream(
        self,
        thread_id: str,
        run_id: str,
        *,
        cancel_on_disconnect: bool = False,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
        last_event_id: str | None = None,
    ) -> Iterator[StreamPart]:
        """Stream output from a run in real-time, until the run is done.
        Output is not buffered, so any output produced before this call will
        not be received here.

        Args:
            thread_id: The thread ID to join.
            run_id: The run ID to join.
            stream_mode: The stream mode(s) to use. Must be a subset of the stream modes passed
                when creating the run. Background runs default to having the union of all
                stream modes.
            cancel_on_disconnect: Whether to cancel the run when the stream is disconnected.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.
            last_event_id: The last event ID to use for the stream.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            client.runs.join_stream(
                thread_id="thread_id_to_join",
                run_id="run_id_to_join",
                stream_mode=["values", "debug"]
            )
            ```

        """
        query_params = {
            "stream_mode": stream_mode,
            "cancel_on_disconnect": cancel_on_disconnect,
        }
        if params:
            query_params.update(params)
        return self.http.stream(
            f"/threads/{thread_id}/runs/{run_id}/stream",
            "GET",
            params=query_params,
            headers={
                **({"Last-Event-ID": last_event_id} if last_event_id else {}),
                **(headers or {}),
            }
            or None,
        )

    def delete(
        self,
        thread_id: str,
        run_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        """Delete a run.

        Args:
            thread_id: The thread ID to delete.
            run_id: The run ID to delete.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            client.runs.delete(
                thread_id="thread_id_to_delete",
                run_id="run_id_to_delete"
            )
            ```

        """
        self.http.delete(
            f"/threads/{thread_id}/runs/{run_id}", headers=headers, params=params
        )
