import asyncio
import binascii
import concurrent.futures
import weakref
from collections.abc import Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    ExitStack,
)
from functools import partial
from typing import Any, Optional
from uuid import UUID

import orjson
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

import langgraph.scheduler.kafka.serde as serde
from langgraph.constants import CONFIG_KEY_DELEGATE, ERROR
from langgraph.errors import CheckpointNotLatest, GraphDelegate, TaskNotFound
from langgraph.pregel import Pregel
from langgraph.pregel.algo import checkpoint_null_version, prepare_single_task
from langgraph.pregel.executor import (
    AsyncBackgroundExecutor,
    BackgroundExecutor,
    Submit,
)
from langgraph.pregel.manager import AsyncChannelsManager, ChannelsManager
from langgraph.pregel.runner import PregelRunner
from langgraph.scheduler.kafka.retry import aretry, retry
from langgraph.scheduler.kafka.types import (
    AsyncConsumer,
    AsyncProducer,
    Consumer,
    ErrorMessage,
    MessageToExecutor,
    MessageToOrchestrator,
    Producer,
    Sendable,
    Topics,
)
from langgraph.types import LoopProtocol, PregelExecutableTask, RetryPolicy
from langgraph.utils.config import patch_configurable, recast_checkpoint_ns


class AsyncKafkaExecutor(AbstractAsyncContextManager):
    consumer: AsyncConsumer

    producer: AsyncProducer

    def __init__(
        self,
        graph: Pregel,
        topics: Topics,
        *,
        batch_max_n: int = 10,
        batch_max_ms: int = 1000,
        retry_policy: Optional[RetryPolicy] = None,
        consumer: Optional[AsyncConsumer] = None,
        producer: Optional[AsyncProducer] = None,
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        self.topics = topics
        self.stack = AsyncExitStack()
        self.kwargs = kwargs
        self.consumer = consumer
        self.producer = producer
        self.batch_max_n = batch_max_n
        self.batch_max_ms = batch_max_ms
        self.retry_policy = retry_policy

    async def __aenter__(self) -> Self:
        loop = asyncio.get_running_loop()
        self.subgraphs = {
            k: v async for k, v in self.graph.aget_subgraphs(recurse=True)
        }
        if self.consumer is None:
            from langgraph.scheduler.kafka.default_async import DefaultAsyncConsumer

            self.consumer = await self.stack.enter_async_context(
                DefaultAsyncConsumer(
                    self.topics.executor,
                    auto_offset_reset="earliest",
                    group_id="executor",
                    enable_auto_commit=False,
                    loop=loop,
                    **self.kwargs,
                )
            )
        if self.producer is None:
            from langgraph.scheduler.kafka.default_async import DefaultAsyncProducer

            self.producer = await self.stack.enter_async_context(
                DefaultAsyncProducer(
                    loop=loop,
                    **self.kwargs,
                )
            )
        return self

    async def __aexit__(self, *args: Any) -> None:
        return await self.stack.__aexit__(*args)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Sequence[MessageToExecutor]:
        # wait for next batch
        recs = await self.consumer.getmany(
            timeout_ms=self.batch_max_ms, max_records=self.batch_max_n
        )
        msgs: list[MessageToExecutor] = [
            serde.loads(msg.value) for msgs in recs.values() for msg in msgs
        ]
        # process batch
        await asyncio.gather(*(self.each(msg) for msg in msgs))
        # commit offsets
        await self.consumer.commit()
        # return message
        return msgs

    async def each(self, msg: MessageToExecutor) -> None:
        try:
            await aretry(self.retry_policy, self.attempt, msg)
        except CheckpointNotLatest:
            pass
        except GraphDelegate as exc:
            for arg in exc.args:
                fut = await self.producer.send(
                    self.topics.orchestrator,
                    value=serde.dumps(
                        MessageToOrchestrator(
                            config=arg["config"],
                            input=orjson.Fragment(
                                self.graph.checkpointer.serde.dumps(arg["input"])
                            ),
                            finally_send=[
                                Sendable(topic=self.topics.executor, value=msg)
                            ],
                        )
                    ),
                    # use thread_id, checkpoint_ns as partition key
                    key=serde.dumps(
                        (
                            arg["config"]["configurable"]["thread_id"],
                            arg["config"]["configurable"].get("checkpoint_ns"),
                        )
                    ),
                )
                await fut
        except Exception as exc:
            fut = await self.producer.send(
                self.topics.error,
                value=serde.dumps(
                    ErrorMessage(
                        topic=self.topics.executor,
                        msg=msg,
                        error=repr(exc),
                    )
                ),
            )
            await fut

    async def attempt(self, msg: MessageToExecutor) -> None:
        # find graph
        if checkpoint_ns := msg["config"]["configurable"].get("checkpoint_ns"):
            # remove task_ids from checkpoint_ns
            recast = recast_checkpoint_ns(checkpoint_ns)
            # find the subgraph with the matching name
            if recast in self.subgraphs:
                graph = self.subgraphs[recast]
            else:
                raise ValueError(f"Subgraph {recast} not found")
        else:
            graph = self.graph
        # process message
        saved = await self.graph.checkpointer.aget_tuple(
            patch_configurable(msg["config"], {"checkpoint_id": None})
        )
        if saved is None:
            raise RuntimeError("Checkpoint not found")
        if saved.checkpoint["id"] != msg["config"]["configurable"]["checkpoint_id"]:
            raise CheckpointNotLatest()
        async with (
            AsyncChannelsManager(
                graph.channels,
                saved.checkpoint,
                LoopProtocol(
                    config=msg["config"],
                    store=self.graph.store,
                    step=saved.metadata["step"] + 1,
                    stop=saved.metadata["step"] + 2,
                ),
            ) as (channels, managed),
            AsyncBackgroundExecutor(msg["config"]) as submit,
        ):
            if task := await asyncio.to_thread(
                prepare_single_task,
                msg["task"]["path"],
                msg["task"]["id"],
                checkpoint=saved.checkpoint,
                pending_writes=saved.pending_writes or [],
                processes=graph.nodes,
                channels=channels,
                managed=managed,
                config=patch_configurable(msg["config"], {CONFIG_KEY_DELEGATE: True}),
                step=saved.metadata["step"] + 1,
                for_execution=True,
                checkpointer=self.graph.checkpointer,
                store=self.graph.store,
                checkpoint_id_bytes=binascii.unhexlify(
                    saved.checkpoint["id"].replace("-", "")
                ),
                checkpoint_null_version=checkpoint_null_version(saved.checkpoint),
            ):
                # execute task, saving writes
                put_writes = partial(self._put_writes, submit, msg["config"])
                runner = PregelRunner(
                    submit=weakref.ref(submit),
                    put_writes=weakref.ref(put_writes),
                    schedule_task=weakref.WeakMethod(self._schedule_task),
                )
                async for _ in runner.atick([task], reraise=False):
                    pass
            else:
                # task was not found
                await self.graph.checkpointer.aput_writes(
                    msg["config"], [(ERROR, TaskNotFound())], str(UUID(int=0))
                )
        # notify orchestrator
        fut = await self.producer.send(
            self.topics.orchestrator,
            value=serde.dumps(
                MessageToOrchestrator(
                    input=None,
                    config=msg["config"],
                    finally_send=msg.get("finally_send"),
                )
            ),
            # use thread_id, checkpoint_ns as partition key
            key=serde.dumps(
                (
                    msg["config"]["configurable"]["thread_id"],
                    msg["config"]["configurable"].get("checkpoint_ns"),
                )
            ),
        )
        await fut

    def _schedule_task(
        self,
        task: PregelExecutableTask,
        idx: int,
    ) -> None:
        # will be scheduled by orchestrator when executor finishes
        pass

    def _put_writes(
        self,
        submit: Submit,
        config: RunnableConfig,
        task_id: str,
        writes: list[tuple[str, Any]],
    ) -> None:
        return submit(self.graph.checkpointer.aput_writes, config, writes, task_id)


class KafkaExecutor(AbstractContextManager):
    consumer: Consumer

    producer: Producer

    def __init__(
        self,
        graph: Pregel,
        topics: Topics,
        *,
        batch_max_n: int = 10,
        batch_max_ms: int = 1000,
        retry_policy: Optional[RetryPolicy] = None,
        consumer: Optional[Consumer] = None,
        producer: Optional[Producer] = None,
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        self.topics = topics
        self.stack = ExitStack()
        self.kwargs = kwargs
        self.consumer = consumer
        self.producer = producer
        self.batch_max_n = batch_max_n
        self.batch_max_ms = batch_max_ms
        self.retry_policy = retry_policy

    def __enter__(self) -> Self:
        self.subgraphs = dict(self.graph.get_subgraphs(recurse=True))
        self.submit = self.stack.enter_context(BackgroundExecutor({}))
        if self.consumer is None:
            from langgraph.scheduler.kafka.default_sync import DefaultConsumer

            self.consumer = self.stack.enter_context(
                DefaultConsumer(
                    self.topics.executor,
                    auto_offset_reset="earliest",
                    group_id="executor",
                    enable_auto_commit=False,
                    **self.kwargs,
                )
            )
        if self.producer is None:
            from langgraph.scheduler.kafka.default_sync import DefaultProducer

            self.producer = self.stack.enter_context(
                DefaultProducer(
                    **self.kwargs,
                )
            )
        return self

    def __exit__(self, *args: Any) -> None:
        return self.stack.__exit__(*args)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Sequence[MessageToExecutor]:
        # wait for next batch
        recs = self.consumer.getmany(
            timeout_ms=self.batch_max_ms, max_records=self.batch_max_n
        )
        msgs: list[MessageToExecutor] = [
            serde.loads(msg.value) for msgs in recs.values() for msg in msgs
        ]
        # process batch
        concurrent.futures.wait(self.submit(self.each, msg) for msg in msgs)
        # commit offsets
        self.consumer.commit()
        # return message
        return msgs

    def each(self, msg: MessageToExecutor) -> None:
        try:
            retry(self.retry_policy, self.attempt, msg)
        except CheckpointNotLatest:
            pass
        except GraphDelegate as exc:
            for arg in exc.args:
                fut = self.producer.send(
                    self.topics.orchestrator,
                    value=serde.dumps(
                        MessageToOrchestrator(
                            config=arg["config"],
                            input=orjson.Fragment(
                                self.graph.checkpointer.serde.dumps(arg["input"])
                            ),
                            finally_send=[
                                Sendable(topic=self.topics.executor, value=msg)
                            ],
                        )
                    ),
                    # use thread_id, checkpoint_ns as partition key
                    key=serde.dumps(
                        (
                            arg["config"]["configurable"]["thread_id"],
                            arg["config"]["configurable"].get("checkpoint_ns"),
                        )
                    ),
                )
                fut.result()
        except Exception as exc:
            fut = self.producer.send(
                self.topics.error,
                value=serde.dumps(
                    ErrorMessage(
                        topic=self.topics.executor,
                        msg=msg,
                        error=repr(exc),
                    )
                ),
            )
            fut.result()

    def attempt(self, msg: MessageToExecutor) -> None:
        # find graph
        if checkpoint_ns := msg["config"]["configurable"].get("checkpoint_ns"):
            # remove task_ids from checkpoint_ns
            recast = recast_checkpoint_ns(checkpoint_ns)
            # find the subgraph with the matching name
            if recast in self.subgraphs:
                graph = self.subgraphs[recast]
            else:
                raise ValueError(f"Subgraph {recast} not found")
        else:
            graph = self.graph
        # process message
        saved = self.graph.checkpointer.get_tuple(
            patch_configurable(msg["config"], {"checkpoint_id": None})
        )
        if saved is None:
            raise RuntimeError("Checkpoint not found")
        if saved.checkpoint["id"] != msg["config"]["configurable"]["checkpoint_id"]:
            raise CheckpointNotLatest()
        with (
            ChannelsManager(
                graph.channels,
                saved.checkpoint,
                LoopProtocol(
                    config=msg["config"],
                    store=self.graph.store,
                    step=saved.metadata["step"] + 1,
                    stop=saved.metadata["step"] + 2,
                ),
            ) as (channels, managed),
            BackgroundExecutor({}) as submit,
        ):
            if task := prepare_single_task(
                msg["task"]["path"],
                msg["task"]["id"],
                checkpoint=saved.checkpoint,
                pending_writes=saved.pending_writes or [],
                processes=graph.nodes,
                channels=channels,
                managed=managed,
                config=patch_configurable(msg["config"], {CONFIG_KEY_DELEGATE: True}),
                step=saved.metadata["step"] + 1,
                for_execution=True,
                checkpointer=self.graph.checkpointer,
                checkpoint_id_bytes=binascii.unhexlify(
                    saved.checkpoint["id"].replace("-", "")
                ),
                checkpoint_null_version=checkpoint_null_version(saved.checkpoint),
            ):
                # execute task, saving writes
                put_writes = partial(self._put_writes, submit, msg["config"])
                runner = PregelRunner(
                    submit=weakref.ref(submit),
                    put_writes=weakref.ref(put_writes),
                    schedule_task=weakref.WeakMethod(self._schedule_task),
                )
                for _ in runner.tick([task], reraise=False):
                    pass
            else:
                # task was not found
                self.graph.checkpointer.put_writes(
                    msg["config"], [(ERROR, TaskNotFound())], str(UUID(int=0))
                )
        # notify orchestrator
        fut = self.producer.send(
            self.topics.orchestrator,
            value=serde.dumps(
                MessageToOrchestrator(
                    input=None,
                    config=msg["config"],
                    finally_send=msg.get("finally_send"),
                )
            ),
            # use thread_id, checkpoint_ns as partition key
            key=serde.dumps(
                (
                    msg["config"]["configurable"]["thread_id"],
                    msg["config"]["configurable"].get("checkpoint_ns"),
                )
            ),
        )
        fut.result()

    def _schedule_task(
        self,
        task: PregelExecutableTask,
        idx: int,
    ) -> None:
        # will be scheduled by orchestrator when executor finishes
        pass

    def _put_writes(
        self,
        submit: Submit,
        config: RunnableConfig,
        task_id: str,
        writes: list[tuple[str, Any]],
    ) -> None:
        return submit(self.graph.checkpointer.put_writes, config, writes, task_id)
