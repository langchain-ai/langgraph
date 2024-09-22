import asyncio
import concurrent.futures
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    ExitStack,
)
from typing import Any, Optional

from langchain_core.runnables import ensure_config
from typing_extensions import Self

import langgraph.scheduler.kafka.serde as serde
from langgraph.constants import (
    CONFIG_KEY_DEDUPE_TASKS,
    CONFIG_KEY_ENSURE_LATEST,
    INTERRUPT,
    NS_END,
    NS_SEP,
    SCHEDULED,
)
from langgraph.errors import CheckpointNotLatest, GraphInterrupt
from langgraph.pregel import Pregel
from langgraph.pregel.executor import BackgroundExecutor, Submit
from langgraph.pregel.loop import AsyncPregelLoop, SyncPregelLoop
from langgraph.scheduler.kafka.retry import aretry, retry
from langgraph.scheduler.kafka.types import (
    AsyncConsumer,
    AsyncProducer,
    Consumer,
    ErrorMessage,
    ExecutorTask,
    MessageToExecutor,
    MessageToOrchestrator,
    Producer,
    Topics,
)
from langgraph.types import RetryPolicy
from langgraph.utils.config import patch_configurable


class AsyncKafkaOrchestrator(AbstractAsyncContextManager):
    consumer: AsyncConsumer

    producer: AsyncProducer

    def __init__(
        self,
        graph: Pregel,
        topics: Topics,
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
        self.subgraphs = {
            k: v async for k, v in self.graph.aget_subgraphs(recurse=True)
        }
        if self.consumer is None:
            from langgraph.scheduler.kafka.default_async import DefaultAsyncConsumer

            self.consumer = await self.stack.enter_async_context(
                DefaultAsyncConsumer(
                    self.topics.orchestrator,
                    auto_offset_reset="earliest",
                    group_id="orchestrator",
                    enable_auto_commit=False,
                    **self.kwargs,
                )
            )
        if self.producer is None:
            from langgraph.scheduler.kafka.default_async import DefaultAsyncProducer

            self.producer = await self.stack.enter_async_context(
                DefaultAsyncProducer(
                    **self.kwargs,
                )
            )
        return self

    async def __aexit__(self, *args: Any) -> None:
        return await self.stack.__aexit__(*args)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> list[MessageToOrchestrator]:
        # wait for next batch
        recs = await self.consumer.getmany(
            timeout_ms=self.batch_max_ms, max_records=self.batch_max_n
        )
        # dedupe messages, eg. if multiple nodes finish around same time
        uniq = set(msg.value for msgs in recs.values() for msg in msgs)
        msgs: list[MessageToOrchestrator] = [serde.loads(msg) for msg in uniq]
        # process batch
        await asyncio.gather(*(self.each(msg) for msg in msgs))
        # commit offsets
        await self.consumer.commit()
        # return message
        return msgs

    async def each(self, msg: MessageToOrchestrator) -> None:
        try:
            await aretry(self.retry_policy, self.attempt, msg)
        except CheckpointNotLatest:
            pass
        except GraphInterrupt:
            pass
        except Exception as exc:
            fut = await self.producer.send(
                self.topics.error,
                value=serde.dumps(
                    ErrorMessage(
                        topic=self.topics.orchestrator,
                        msg=msg,
                        error=repr(exc),
                    )
                ),
            )
            await fut

    async def attempt(self, msg: MessageToOrchestrator) -> None:
        # find graph
        if checkpoint_ns := msg["config"]["configurable"].get("checkpoint_ns"):
            # remove task_ids from checkpoint_ns
            recast_checkpoint_ns = NS_SEP.join(
                part.split(NS_END)[0] for part in checkpoint_ns.split(NS_SEP)
            )
            # find the subgraph with the matching name
            if recast_checkpoint_ns in self.subgraphs:
                graph = self.subgraphs[recast_checkpoint_ns]
            else:
                raise ValueError(f"Subgraph {recast_checkpoint_ns} not found")
        else:
            graph = self.graph
        # process message
        async with AsyncPregelLoop(
            msg["input"],
            config=ensure_config(msg["config"]),
            stream=None,
            store=self.graph.store,
            checkpointer=self.graph.checkpointer,
            nodes=graph.nodes,
            specs=graph.channels,
            output_keys=graph.output_channels,
            stream_keys=graph.stream_channels,
            check_subgraphs=False,
        ) as loop:
            if loop.tick(
                input_keys=graph.input_channels,
                interrupt_after=graph.interrupt_after_nodes,
                interrupt_before=graph.interrupt_before_nodes,
            ):
                # wait for checkpoint to be saved
                if hasattr(loop, "_put_checkpoint_fut"):
                    await loop._put_checkpoint_fut
                # schedule any new tasks
                if new_tasks := [t for t in loop.tasks.values() if not t.scheduled]:
                    # send messages to executor
                    futures = await asyncio.gather(
                        *(
                            self.producer.send(
                                self.topics.executor,
                                value=serde.dumps(
                                    MessageToExecutor(
                                        config=patch_configurable(
                                            loop.config,
                                            {
                                                **loop.checkpoint_config[
                                                    "configurable"
                                                ],
                                                CONFIG_KEY_DEDUPE_TASKS: True,
                                                CONFIG_KEY_ENSURE_LATEST: True,
                                            },
                                        ),
                                        task=ExecutorTask(id=task.id, path=task.path),
                                        finally_send=msg.get("finally_send"),
                                    )
                                ),
                            )
                            for task in new_tasks
                        )
                    )
                    # wait for messages to be sent
                    await asyncio.gather(*futures)
                    # mark as scheduled
                    for task in new_tasks:
                        loop.put_writes(
                            task.id,
                            [
                                (
                                    SCHEDULED,
                                    max(
                                        loop.checkpoint["versions_seen"]
                                        .get(INTERRUPT, {})
                                        .values(),
                                        default=None,
                                    ),
                                )
                            ],
                        )
            elif loop.status == "done" and msg.get("finally_send"):
                # send any finally_send messages
                futs = await asyncio.gather(
                    *(
                        self.producer.send(
                            m["topic"],
                            value=serde.dumps(m["value"]) if m.get("value") else None,
                            key=serde.dumps(m["key"]) if m.get("key") else None,
                        )
                        for m in msg["finally_send"]
                    )
                )
                # wait for messages to be sent
                await asyncio.gather(*futs)


class KafkaOrchestrator(AbstractContextManager):
    consumer: Consumer

    producer: Producer

    submit: Submit

    def __init__(
        self,
        graph: Pregel,
        topics: Topics,
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
                    self.topics.orchestrator,
                    auto_offset_reset="earliest",
                    group_id="orchestrator",
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

    def __next__(self) -> list[MessageToOrchestrator]:
        # wait for next batch
        recs = self.consumer.getmany(
            timeout_ms=self.batch_max_ms, max_records=self.batch_max_n
        )
        # dedupe messages, eg. if multiple nodes finish around same time
        uniq = set(msg.value for msgs in recs.values() for msg in msgs)
        msgs: list[MessageToOrchestrator] = [serde.loads(msg) for msg in uniq]
        # process batch
        concurrent.futures.wait(self.submit(self.each, msg) for msg in msgs)
        # commit offsets
        self.consumer.commit()
        # return message
        return msgs

    def each(self, msg: MessageToOrchestrator) -> None:
        try:
            retry(self.retry_policy, self.attempt, msg)
        except CheckpointNotLatest:
            pass
        except GraphInterrupt:
            pass
        except Exception as exc:
            fut = self.producer.send(
                self.topics.error,
                value=serde.dumps(
                    ErrorMessage(
                        topic=self.topics.orchestrator,
                        msg=msg,
                        error=repr(exc),
                    )
                ),
            )
            fut.result()

    def attempt(self, msg: MessageToOrchestrator) -> None:
        # find graph
        if checkpoint_ns := msg["config"]["configurable"].get("checkpoint_ns"):
            # remove task_ids from checkpoint_ns
            recast_checkpoint_ns = NS_SEP.join(
                part.split(NS_END)[0] for part in checkpoint_ns.split(NS_SEP)
            )
            # find the subgraph with the matching name
            if recast_checkpoint_ns in self.subgraphs:
                graph = self.subgraphs[recast_checkpoint_ns]
            else:
                raise ValueError(f"Subgraph {recast_checkpoint_ns} not found")
        else:
            graph = self.graph
        # process message
        with SyncPregelLoop(
            msg["input"],
            config=ensure_config(msg["config"]),
            stream=None,
            store=self.graph.store,
            checkpointer=self.graph.checkpointer,
            nodes=graph.nodes,
            specs=graph.channels,
            output_keys=graph.output_channels,
            stream_keys=graph.stream_channels,
        ) as loop:
            if loop.tick(
                input_keys=graph.input_channels,
                interrupt_after=graph.interrupt_after_nodes,
                interrupt_before=graph.interrupt_before_nodes,
            ):
                # wait for checkpoint to be saved
                if hasattr(loop, "_put_checkpoint_fut"):
                    loop._put_checkpoint_fut.result()
                # schedule any new tasks
                if new_tasks := [t for t in loop.tasks.values() if not t.scheduled]:
                    # send messages to executor
                    futures = [
                        self.producer.send(
                            self.topics.executor,
                            value=serde.dumps(
                                MessageToExecutor(
                                    config=patch_configurable(
                                        loop.config,
                                        {
                                            **loop.checkpoint_config["configurable"],
                                            CONFIG_KEY_DEDUPE_TASKS: True,
                                            CONFIG_KEY_ENSURE_LATEST: True,
                                        },
                                    ),
                                    task=ExecutorTask(id=task.id, path=task.path),
                                    finally_send=msg.get("finally_send"),
                                )
                            ),
                        )
                        for task in new_tasks
                    ]
                    # wait for messages to be sent
                    concurrent.futures.wait(futures)
                    # mark as scheduled
                    for task in new_tasks:
                        loop.put_writes(
                            task.id,
                            [
                                (
                                    SCHEDULED,
                                    max(
                                        loop.checkpoint["versions_seen"]
                                        .get(INTERRUPT, {})
                                        .values(),
                                        default=None,
                                    ),
                                )
                            ],
                        )
            elif loop.status == "done" and msg.get("finally_send"):
                # schedule any finally_send msgs
                futs = [
                    self.producer.send(
                        m["topic"],
                        value=serde.dumps(m["value"]) if m.get("value") else None,
                        key=serde.dumps(m["key"]) if m.get("key") else None,
                    )
                    for m in msg["finally_send"]
                ]
                # wait for messages to be sent
                concurrent.futures.wait(futs)
