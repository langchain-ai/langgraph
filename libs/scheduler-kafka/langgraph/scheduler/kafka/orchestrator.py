import asyncio
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any, Optional

import aiokafka
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
from langgraph.pregel.loop import AsyncPregelLoop
from langgraph.pregel.types import RetryPolicy
from langgraph.scheduler.kafka.retry import aretry
from langgraph.scheduler.kafka.types import (
    ErrorMessage,
    ExecutorTask,
    MessageToExecutor,
    MessageToOrchestrator,
    Topics,
)
from langgraph.utils.config import patch_configurable


class KafkaOrchestrator(AbstractAsyncContextManager):
    def __init__(
        self,
        graph: Pregel,
        topics: Topics,
        group_id: str = "orchestrator",
        batch_max_n: int = 10,
        batch_max_ms: int = 1000,
        retry_policy: Optional[RetryPolicy] = None,
        consumer_kwargs: Optional[dict[str, Any]] = None,
        producer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        self.topics = topics
        self.stack = AsyncExitStack()
        self.kwargs = kwargs
        self.consumer_kwargs = consumer_kwargs or {}
        self.producer_kwargs = producer_kwargs or {}
        self.group_id = group_id
        self.batch_max_n = batch_max_n
        self.batch_max_ms = batch_max_ms
        self.retry_policy = retry_policy

    async def __aenter__(self) -> Self:
        self.consumer = await self.stack.enter_async_context(
            aiokafka.AIOKafkaConsumer(
                self.topics.orchestrator,
                auto_offset_reset="earliest",
                group_id=self.group_id,
                enable_auto_commit=False,
                **self.kwargs,
                **self.consumer_kwargs,
            )
        )
        self.producer = await self.stack.enter_async_context(
            aiokafka.AIOKafkaProducer(
                value_serializer=serde.dumps,
                **self.kwargs,
                **self.producer_kwargs,
            )
        )
        self.subgraphs = {
            k: v async for k, v in self.graph.aget_subgraphs(recurse=True)
        }
        return self

    async def __aexit__(self, *args: Any) -> None:
        return await self.stack.__aexit__(*args)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> list[MessageToOrchestrator]:
        # wait for next batch
        try:
            recs = await self.consumer.getmany(
                timeout_ms=self.batch_max_ms, max_records=self.batch_max_n
            )
            # dedupe messages, eg. if multiple nodes finish around same time
            uniq = set(msg.value for msgs in recs.values() for msg in msgs)
            msgs: list[MessageToOrchestrator] = [serde.loads(msg) for msg in uniq]
        except aiokafka.ConsumerStoppedError:
            raise StopAsyncIteration from None
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
            await self.producer.send_and_wait(
                self.topics.error,
                value=ErrorMessage(
                    topic=self.topics.orchestrator,
                    msg=msg,
                    error=repr(exc),
                ),
            )

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
                    futures: list[asyncio.Future] = await asyncio.gather(
                        *(
                            self.producer.send(
                                self.topics.executor,
                                value=MessageToExecutor(
                                    config=patch_configurable(
                                        loop.config,
                                        {
                                            **loop.checkpoint_config["configurable"],
                                            CONFIG_KEY_DEDUPE_TASKS: True,
                                            CONFIG_KEY_ENSURE_LATEST: True,
                                        },
                                    ),
                                    task=ExecutorTask(id=task.id, path=task.path),
                                    finally_executor=msg.get("finally_executor"),
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
            elif loop.status == "done" and msg.get("finally_executor"):
                # schedule any finally_executor tasks
                futs = await asyncio.gather(
                    *(
                        self.producer.send(self.topics.executor, value=m)
                        for m in msg["finally_executor"]
                    )
                )
                # wait for messages to be sent
                await asyncio.gather(*futs)
