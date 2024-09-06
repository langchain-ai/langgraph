import asyncio
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any, Self

import aiokafka
from langchain_core.runnables import ensure_config

import langgraph.scheduler.kafka.serde as serde
from langgraph.constants import CONFIG_KEY_DEDUPE_TASKS, SCHEDULED
from langgraph.pregel import Pregel
from langgraph.pregel.loop import INPUT_RESUMING, AsyncPregelLoop
from langgraph.scheduler.kafka.types import (
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
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        self.topics = topics
        self.stack = AsyncExitStack()
        self.kwargs = kwargs
        self.group_id = group_id
        self.batch_max_n = batch_max_n
        self.batch_max_ms = batch_max_ms

    async def __aenter__(self) -> Self:
        self.consumer = await self.stack.enter_async_context(
            aiokafka.AIOKafkaConsumer(
                self.topics.orchestrator,
                value_deserializer=serde.loads,
                auto_offset_reset="earliest",
                group_id=self.group_id,
                enable_auto_commit=False,
                **self.kwargs,
            )
        )
        self.producer = await self.stack.enter_async_context(
            aiokafka.AIOKafkaProducer(value_serializer=serde.dumps, **self.kwargs)
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stack.__aexit__(*args)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> list[MessageToOrchestrator]:
        # wait for next batch
        try:
            recs = await self.consumer.getmany(
                timeout_ms=self.batch_max_ms, max_records=self.batch_max_n
            )
            msgs: list[MessageToOrchestrator] = [
                msg.value for msgs in recs.values() for msg in msgs
            ]
        except aiokafka.ConsumerStoppedError:
            raise StopAsyncIteration from None
        # process batch
        await asyncio.gather(*(self.each(msg) for msg in msgs))
        # commit offsets
        await self.consumer.commit()
        # return message
        return msgs

    async def each(self, msg: MessageToOrchestrator) -> None:
        # process message
        async with AsyncPregelLoop(
            msg["input"],
            config=ensure_config(msg["config"]),
            stream=None,
            store=self.graph.store,
            checkpointer=self.graph.checkpointer,
            nodes=self.graph.nodes,
            specs=self.graph.channels,
            output_keys=self.graph.output_channels,
            stream_keys=self.graph.stream_channels,
        ) as loop:
            if loop.tick(input_keys=self.graph.input_channels):
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
                                        },
                                    ),
                                    task=ExecutorTask(
                                        id=task.id,
                                        path=task.path,
                                        step=loop.step,
                                        resuming=loop.input is INPUT_RESUMING,
                                    ),
                                ),
                            )
                            for task in new_tasks
                        )
                    )
                    # wait for messages to be sent
                    await asyncio.gather(*futures)
                    # mark as scheduled
                    for task in new_tasks:
                        loop.put_writes(task.id, [(SCHEDULED, None)])
