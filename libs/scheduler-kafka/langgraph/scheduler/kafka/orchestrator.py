from contextlib import AbstractAsyncContextManager
from typing import Any

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
    def __init__(self, graph: Pregel, topics: Topics, **kwargs: Any) -> None:
        self.graph = graph
        self.topics = topics
        self.consumer = aiokafka.AIOKafkaConsumer(
            topics.orchestrator, value_deserializer=serde.loads, **kwargs
        )
        self.producer = aiokafka.AIOKafkaProducer(
            value_serializer=serde.dumps, **kwargs
        )

    async def __aenter__(self) -> "KafkaOrchestrator":
        await self.consumer.start()
        await self.producer.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.consumer.stop()
        await self.producer.stop()

    def __aiter__(self) -> "KafkaOrchestrator":
        return self

    async def __anext__(self) -> Any:
        # wait for next message
        try:
            rec = await self.consumer.getone()
            msg: MessageToOrchestrator = rec.value
        except aiokafka.ConsumerStoppedError:
            raise StopAsyncIteration from None
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
                if hasattr(loop, "_put_checkpoint_fut"):
                    await loop._put_checkpoint_fut
                if new_tasks := [t for t in loop.tasks.values() if not t.scheduled]:
                    # send messages to executor
                    for task in new_tasks:
                        if task.scheduled:
                            continue
                        await self.producer.send(
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
                    # flush producer
                    await self.producer.flush()
                    # mark as scheduled
                    for task in new_tasks:
                        loop.put_writes(task.id, [(SCHEDULED, None)])
        # return message
        return msg
