import asyncio
from contextlib import AbstractAsyncContextManager
from typing import Any

import aiokafka

import langgraph.scheduler.kafka.serde as serde
from langgraph.constants import ERROR
from langgraph.errors import TaskNotFound
from langgraph.pregel import Pregel
from langgraph.pregel.algo import prepare_single_task
from langgraph.pregel.executor import AsyncBackgroundExecutor
from langgraph.pregel.manager import AsyncChannelsManager
from langgraph.pregel.runner import PregelRunner
from langgraph.scheduler.kafka.types import (
    MessageToExecutor,
    MessageToOrchestrator,
    Topics,
)


class KafkaExecutor(AbstractAsyncContextManager):
    def __init__(self, graph: Pregel, topics: Topics, **kwargs: Any) -> None:
        self.graph = graph
        self.topics = topics
        self.consumer = aiokafka.AIOKafkaConsumer(
            topics.executor, value_deserializer=serde.loads, **kwargs
        )
        self.producer = aiokafka.AIOKafkaProducer(
            value_serializer=serde.dumps, **kwargs
        )

    async def __aenter__(self) -> "KafkaExecutor":
        await self.consumer.start()
        await self.producer.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.consumer.stop()
        await self.producer.stop()

    def __aiter__(self) -> "KafkaExecutor":
        return self

    async def __anext__(self) -> Any:
        # wait for next message
        try:
            rec = await self.consumer.getone()
            msg: MessageToExecutor = rec.value
        except aiokafka.ConsumerStoppedError:
            raise StopAsyncIteration from None
        # process message
        saved = await self.graph.checkpointer.aget_tuple(msg["config"])
        if saved is None:
            raise RuntimeError("Checkpoint not found")
        async with AsyncChannelsManager(
            self.graph.channels, saved.checkpoint, msg["config"], self.graph.store
        ) as (channels, managed), AsyncBackgroundExecutor() as submit:

            def put_writes(task_id: str, writes: list[tuple[str, Any]]) -> None:
                print("put_writes", task_id, writes)
                return submit(
                    self.graph.checkpointer.aput_writes, msg["config"], writes, task_id
                )

            if task := await asyncio.to_thread(
                prepare_single_task,
                msg["task"]["path"],
                msg["task"]["id"],
                checkpoint=saved.checkpoint,
                processes=self.graph.nodes,
                channels=channels,
                managed=managed,
                config=msg["config"],
                step=msg["task"]["step"],
                for_execution=True,
                is_resuming=msg["task"]["resuming"],
            ):
                # execute task, saving writes
                runner = PregelRunner(submit=submit, put_writes=put_writes)
                async for _ in runner.atick([task]):
                    pass
            else:
                # task was not found
                await self.graph.checkpointer.put_writes(
                    msg["config"], [(ERROR, TaskNotFound())]
                )
        # notify orchestrator
        await self.producer.send(
            self.topics.orchestrator,
            value=MessageToOrchestrator(input=None, config=msg["config"]),
        )
        # return message
        return msg
