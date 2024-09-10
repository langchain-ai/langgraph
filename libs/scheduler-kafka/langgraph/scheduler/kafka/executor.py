import asyncio
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from functools import partial
from typing import Any, Optional, Self, Sequence

import aiokafka
import orjson
from langchain_core.runnables import RunnableConfig

import langgraph.scheduler.kafka.serde as serde
from langgraph.constants import CONFIG_KEY_DELEGATE, ERROR, NS_END, NS_SEP
from langgraph.errors import CheckpointNotLatest, GraphDelegate, TaskNotFound
from langgraph.pregel import Pregel
from langgraph.pregel.algo import prepare_single_task
from langgraph.pregel.executor import AsyncBackgroundExecutor, Submit
from langgraph.pregel.manager import AsyncChannelsManager
from langgraph.pregel.runner import PregelRunner
from langgraph.pregel.types import RetryPolicy
from langgraph.scheduler.kafka.retry import aretry
from langgraph.scheduler.kafka.types import (
    ErrorMessage,
    MessageToExecutor,
    MessageToOrchestrator,
    Topics,
)
from langgraph.utils.config import patch_configurable


class KafkaExecutor(AbstractAsyncContextManager):
    def __init__(
        self,
        graph: Pregel,
        topics: Topics,
        *,
        group_id: str = "executor",
        batch_max_n: int = 10,
        batch_max_ms: int = 1000,
        retry_policy: Optional[RetryPolicy] = None,
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        self.topics = topics
        self.stack = AsyncExitStack()
        self.kwargs = kwargs
        self.group_id = group_id
        self.batch_max_n = batch_max_n
        self.batch_max_ms = batch_max_ms
        self.retry_policy = retry_policy

    async def __aenter__(self) -> Self:
        self.consumer = await self.stack.enter_async_context(
            aiokafka.AIOKafkaConsumer(
                self.topics.executor,
                value_deserializer=serde.loads,
                auto_offset_reset="earliest",
                group_id=self.group_id,
                enable_auto_commit=False,
                **self.kwargs,
            )
        )
        self.producer = await self.stack.enter_async_context(
            aiokafka.AIOKafkaProducer(
                key_serializer=serde.dumps,
                value_serializer=serde.dumps,
                **self.kwargs,
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

    async def __anext__(self) -> Sequence[MessageToExecutor]:
        # wait for next batch
        try:
            recs = await self.consumer.getmany(
                timeout_ms=self.batch_max_ms, max_records=self.batch_max_n
            )
            msgs: list[MessageToExecutor] = [
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

    async def each(self, msg: MessageToExecutor) -> None:
        try:
            await aretry(self.retry_policy, self.attempt, msg)
        except CheckpointNotLatest:
            pass
        except GraphDelegate as exc:
            for arg in exc.args:
                await self.producer.send_and_wait(
                    self.topics.orchestrator,
                    value=MessageToOrchestrator(
                        config=arg["config"],
                        input=orjson.Fragment(
                            self.graph.checkpointer.serde.dumps(arg["input"])
                        ),
                        finally_executor=[msg],
                    ),
                    # use thread_id, checkpoint_ns as partition key
                    key=(
                        arg["config"]["configurable"]["thread_id"],
                        arg["config"]["configurable"].get("checkpoint_ns"),
                    ),
                )
        except Exception as exc:
            await self.producer.send_and_wait(
                self.topics.error,
                value=ErrorMessage(
                    topic=self.topics.executor,
                    msg=msg,
                    error=repr(exc),
                ),
            )

    async def attempt(self, msg: MessageToExecutor) -> None:
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
        saved = await self.graph.checkpointer.aget_tuple(
            patch_configurable(msg["config"], {"checkpoint_id": None})
        )
        if saved is None:
            raise RuntimeError("Checkpoint not found")
        if saved.checkpoint["id"] != msg["config"]["configurable"]["checkpoint_id"]:
            raise CheckpointNotLatest()
        async with AsyncChannelsManager(
            graph.channels, saved.checkpoint, msg["config"], self.graph.store
        ) as (channels, managed), AsyncBackgroundExecutor() as submit:
            if task := await asyncio.to_thread(
                prepare_single_task,
                msg["task"]["path"],
                msg["task"]["id"],
                checkpoint=saved.checkpoint,
                processes=graph.nodes,
                channels=channels,
                managed=managed,
                config=patch_configurable(msg["config"], {CONFIG_KEY_DELEGATE: True}),
                step=saved.metadata["step"] + 1,
                for_execution=True,
                checkpointer=self.graph.checkpointer,
            ):
                # execute task, saving writes
                runner = PregelRunner(
                    submit=submit,
                    put_writes=partial(self._put_writes, submit, msg["config"]),
                )
                async for _ in runner.atick([task], reraise=False):
                    pass
            else:
                # task was not found
                await self.graph.checkpointer.put_writes(
                    msg["config"], [(ERROR, TaskNotFound())]
                )
        # notify orchestrator
        await self.producer.send_and_wait(
            self.topics.orchestrator,
            value=MessageToOrchestrator(
                input=None,
                config=msg["config"],
                finally_executor=msg.get("finally_executor"),
            ),
            # use thread_id, checkpoint_ns as partition key
            key=(
                msg["config"]["configurable"]["thread_id"],
                msg["config"]["configurable"].get("checkpoint_ns"),
            ),
        )

    def _put_writes(
        self,
        submit: Submit,
        config: RunnableConfig,
        task_id: str,
        writes: list[tuple[str, Any]],
    ) -> None:
        return submit(self.graph.checkpointer.aput_writes, config, writes, task_id)
