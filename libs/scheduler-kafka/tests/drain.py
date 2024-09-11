import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, TypeVar

import anyio
from aiokafka import AIOKafkaConsumer
from typing_extensions import ParamSpec

from langgraph.pregel import Pregel
from langgraph.scheduler.kafka.default_sync import DefaultConsumer
from langgraph.scheduler.kafka.executor import AsyncKafkaExecutor, KafkaExecutor
from langgraph.scheduler.kafka.orchestrator import (
    AsyncKafkaOrchestrator,
    KafkaOrchestrator,
)
from langgraph.scheduler.kafka.types import MessageToOrchestrator, Topics

C = ParamSpec("C")
R = TypeVar("R")


async def drain_topics_async(
    topics: Topics, graph: Pregel, *, debug: bool = False
) -> tuple[list[MessageToOrchestrator], list[MessageToOrchestrator]]:
    scope: Optional[anyio.CancelScope] = None
    orch_msgs = []
    exec_msgs = []
    errors = []

    def done() -> bool:
        return (
            len(orch_msgs) > 0
            and len(exec_msgs) > 0
            and not orch_msgs[-1]
            and not exec_msgs[-1]
        )

    async def orchestrator() -> None:
        async with AsyncKafkaOrchestrator(graph, topics) as orch:
            async for msgs in orch:
                orch_msgs.append(msgs)
                if debug:
                    print("\n---\norch", len(msgs), msgs)
                if done():
                    scope.cancel()

    async def executor() -> None:
        async with AsyncKafkaExecutor(graph, topics) as exec:
            async for msgs in exec:
                exec_msgs.append(msgs)
                if debug:
                    print("\n---\nexec", len(msgs), msgs)
                if done():
                    scope.cancel()

    async def error_consumer() -> None:
        async with AIOKafkaConsumer(topics.error) as consumer:
            async for msg in consumer:
                errors.append(msg)
                if scope:
                    scope.cancel()

    # start error consumer
    error_task = asyncio.create_task(error_consumer(), name="error_consumer")

    # run the orchestrator and executor until break_when
    async with anyio.create_task_group() as tg:
        tg.cancel_scope.deadline = anyio.current_time() + 20
        scope = tg.cancel_scope
        tg.start_soon(orchestrator, name="orchestrator")
        tg.start_soon(executor, name="executor")

    # cancel error consumer
    error_task.cancel()

    try:
        await error_task
    except asyncio.CancelledError:
        pass

    # check no errors
    assert not errors, errors

    return [m for mm in orch_msgs for m in mm], [m for mm in exec_msgs for m in mm]


def drain_topics(
    topics: Topics, graph: Pregel, *, debug: bool = False
) -> tuple[list[MessageToOrchestrator], list[MessageToOrchestrator]]:
    orch_msgs = []
    exec_msgs = []
    errors = []
    event = threading.Event()

    def done() -> bool:
        return (
            len(orch_msgs) > 0
            and len(exec_msgs) > 0
            and not orch_msgs[-1]
            and not exec_msgs[-1]
        )

    def orchestrator() -> None:
        try:
            with KafkaOrchestrator(graph, topics) as orch:
                for msgs in orch:
                    orch_msgs.append(msgs)
                    if debug:
                        print("\n---\norch", len(msgs), msgs)
                    if done():
                        print("am i done? orchestrator")
                        event.set()
                    if event.is_set():
                        break
        except Exception as e:
            errors.append(e)
            event.set()

    def executor() -> None:
        try:
            with KafkaExecutor(graph, topics) as exec:
                for msgs in exec:
                    exec_msgs.append(msgs)
                    if debug:
                        print("\n---\nexec", len(msgs), msgs)
                    if done():
                        print("am i done? executor")
                        event.set()
                    if event.is_set():
                        break
        except Exception as e:
            errors.append(e)
            event.set()

    def error_consumer() -> None:
        try:
            with DefaultConsumer(topics.error) as consumer:
                while not event.is_set():
                    if msg := consumer.poll(timeout_ms=100):
                        errors.append(msg)
                        event.set()
        except Exception as e:
            errors.append(e)
            event.set()

    with ThreadPoolExecutor() as pool:
        # start error consumer
        pool.submit(error_consumer)

        # run the orchestrator and executor until break_when
        pool.submit(orchestrator)
        pool.submit(executor)

        # timeout
        start = time.time()
        while not event.is_set():
            time.sleep(0.1)
            if time.time() - start > 20:
                event.set()

    # check no errors
    assert not errors, errors

    return [m for mm in orch_msgs for m in mm], [m for mm in exec_msgs for m in mm]
