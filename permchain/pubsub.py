from __future__ import annotations

import queue
from abc import ABC
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from itertools import filterfalse
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from langchain.callbacks.manager import CallbackManager
from langchain.load.dump import dumpd
from langchain.load.serializable import Serializable
from langchain.schema.runnable import Runnable, RunnableConfig, _patch_config

from permchain.connection import PubSubConnection
from permchain.topic import INPUT_TOPIC, OUTPUT_TOPIC, RunnableSubscriber

T = TypeVar("T")
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


def partition(
    pred: Callable[[T], bool], seq: Sequence[T]
) -> Tuple[Sequence[T], Sequence[T]]:
    """Partition entries into true entries and false entries.

    partition(is_even, range(10)) --> 0 2 4 6 8  and  1 3 5 7 9
    """
    return list(filter(pred, seq)), list(filterfalse(pred, seq))


class IterableQueue(queue.SimpleQueue):
    done_sentinel = object()

    def get(self, block: bool = True, timeout: float = None):
        return super().get(block=block, timeout=timeout)

    def __iter__(self):
        return iter(self.get, self.done_sentinel)

    def close(self):
        self.put(self.done_sentinel)


class PubSub(Serializable, Runnable[Any, Any], ABC):
    processes: Sequence[RunnableSubscriber[Any]]

    connection: PubSubConnection

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        collected = []
        for chunk in self.stream(input, config):
            collected.append(chunk)
        return collected

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> Iterator[Any]:
        input_processes, listener_processes = partition(
            lambda r: r.topic.name == INPUT_TOPIC, self.processes
        )

        # setup callbacks
        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = callback_manager.on_chain_start(dumpd(self), {"input": input})

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            # Track inflight futures
            inflight: Set[Future] = set()
            # Track exceptions
            exceptions: List[Exception] = []
            # Track output
            output = IterableQueue()

            def send(topic_name: str, message: Any) -> None:
                """Send a message to a topic. Injected into config."""
                if topic_name == OUTPUT_TOPIC:
                    output.put(message)
                else:
                    self.connection.send(topic_name, message)

            def cleanup_run(fut: Future) -> None:
                """Cleanup after a process runs."""
                inflight.remove(fut)

                try:
                    exc = fut.exception()
                except CancelledError:
                    exc = None
                except Exception as e:
                    exc = e
                if exc is not None:
                    exceptions.append(exc)

                # Close output iterator if
                # - all processes are done, or
                # - an exception occurred
                if not inflight or exc is not None:
                    output.close()

            def run_once(process: RunnableSubscriber[Any], value: Any) -> None:
                """Run a process once."""

                def get(topic_name: str) -> Any:
                    if topic_name == INPUT_TOPIC:
                        return input
                    elif topic_name == process.topic.name:
                        return value
                    else:
                        raise ValueError(
                            f"Cannot get value for {topic_name} in this context"
                        )

                # Run process once in executor
                fut = executor.submit(
                    process.invoke,
                    value,
                    config={
                        **_patch_config(
                            config, run_manager.get_child(process.topic.name)
                        ),
                        "send": send,
                        "get": get,
                    },
                )

                # Add callback to cleanup
                inflight.add(fut)
                fut.add_done_callback(cleanup_run)

            # Listen on all subscribed topics
            for process in listener_processes:
                self.connection.listen(process.topic.name, partial(run_once, process))

            # Run input processes once
            for process in input_processes:
                run_once(process, input)

            try:
                # Yield output until all processes are done
                final_output = None
                for chunk in output:
                    yield chunk
                    if final_output is None:
                        final_output = chunk
                    else:
                        final_output += chunk
            finally:
                # Cleanup
                for fut in inflight:
                    fut.cancel()

                for process in listener_processes:
                    self.connection.disconnect(process.topic.name)

                # Raise exceptions if any
                if exceptions:
                    run_manager.on_chain_error(exceptions[0])
                    raise exceptions[0]
                else:
                    run_manager.on_chain_end(final_output)


PubSub.update_forward_refs()
