from __future__ import annotations
from ctypes import Union

import queue
from abc import ABC
from concurrent.futures import CancelledError, Future
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

from langchain.load.dump import dumpd
from langchain.load.serializable import Serializable
from langchain.schema.runnable import Runnable, RunnableConfig, patch_config
from langchain.schema.runnable.config import (
    ensure_config,
    get_callback_manager_for_config,
    get_executor_for_config,
)

from permchain.connection import PubSubConnection
from permchain.constants import CONFIG_GET_KEY, CONFIG_SEND_KEY
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

    def batch(
        self,
        input: Sequence[Any],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
    ) -> Any:
        configs = self._get_config_list(config)
        with get_executor_for_config(configs[0]) as executor:
            return super().batch(
                input,
                [patch_config(config, executor=executor) for config in configs],
            )

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
    ) -> Iterator[Any]:
        input_processes, listener_processes = partition(
            lambda r: r.topic.name == INPUT_TOPIC, self.processes
        )

        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(dumpd(self), {"input": input})

        with get_executor_for_config(config) as executor:
            # Track inflight futures
            inflight: Set[Future] = set()
            # Track exceptions
            exceptions: List[Exception] = []
            # Track output
            output = IterableQueue()

            # Namespace topics for each run
            def prefix_topic_name(topic_name: str) -> str:
                return f"{run_manager.run_id}/{topic_name}"

            def send(topic_name: str, message: Any) -> None:
                """Send a message to a topic. Injected into config."""
                if topic_name == OUTPUT_TOPIC:
                    output.put(message)
                else:
                    self.connection.send(prefix_topic_name(topic_name), message)

            def cleanup_run(fut: Future) -> None:
                """Cleanup after a process runs."""
                inflight.discard(fut)

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
                        **patch_config(
                            config,
                            callbacks=run_manager.get_child(process.topic.name),
                            executor=executor,
                        ),
                        CONFIG_SEND_KEY: send,
                        CONFIG_GET_KEY: get,
                    },
                )

                # Add callback to cleanup
                inflight.add(fut)
                fut.add_done_callback(cleanup_run)

            # Listen on all subscribed topics
            for process in listener_processes:
                self.connection.listen(
                    prefix_topic_name(process.topic.name), partial(run_once, process)
                )

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
                while inflight:
                    inflight.pop().cancel()

                for process in listener_processes:
                    self.connection.disconnect(prefix_topic_name(process.topic.name))

                # Raise exceptions if any
                if exceptions:
                    run_manager.on_chain_error(exceptions[0])
                    raise exceptions[0]
                else:
                    run_manager.on_chain_end(final_output)


PubSub.update_forward_refs()
