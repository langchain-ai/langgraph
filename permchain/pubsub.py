from __future__ import annotations

from abc import ABC
from concurrent.futures import CancelledError, Future
from functools import partial
from itertools import groupby
from typing import Any, Iterator, List, Optional, Sequence, Set, TypeVar

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.load.serializable import Serializable
from langchain.schema.runnable import Runnable, RunnableConfig, patch_config
from langchain.schema.runnable.base import Runnable
from langchain.schema.runnable.config import get_executor_for_config

from permchain.connection import PubSubConnection
from permchain.constants import CONFIG_GET_KEY, CONFIG_SEND_KEY
from permchain.topic import INPUT_TOPIC, OUTPUT_TOPIC, RunnableSubscriber

T = TypeVar("T")
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class PubSub(Serializable, Runnable[Any, Any], ABC):
    processes: Sequence[RunnableSubscriber[Any]]

    connection: PubSubConnection

    class Config:
        arbitrary_types_allowed = True

    def with_retry(self, **kwargs: Any) -> Runnable[Any, Any]:
        return self.__class__(
            processes=[p.with_retry(**kwargs) for p in self.processes],
            connection=self.connection,
        )

    def _transform(
        self,
        input: Iterator[Any],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Iterator[Any]:
        # Consume input iterator into a single value
        input_value = None
        for chunk in input:
            if input_value is None:
                input_value = chunk
            else:
                input_value += chunk

        with get_executor_for_config(config) as executor:
            # Namespace topics for each run
            topic_prefix = str(run_manager.parent_run_id or run_manager.run_id)
            # Track inflight futures
            inflight: Set[Future] = set()
            # Track exceptions
            exceptions: List[Exception] = []

            def run_once(process: RunnableSubscriber[Any], value: Any) -> None:
                """Run a process once."""

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
                        self.connection.disconnect(topic_prefix)

                def get(topic_name: str) -> Any:
                    if topic_name == INPUT_TOPIC:
                        return input_value
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
                            callbacks=run_manager.get_child(),
                            run_name=f"Topic: {process.topic.name}",
                        ),
                        CONFIG_SEND_KEY: partial(self.connection.send, topic_prefix),
                        CONFIG_GET_KEY: get,
                    },
                )

                # Add callback to cleanup
                inflight.add(fut)
                fut.add_done_callback(cleanup_run)

            # Listen on all subscribed topics
            listeners_by_topic = groupby(
                sorted(self.processes, key=lambda p: p.topic.name),
                lambda p: p.topic.name,
            )
            for topic_name, processes in listeners_by_topic:
                self.connection.listen(
                    topic_prefix,
                    topic_name,
                    [partial(run_once, process) for process in processes],
                )

            # Send input to input processes
            self.connection.send(topic_prefix, INPUT_TOPIC, input_value)

            try:
                if inflight:
                    # Yield output until all processes are done
                    for chunk in self.connection.iterate(topic_prefix, OUTPUT_TOPIC):
                        yield chunk
                else:
                    self.connection.disconnect(topic_prefix)
            finally:
                # Cancel all inflight futures
                while inflight:
                    inflight.pop().cancel()

                # Raise exceptions if any
                if exceptions:
                    raise exceptions[0]

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Any]:
        yield from self._transform_stream_with_config(
            iter([input]), self._transform, config, **kwargs
        )

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        collected = []
        for chunk in self.stream(input, config):
            collected.append(chunk)
        return collected


PubSub.update_forward_refs()
