from __future__ import annotations

import threading
from abc import ABC
from collections import defaultdict
from concurrent.futures import CancelledError, Future
from functools import partial
from typing import Any, Iterator, List, Optional, Sequence, Set, TypeVar, Union
from uuid import uuid4

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema.runnable import Runnable, RunnableConfig, patch_config
from langchain.schema.runnable.config import RunnableConfig, get_executor_for_config

from permchain.connection import PubSubConnection, PubSubMessage
from permchain.constants import CONFIG_GET_KEY, CONFIG_SEND_KEY
from permchain.topic import (
    INPUT_TOPIC,
    OUTPUT_TOPIC,
    RunnableReducer,
    RunnableSubscriber,
)

T = TypeVar("T")
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")

Process = RunnableSubscriber[T_in] | RunnableReducer[T_in]


class PubSub(Runnable[Any, Any], ABC):
    processes: Sequence[Process]

    connection: PubSubConnection

    def __init__(
        self,
        *procs: Process | Sequence[Process],
        processes: Sequence[Process] = (),
        connection: PubSubConnection,
    ) -> None:
        super().__init__()

        self.lock = threading.Lock()
        self.inflight_namespaces = set()

        self.connection = connection
        self.processes = list(processes)
        for proc in procs:
            if isinstance(proc, Sequence):
                self.processes.extend(proc)
            else:
                self.processes.append(proc)

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
        *,
        input_correlation_ids: list[str] | None = None,
    ) -> Iterator[Any]:
        # Split processes into subscribers and reducers, and group by topic
        subscribers: defaultdict[str, list[RunnableSubscriber[Any]]] = defaultdict(list)
        reducers: defaultdict[str, list[RunnableReducer[Any]]] = defaultdict(list)
        for process in self.processes:
            if isinstance(process, RunnableReducer):
                reducers[process.topic.name].append(process)
            elif isinstance(process, RunnableSubscriber):
                subscribers[process.topic.name].append(process)
            else:
                raise ValueError(f"Unknown process type: {process}")

        # Track correlation_ids for inputs and outputs
        # This is used to determine when all expected output has been received
        correlation_ids_seen_input = set()
        correlation_ids_seen_output = set()

        with get_executor_for_config(config) as executor:
            # Namespace topics for each run, default to run_id, ie. isolated
            topic_prefix = str(config.get("correlation_id") or run_manager.run_id)

            # Check if this correlation_id is currently inflight. If so, raise an error,
            # as that would make the output iterator produce incorrect results.
            with self.lock:
                if topic_prefix in self.inflight_namespaces:
                    raise RuntimeError(
                        f"Cannot run {self} in namespace {topic_prefix} "
                        "because it is currently in use"
                    )
                self.inflight_namespaces.add(topic_prefix)

            # Track inflight futures
            inflight: Set[Future] = set()
            # Track exceptions
            exceptions: List[BaseException] = []

            def on_idle() -> None:
                """Called when all subscribed topics are empty.
                It first runs any topic reducers. Then, if all subscribed topics
                still empty, it closes the computation.
                """
                if reducers:
                    for topic_name, processes in reducers.items():
                        # Collect all pending messages for each topic
                        messages = list(
                            self.connection.iterate(
                                topic_prefix, topic_name, wait=False
                            )
                        )
                        # Run each reducer once with the collected messages
                        if messages:
                            for process in processes:
                                run_once(process, messages)

                if not inflight:
                    self.connection.disconnect(topic_prefix)

            def check_if_idle(fut: Future) -> None:
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
                    on_idle()

            def run_once(
                process: RunnableSubscriber[Any] | RunnableReducer[Any],
                messages: PubSubMessage | list[PubSubMessage],
            ) -> None:
                """Run a process once."""
                value = (
                    [m["value"] for m in messages]
                    if isinstance(messages, list)
                    else messages["value"]
                )
                correlation_ids = list(
                    set(
                        [cid for m in messages for cid in m["correlation_ids"]]
                        if isinstance(messages, list)
                        else messages["correlation_ids"]
                    )
                )
                correlation_ids_seen_input.update(correlation_ids)

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
                try:
                    fut = executor.submit(
                        process.invoke,
                        value,
                        config={
                            **patch_config(
                                config,
                                callbacks=run_manager.get_child(),
                                run_name=f"Topic: {process.topic.name}",
                            ),
                            CONFIG_SEND_KEY: partial(
                                self.connection.send,
                                topic_prefix,
                                correlation_ids=correlation_ids,
                            ),
                            CONFIG_GET_KEY: get,
                            # TODO below doesn't work for batch calls nested inside
                            # another pubsub, eg. test_invoke_join_then_call_other_pubsub
                            # as all messages in each batch would share same correlation_id
                            # "correlation_id": self.connection.full_name(
                            #     topic_prefix,
                            #     process.topic.name,
                            #     str(self.processes.index(process)),
                            # ),
                        },
                    )

                    # Add callback to cleanup
                    inflight.add(fut)
                    fut.add_done_callback(check_if_idle)
                except RuntimeError:
                    # If executor is now closed, just ignore this process
                    # This could happen eg. if an OUT message was published durin
                    # execution of run_once
                    pass

            # Listen on all subscribed topics
            for topic_name, processes in subscribers.items():
                self.connection.listen(
                    topic_prefix,
                    topic_name,
                    [partial(run_once, process) for process in processes],
                )

            # Send each input value to input processes
            # Assign each input value a unique correlation_id, which can be used
            # to link derived messages back to the input value(s) that caused them
            input_correlation_ids = (
                input_correlation_ids.copy()
                if input_correlation_ids is not None
                else None
            )
            for input_value in input:
                if input_correlation_ids is None:
                    correlation_id = str(uuid4())
                else:
                    correlation_id = input_correlation_ids.pop(0)
                self.connection.send(
                    topic_prefix, INPUT_TOPIC, input_value, [correlation_id]
                )

            try:
                if inflight:
                    # Yield output until all processes are done
                    # This blocks the current thread, all other work needs to go
                    # through the executor
                    for chunk in self.connection.observe(topic_prefix):
                        yield chunk
                        if chunk["topic"] == OUTPUT_TOPIC:
                            correlation_ids_seen_output.update(chunk["correlation_ids"])
                            if (
                                correlation_ids_seen_output
                                == correlation_ids_seen_input
                            ):
                                # All expected output has been received,
                                # close the computation
                                self.connection.disconnect(topic_prefix)
                                break
                else:
                    on_idle()
            finally:
                # Cancel all inflight futures
                while inflight:
                    inflight.pop().cancel()

                # Remove namespace from inflight set
                with self.lock:
                    self.inflight_namespaces.remove(topic_prefix)

                # Raise exceptions if any
                if exceptions:
                    raise exceptions[0]

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[PubSubMessage]:
        yield from self._transform_stream_with_config(
            iter([input]), self._transform, config, **kwargs
        )

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        for chunk in self.stream(input, config):
            if chunk["topic"] == OUTPUT_TOPIC:
                return chunk["value"]

    def batch(
        self,
        inputs: List[Any],
        config: RunnableConfig | List[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> List[Any]:
        if return_exceptions:
            raise NotImplementedError("return_exceptions is not supported")
        if isinstance(config, list):
            raise NotImplementedError("config as list is not supported")

        input_correlation_ids = [str(uuid4()) for _ in inputs]
        sorted_outputs = []
        unsorted_outputs = []
        for chunk in self._transform_stream_with_config(
            iter(inputs),
            partial(self._transform, input_correlation_ids=input_correlation_ids),
            config,
            **kwargs,
        ):
            if chunk["topic"] == OUTPUT_TOPIC:
                if len(chunk["correlation_ids"]) == 1:
                    # outputs that are derived from a single input value
                    # are sorted by the order of the input value
                    idx = input_correlation_ids.index(chunk["correlation_ids"][0])
                    sorted_outputs.append((idx, chunk["value"]))
                else:
                    # outputs that are not derived from a single input value
                    # are appended to the end of the list
                    unsorted_outputs.append(chunk["value"])

        # In the case where no join() is used outputs are returned in the order
        # of the inputs sent in, and will map 1-1 to the inputs.
        # In the case where join() is used outputs are returned in the order they
        # were emitted, and may not map 1-1 to the inputs.
        sorted_outputs.sort(key=lambda x: x[0])
        return [x[1] for x in sorted_outputs] + unsorted_outputs
