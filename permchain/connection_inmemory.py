import queue
import threading
from collections import defaultdict
from typing import Iterator, cast

from permchain.connection import PubSubConnection, PubSubListener, PubSubMessage


class IterableQueue(queue.SimpleQueue):
    done_sentinel = object()

    def put(
        self, item: PubSubMessage, block: bool = True, timeout: float | None = None
    ) -> None:
        return super().put(item, block, timeout)

    def get(
        self, block: bool = True, timeout: float | None = None
    ) -> PubSubMessage | object:
        return super().get(block=block, timeout=timeout)

    def __iter__(self) -> Iterator[PubSubMessage]:
        return iter(self.get, self.done_sentinel)

    def close(self) -> None:
        self.put(self.done_sentinel)


class InMemoryPubSubConnection(PubSubConnection):
    logs: defaultdict[str, IterableQueue]
    topics: defaultdict[str, IterableQueue]
    listeners: defaultdict[str, list[PubSubListener]]
    lock: threading.RLock

    def __init__(self) -> None:
        self.logs = defaultdict(IterableQueue)
        self.topics = defaultdict(IterableQueue)
        self.listeners = defaultdict(list)
        self.lock = threading.RLock()

    def observe(self, prefix: str) -> Iterator[PubSubMessage]:
        return iter(self.logs[str(prefix)])

    def iterate(
        self, prefix: str, topic: str, *, wait: bool
    ) -> Iterator[PubSubMessage]:
        topic = self.full_name(prefix, topic)

        # This connection doesn't support iterating over topics with listeners connected
        with self.lock:
            if self.listeners[topic]:
                raise RuntimeError(
                    f"Cannot iterate over topic {topic} while listeners are connected"
                )

        # If wait is False, add sentinel to queue to ensure the iterator terminates
        if not wait:
            self.topics[topic].close()

        return iter(self.topics[topic])

    def listen(self, prefix: str, topic: str, listeners: list[PubSubListener]) -> None:
        full_name = self.full_name(prefix, topic)

        with self.lock:
            # Add the listeners for future messages
            self.listeners[full_name].extend(listeners)

            # Send any pending messages to the listeners
            topic_queue = self.topics[full_name]
            while not topic_queue.empty():
                message = topic_queue.get()
                if message is not topic_queue.done_sentinel:
                    for listener in self.listeners[full_name]:
                        listener(cast(PubSubMessage, message))

    def send(self, message: PubSubMessage) -> None:
        full_name = self.full_name(message["correlation_id"], message["topic"])

        # Add the message to the log
        self.logs[message["correlation_id"]].put(message)
        with self.lock:
            listeners = self.listeners[full_name]
            if listeners:
                # Send the message to listeners if any are connected
                for listener in listeners:
                    listener(message)
            else:
                # Otherwise add the message to the topic queue for later
                self.topics[full_name].put(message)

    def disconnect(self, name: str) -> None:
        with self.lock:
            if name in self.logs:
                self.logs[name].close()
                del self.logs[name]

            to_delete = []
            for topic, q in self.topics.items():
                if topic.startswith(name):
                    q.close()
                    # can't delete while iterating
                    to_delete.append(topic)
            for topic in to_delete:
                del self.topics[topic]

            to_delete = []
            for topic in self.listeners:
                if topic.startswith(name):
                    # can't delete while iterating
                    to_delete.append(topic)
            for topic in to_delete:
                del self.listeners[topic]
