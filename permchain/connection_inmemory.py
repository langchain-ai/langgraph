import queue
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterator

from permchain.connection import LogMessage, PubSubConnection, PubSubListener


class IterableQueue(queue.SimpleQueue):
    done_sentinel = object()

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        return super().get(block=block, timeout=timeout)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.get, self.done_sentinel)

    def close(self) -> None:
        self.put(self.done_sentinel)


class InMemoryPubSubConnection(PubSubConnection):
    clear_on_disconnect: bool
    logs: defaultdict[str, list[Any]]
    topics: defaultdict[str, IterableQueue]
    listeners: defaultdict[str, list[PubSubListener]]
    lock: threading.RLock

    def __init__(self, clear_on_disconnect: bool = True) -> None:
        self.clear_on_disconnect = clear_on_disconnect
        self.logs = defaultdict(list)
        self.topics = defaultdict(IterableQueue)
        self.listeners = defaultdict(list)
        self.lock = threading.RLock()

    def peek(self, prefix: str) -> Iterator[LogMessage]:
        return iter(self.logs[prefix])

    def iterate(self, prefix: str, topic_name: str) -> Iterator[Any]:
        topic = self.full_topic_name(prefix, topic_name)
        with self.lock:
            if self.listeners[topic]:
                raise RuntimeError(
                    f"Cannot iterate over topic {topic} while listeners are connected"
                )

        return iter(self.topics[topic])

    def listen(
        self, prefix: str, topic_name: str, listeners: list[PubSubListener]
    ) -> None:
        topic = self.full_topic_name(prefix, topic_name)
        self.disconnect(topic)

        with self.lock:
            # Add the listeners for future messages
            self.listeners[topic].extend(listeners)

            # Send any pending messages to the listeners
            topic_queue = self.topics[topic]
            while not topic_queue.empty():
                message = topic_queue.get()
                if message is not topic_queue.done_sentinel:
                    for listener in self.listeners[topic]:
                        listener(message)

    def send(self, prefix: str, topic_name: str, message: Any) -> None:
        topic = self.full_topic_name(prefix, topic_name)

        with self.lock:
            # Add the message to the log
            self.logs[prefix].append(
                LogMessage(
                    message=message,
                    topic_name=topic_name,
                    started_at=datetime.now().isoformat(),
                )
            )
            listeners = self.listeners[topic]
            if listeners:
                # Send the message to listeners if any are connected
                for listener in listeners:
                    listener(message)
            else:
                # Otherwise add the message to the topic queue for later
                self.topics[topic].put(message)

    def disconnect(self, prefix_or_topic: str) -> None:
        with self.lock:
            if self.clear_on_disconnect:
                if prefix_or_topic in self.logs:
                    del self.logs[prefix_or_topic]

            to_delete = []
            for topic, queue in self.topics.items():
                if topic.startswith(prefix_or_topic):
                    queue.close()
                    if self.clear_on_disconnect:
                        to_delete.append(topic)
            # can't delete while iterating
            for topic in to_delete:
                del self.topics[topic]

            to_delete = []
            for topic in self.listeners:
                if topic.startswith(prefix_or_topic):
                    to_delete.append(topic)
            # can't delete while iterating
            for topic in to_delete:
                del self.listeners[topic]
