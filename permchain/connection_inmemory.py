import threading
import queue
from collections import defaultdict
from typing import Any, Iterator

q = queue.Queue()

from permchain.connection import PubSubConnection, PubSubListener


class IterableQueue(queue.SimpleQueue):
    done_sentinel = object()

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        return super().get(block=block, timeout=timeout)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.get, self.done_sentinel)

    def close(self) -> None:
        self.put(self.done_sentinel)


class InMemoryPubSubConnection(PubSubConnection):
    topics: defaultdict[str, IterableQueue]
    listeners: defaultdict[str, list[PubSubListener]]
    lock: threading.RLock

    def __init__(self) -> None:
        self.topics = defaultdict(IterableQueue)
        self.listeners = defaultdict(list)
        self.lock = threading.RLock()

    def iterate(self, topic_name: str) -> Iterator[Any]:
        with self.lock:
            if self.listeners[topic_name]:
                raise RuntimeError(
                    f"Cannot iterate over topic {topic_name} while listeners are connected"
                )

        return iter(self.topics[topic_name])

    def listen(self, topic_name: str, listeners: list[PubSubListener]) -> None:
        self.disconnect(topic_name)

        with self.lock:
            self.listeners[topic_name].extend(listeners)
            topic_queue = self.topics[topic_name]
            while not topic_queue.empty():
                message = topic_queue.get()
                for listener in self.listeners[topic_name]:
                    listener(message)

    def send(self, topic_name: str, message: Any) -> None:
        with self.lock:
            listeners = self.listeners[topic_name]
            if listeners:
                for listener in listeners:
                    listener(message)
            else:
                self.topics[topic_name].put(message)

    def disconnect(self, prefix: str) -> None:
        with self.lock:
            to_delete = []
            for topic, queue in self.topics.items():
                if topic.startswith(prefix):
                    queue.close()
                    to_delete.append(topic)
            # can't delete while iterating
            for topic in to_delete:
                del self.topics[topic]

            to_delete = []
            for topic in self.listeners:
                if topic.startswith(prefix):
                    to_delete.append(topic)
            # can't delete while iterating
            for topic in to_delete:
                del self.listeners[topic]
