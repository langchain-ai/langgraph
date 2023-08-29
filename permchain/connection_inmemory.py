import threading
import queue
from collections import defaultdict
from typing import Any

q = queue.Queue()

from permchain.connection import PubSubConnection, PubSubListener


class InMemoryPubSubConnection(PubSubConnection):
    listeners: defaultdict[str, list[PubSubListener]]
    lock: threading.RLock

    def __init__(self) -> None:
        self.listeners = defaultdict(list)
        self.lock = threading.RLock()

    def listen(self, topic_name: str, listener: PubSubListener) -> None:
        with self.lock:
            self.listeners[topic_name].append(listener)

    def send(self, topic_name: str, message: Any) -> None:
        with self.lock:
            for listener in self.listeners[topic_name]:
                listener(message)

    def disconnect(self, topic_name: str) -> None:
        with self.lock:
            if topic_name in self.listeners:
                del self.listeners[topic_name]
