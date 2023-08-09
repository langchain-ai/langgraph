from collections import defaultdict
import threading
from typing import Any

from permchain.connection import PubSubConnection, PubSubListener


class InMemoryPubSubConnection(PubSubConnection):
    def __init__(self) -> None:
        self.topics = defaultdict(list)
        self.lock = threading.Lock()

    def listen(self, topic_name: str, listener: PubSubListener) -> None:
        with self.lock:
            self.topics[topic_name].append(listener)

    def send(self, topic_name: str, message: Any) -> None:
        with self.lock:
            for listener in self.topics[topic_name]:
                listener(message)

    def disconnect(self, topic_name: str) -> None:
        with self.lock:
            self.topics[topic_name] = []
