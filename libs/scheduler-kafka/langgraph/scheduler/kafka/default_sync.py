import concurrent.futures
from typing import Optional, Sequence

from kafka import KafkaConsumer, KafkaProducer
from langgraph.scheduler.kafka.types import ConsumerRecord, TopicPartition


class DefaultConsumer(KafkaConsumer):
    def getmany(
        self, timeout_ms: int, max_records: int
    ) -> dict[TopicPartition, Sequence[ConsumerRecord]]:
        return self.poll(timeout_ms=timeout_ms, max_records=max_records)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DefaultProducer(KafkaProducer):
    def send(
        self,
        topic: str,
        *,
        key: Optional[bytes] = None,
        value: Optional[bytes] = None,
    ) -> concurrent.futures.Future:
        fut = concurrent.futures.Future()
        kfut = super().send(topic, key=key, value=value)
        kfut.add_callback(fut.set_result)
        kfut.add_errback(fut.set_exception)
        return fut

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
