import dataclasses
from typing import Any, Sequence

import aiokafka


class DefaultAsyncConsumer(aiokafka.AIOKafkaConsumer):
    async def getmany(
        self, timeout_ms: int, max_records: int
    ) -> dict[str, Sequence[dict[str, Any]]]:
        batch = await super().getmany(timeout_ms=timeout_ms, max_records=max_records)
        return {t: [dataclasses.asdict(m) for m in msgs] for t, msgs in batch.items()}


class DefaultAsyncProducer(aiokafka.AIOKafkaProducer):
    pass
