from typing import AsyncIterator, Iterator
from uuid import uuid4

import kafka.admin
import pytest
from psycopg import AsyncConnection

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.scheduler.kafka.types import Topics

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5443/"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def topics() -> Iterator[Topics]:
    o = f"test_{uuid4().hex[:16]}"
    e = f"test_{uuid4().hex[:16]}"
    admin = kafka.admin.KafkaAdminClient()
    # create topics
    admin.create_topics(
        [
            kafka.admin.NewTopic(name=o, num_partitions=1, replication_factor=1),
            kafka.admin.NewTopic(name=e, num_partitions=1, replication_factor=1),
        ]
    )
    # yield topics
    yield Topics(orchestrator=o, executor=e)
    # delete topics
    admin.delete_topics([o, e])
    admin.close()


@pytest.fixture
async def checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with AsyncPostgresSaver.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")
