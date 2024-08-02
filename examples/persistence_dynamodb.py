import pickle
import base64
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, Dict, Iterator, Optional, List, Tuple

import boto3
from boto3.dynamodb.conditions import Key

from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.serde.jsonplus import JsonPlusSerializer


class JsonPlusSerializerCompat(JsonPlusSerializer):
    """A serializer that supports loading pickled checkpoints for backwards compatibility."""

    def loads(self, data: bytes) -> Any:
        if data.startswith(b"\x80") and data.endswith(b"."):
            return pickle.loads(data)
        return super().loads(data)


class DynamoDBSaver(AbstractContextManager, BaseCheckpointSaver):
    """A checkpoint saver that stores checkpoints in a DynamoDB table."""

    serde = JsonPlusSerializerCompat()

    def __init__(
            self,
            table_name: str,
            *,
            region_name: str = 'us-east-1',
            serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)

    def __enter__(self) -> Self:
        return self
    def __exit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return True

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database."""

        thread_id = config["configurable"]["thread_id"]
        if config["configurable"].get("thread_ts"):
            response = self.table.get_item(
                Key={
                    'thread_id': thread_id,
                    'thread_ts': config["configurable"]["thread_ts"]
                }
            )
        else:
            response = self.table.query(
                KeyConditionExpression=Key('thread_id').eq(thread_id),
                ScanIndexForward=False,
                Limit=1
            )


        if 'Item' in response:
            item = response['Item']
            return CheckpointTuple(
                config,
                self.serde.loads(base64.b64decode(item["checkpoint"])),
                self.serde.loads(base64.b64decode(item["metadata"])),
                (
                    {
                        "configurable": {
                            "thread_id": item["thread_id"],
                            "thread_ts": item.get("parent_ts"),
                        }
                    }
                    if item.get("parent_ts")
                    else None
                ),
            )

        if 'Items' in response:
            items = response['Items']
            if len(items)>0 :
                for item in items:
                    return CheckpointTuple(
                        config,
                        self.serde.loads(base64.b64decode(item["checkpoint"])),
                        self.serde.loads(base64.b64decode(item["metadata"])),
                        (
                            {
                                "configurable": {
                                    "thread_id": item["thread_id"],
                                    "thread_ts": item.get("parent_ts"),
                                }
                            }
                            if item.get("parent_ts")
                            else None
                        ),
                    )

    def list(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database."""

        query = Key('thread_id').eq(config["configurable"]["thread_id"])
        if before:
            query = query & Key('thread_ts').lt(before["configurable"]["thread_ts"])

        response = self.table.query(
            KeyConditionExpression=query,
            ScanIndexForward=False,
            Limit=limit
        )

        for item in response['Items']:
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": item["thread_id"],
                        "thread_ts": item["thread_ts"],
                    }
                },
                self.serde.loads(base64.b64decode(item["checkpoint"])),
               self.serde.loads(base64.b64decode(item["metadata"])),
                (
                    {
                        "configurable": {
                            "thread_id": item["thread_id"],
                            "thread_ts": item.get("parent_ts"),
                        }
                    }
                    if item.get("parent_ts")
                    else None
                ),
            )

    def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Save a checkpoint to the database."""

        item = {
            "thread_id": config["configurable"]["thread_id"],
            "thread_ts": checkpoint["id"],
            "checkpoint": base64.b64encode(self.serde.dumps(checkpoint)).decode('utf-8'),
            "metadata": base64.b64encode(self.serde.dumps(metadata)).decode('utf-8'),
        }
        if config["configurable"].get("thread_ts"):
            item["parent_ts"] = config["configurable"]["thread_ts"]
        self.table.put_item(Item=item)
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["id"],
           }
        }

    import boto3
    from boto3.dynamodb.conditions import Attr
    def put_writes(
            self,
            config: RunnableConfig,
            writes: List[Tuple[str, Any]],
            task_id: str,
    ) -> None:

        with self.table.batch_writer() as batch:
            for idx, (channel, value) in enumerate(writes):

                batch.put_item(
                    Item={
                        "thread_id": config["configurable"]["thread_id"],
                        "thread_ts": config["configurable"]["thread_ts"]+'_'+task_id+'_'+str(idx),
                        "task_id": task_id,
                        "idx": idx,
                        "channel": channel,
                        "value": self.serde.dumps(value),
                    }
                )