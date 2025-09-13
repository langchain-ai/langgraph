from __future__ import annotations
import json
import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import boto3
from botocore.client import BaseClient

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langchain_core.runnables.config import RunnableConfig

DEFAULT_SERDE: SerializerProtocol | None = None  # defer to LangGraph default if None


class DynamoDBSaver(BaseCheckpointSaver):
    """
    Minimal DynamoDB-backed CheckpointSaver.

    Tables:
      checkpoints: PK=thread_id (S), SK=checkpoint_id (S)
      writes: PK=thread_id_checkpoint_id_checkpoint_ns (S), SK=task_id_idx (S)

    TTL: if you enable TTL on the tables, the attribute name is "ttl" and its value
    must be UNIX epoch **seconds** (Number).
    """

    def __init__(
        self,
        checkpoints_table: str,
        writes_table: str,
        *,
        client: Optional[BaseClient] = None,
        serde: Optional[SerializerProtocol] = None,
        ttl_seconds: Optional[int] = None,
        region_name: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(serde=serde or DEFAULT_SERDE)
        self._client = client or boto3.client("dynamodb", region_name=region_name, **(client_kwargs or {}))
        self._checkpoints_table = checkpoints_table
        self._writes_table = writes_table
        self._ttl_seconds = ttl_seconds

    # ----- helpers -----
    @staticmethod
    def _config_ids(config: RunnableConfig) -> Tuple[Optional[str], Optional[str]]:
        cfg = config.get("configurable", {}) if config else {}
        return cfg.get("thread_id"), cfg.get("checkpoint_id")

    @staticmethod
    def _now_s() -> int:
        return int(time.time())

    def _ttl(self) -> Optional[int]:
        return self._now_s() + self._ttl_seconds if self._ttl_seconds else None

    # ----- API -----
    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        tup = self.get_tuple(config)
        return tup.checkpoint if tup else None

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id, checkpoint_id = self._config_ids(config)
        if not thread_id:
            return None

        if checkpoint_id:
            resp = self._client.get_item(
                TableName=self._checkpoints_table,
                Key={"thread_id": {"S": thread_id}, "checkpoint_id": {"S": checkpoint_id}},
                ConsistentRead=True,
            )
            item = resp.get("Item")
            return self._to_tuple(item) if item else None

        # latest by descending SK
        resp = self._client.query(
            TableName=self._checkpoints_table,
            KeyConditionExpression="thread_id = :tid",
            ExpressionAttributeValues={":tid": {"S": thread_id}},
            ScanIndexForward=False,
            Limit=1,
            ConsistentRead=True,
        )
        items = resp.get("Items", [])
        return self._to_tuple(items[0]) if items else None

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id, _ = self._config_ids(config or {})
        if not thread_id:
            return iter(())
        kwargs: Dict[str, Any] = dict(
            TableName=self._checkpoints_table,
            KeyConditionExpression="thread_id = :tid",
            ExpressionAttributeValues={":tid": {"S": thread_id}},
            ScanIndexForward=False,
        )
        if before:
            _, before_id = self._config_ids(before)
            if before_id:
                kwargs["KeyConditionExpression"] += " AND checkpoint_id < :cid"
                kwargs["ExpressionAttributeValues"][":cid"] = {"S": before_id}
        if limit:
            kwargs["Limit"] = int(limit)

        resp = self._client.query(**kwargs)
        for item in resp.get("Items", []):
            yield self._to_tuple(item)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, str],
    ) -> RunnableConfig:
        thread_id, _ = self._config_ids(config)
        if not thread_id:
            raise ValueError("config.configurable.thread_id is required")

        ttl = self._ttl()
        item = {
            "thread_id": {"S": thread_id},
            "checkpoint_id": {"S": checkpoint["id"]},
            "checkpoint": {"S": json.dumps(checkpoint)},
            "metadata": {"S": json.dumps(metadata)},
            "parent": {"S": json.dumps(config.get("configurable", {}))},
            "ts": {"N": str(int(time.time() * 1000))},
        }
        if ttl:
            item["ttl"] = {"N": str(ttl)}
        self._client.put_item(TableName=self._checkpoints_table, Item=item)

        cfg = dict(config)
        cfg.setdefault("configurable", {})
        cfg["configurable"]["checkpoint_id"] = checkpoint["id"]
        return cfg

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, checkpoint_id = self._config_ids(config)
        if not thread_id or not checkpoint_id:
            return  # nothing to persist
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "0")
        pk = f"{thread_id}:{checkpoint_id}:{checkpoint_ns}"
        ttl = self._ttl()

        # batch write in chunks of <=25
        for start in range(0, len(writes), 25):
            chunk = writes[start : start + 25]
            reqs = []
            for idx, (channel, value) in enumerate(chunk):
                item = {
                    "thread_id_checkpoint_id_checkpoint_ns": {"S": pk},
                    "task_id_idx": {"S": f"{task_id}:{idx:06d}"},
                    "channel": {"S": channel},
                    "value": {"S": json.dumps(value)},
                    "ts": {"N": str(self._now_s())},
                }
                if ttl:
                    item["ttl"] = {"N": str(ttl)}
                reqs.append({"PutRequest": {"Item": item}})
            self._client.batch_write_item(RequestItems={self._writes_table: reqs})

    def delete_thread(self, thread_id: str) -> None:
        # delete all checkpoints for thread
        resp = self._client.query(
            TableName=self._checkpoints_table,
            KeyConditionExpression="thread_id = :tid",
            ExpressionAttributeValues={":tid": {"S": thread_id}},
            ProjectionExpression="checkpoint_id",
        )
        for it in resp.get("Items", []):
            self._client.delete_item(
                TableName=self._checkpoints_table,
                Key={"thread_id": {"S": thread_id}, "checkpoint_id": {"S": it["checkpoint_id"]["S"]}},
            )

        # naive scan to delete writes; for scale, a GSI could be added later
        wr = self._client.scan(
            TableName=self._writes_table,
            ProjectionExpression="thread_id_checkpoint_id_checkpoint_ns, task_id_idx",
        )
        for it in wr.get("Items", []):
            if it["thread_id_checkpoint_id_checkpoint_ns"]["S"].startswith(f"{thread_id}:"):
                self._client.delete_item(
                    TableName=self._writes_table,
                    Key={
                        "thread_id_checkpoint_id_checkpoint_ns": {"S": it["thread_id_checkpoint_id_checkpoint_ns"]["S"]},
                        "task_id_idx": {"S": it["task_id_idx"]["S"]},
                    },
                )

    # ----- async shims -----
    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        return self.get(config)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ):
        return self.list(config, filter=filter, before=before, limit=limit)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, str],
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)

    # ----- conversion -----
    def _to_tuple(self, item: Dict[str, Any]) -> CheckpointTuple:
        checkpoint: Checkpoint = json.loads(item["checkpoint"]["S"])
        metadata: CheckpointMetadata = json.loads(item["metadata"]["S"])
        cfg = json.loads(item.get("parent", {"S": "{}"})["S"])
        return CheckpointTuple(config=cfg, checkpoint=checkpoint, metadata=metadata)
