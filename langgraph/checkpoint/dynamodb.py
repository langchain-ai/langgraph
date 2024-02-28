import boto3

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec
from copy import deepcopy
from typing import Any, Optional

ddb = boto3.resource("dynamodb", region_name="us-east-1")


class DynamoDBSaver(BaseCheckpointSaver):
    table: Any

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_table(cls, table_name: str) -> "DynamoDBSaver":
        return DynamoDBSaver(table=ddb.Table(table_name))

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id="thread_id",
                annotation=str,
                name="Activation ID",
                description=None,
                default="",
                is_shared=True,
            ),
        ]

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        result = self.table.get_item(Key={"thread_id": config["configurable"]["thread_id"]})
        if "Item" not in result:
            return None

        checkpoint = result["Item"]

        return Checkpoint(
            v=checkpoint["v"],
            ts=checkpoint["ts"],
            channel_values=checkpoint["channel_values"].copy(),
            channel_versions=checkpoint["channel_versions"].copy(),
            versions_seen=deepcopy(checkpoint["versions_seen"]),
        )

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        return self.table.put_item(Item={"thread_id": config["configurable"]["thread_id"], **checkpoint})
