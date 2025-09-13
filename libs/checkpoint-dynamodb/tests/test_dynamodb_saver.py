from moto import mock_aws
import boto3
from langgraph.checkpoint.dynamodb import DynamoDBSaver

@mock_aws
def test_put_get_list_put_writes_delete():
    # Create mock tables
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    ddb.create_table(
        TableName="checkpoints",
        BillingMode="PAY_PER_REQUEST",
        KeySchema=[{"AttributeName":"thread_id","KeyType":"HASH"},
                   {"AttributeName":"checkpoint_id","KeyType":"RANGE"}],
        AttributeDefinitions=[{"AttributeName":"thread_id","AttributeType":"S"},
                              {"AttributeName":"checkpoint_id","AttributeType":"S"}],
    )
    ddb.create_table(
        TableName="writes",
        BillingMode="PAY_PER_REQUEST",
        KeySchema=[{"AttributeName":"thread_id_checkpoint_id_checkpoint_ns","KeyType":"HASH"},
                   {"AttributeName":"task_id_idx","KeyType":"RANGE"}],
        AttributeDefinitions=[{"AttributeName":"thread_id_checkpoint_id_checkpoint_ns","AttributeType":"S"},
                              {"AttributeName":"task_id_idx","AttributeType":"S"}],
    )

    saver = DynamoDBSaver("checkpoints","writes",region_name="us-east-1",ttl_seconds=3600)

    # Save a checkpoint
    cfg = {"configurable": {"thread_id": "t1"}}
    ckpt = {"id":"c1","v":1,"ts":"2025-01-01T00:00:00Z","channel_values":{},"channel_versions":{},"versions_seen":{}}
    meta = {"source":"input","step":-1,"parents":{}}
    cfg2 = saver.put(cfg, ckpt, meta, new_versions={})
    assert cfg2["configurable"]["checkpoint_id"] == "c1"

    # Get latest for thread
    got = saver.get({"configurable": {"thread_id": "t1"}})
    assert got and got["id"] == "c1"

    # List
    items = list(saver.list({"configurable": {"thread_id": "t1"}}, limit=5))
    assert len(items) == 1 and items[0].checkpoint["id"] == "c1"

    # Writes (smoke)
    saver.put_writes({"configurable": {"thread_id":"t1","checkpoint_id":"c1","checkpoint_ns":"0"}},
                     [("log", {"ok": True}), ("state", {"x": 1})],
                     task_id="task-1")

    # Delete
    saver.delete_thread("t1")
    assert saver.get({"configurable": {"thread_id": "t1"}}) is None
