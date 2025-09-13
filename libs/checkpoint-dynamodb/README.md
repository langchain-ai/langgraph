# DynamoDB Checkpointer for LangGraph

A DynamoDB-backed implementation of LangGraph’s `BaseCheckpointSaver`.

## Tables (provision or use PAY_PER_REQUEST)

**checkpoints**  
PK=`thread_id` (S), SK=`checkpoint_id` (S)  
Attributes: `checkpoint` (JSON string), `metadata` (JSON string), `parent` (JSON string), `ts` (Number), optional `ttl` (Number, epoch seconds)

**writes**  
PK=`thread_id_checkpoint_id_checkpoint_ns` (S), SK=`task_id_idx` (S)  
Attributes: `channel` (S), `value` (JSON string), `ts` (Number), optional `ttl` (Number)

**TTL tip:** Enable table TTL on attribute name `ttl`. Store UNIX epoch **seconds** (Number).

**Item size:** DynamoDB items must be ≤ ~400 KB. Keep checkpoints compact.

### Example DDL (boto3)
```python
import boto3
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
```

### Minimal usage

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.dynamodb import DynamoDBSaver

memory = DynamoDBSaver("checkpoints","writes",region_name="us-east-1",ttl_seconds=7*24*3600)

def inc(state): return {"n": state.get("n",0)+1}
g = StateGraph(dict)
g.add_node("inc", inc)
g.set_entry_point("inc")
graph = g.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"n":0}, config=config)
```
