# Migration to JSON Serializer

## Overview

The Kusto checkpointer now uses `JsonStringSerializer` by default, which stores data as JSON strings instead of msgpack binary format. This provides:

- ✅ **Better readability** - Data is human-readable in Kusto
- ✅ **No encoding** - No need for base64 encoding
- ✅ **LangChain support** - Proper serialization of LangChain messages
- ✅ **Simpler code** - Cleaner serialization logic

## Breaking Change

**Important:** The new `JsonStringSerializer` does NOT support reading old msgpack-formatted data. If you have existing data in Kusto, you need to clear it before using the new serializer.

## Clearing Old Data

### Option 1: Using Kusto Query (Recommended)

Run these commands in your Kusto cluster:

```kql
// Clear checkpoint writes
.clear table CheckpointWrites data

// Clear checkpoints
.clear table Checkpoints data
```

### Option 2: Using Python Script

Run the provided cleanup script:

```bash
python clear_kusto_data.py
```

### Option 3: Drop and Recreate Tables

If you want a fresh start:

```kql
// Drop tables
.drop table CheckpointWrites ifexists
.drop table Checkpoints ifexists

// Then re-run provision.kql to recreate them
```

## Verifying the Migration

After clearing old data, verify that new data uses JSON format:

```kql
CheckpointWrites
| where created_at > ago(1h)
| project thread_id, checkpoint_id, channel, type, value_json
| take 10
```

You should see:
- `type` column = `"json"` (not `"msgpack"`)
- `value_json` column contains readable JSON (not base64-encoded binary)

## Custom Serializers

If you need to use a different serializer:

```python
from langgraph.checkpoint.kusto import AsyncKustoSaver
from my_custom_serializer import MySerializer

checkpointer = AsyncKustoSaver(
    query_client=...,
    ingest_client=...,
    database="...",
    serde=MySerializer()  # Use your custom serializer
)
```

## Troubleshooting

### Error: "Unsupported type: msgpack"

This means you're trying to read old msgpack data with the new JSON serializer. Clear your Kusto data as described above.

### Old tutorials still use msgpack

Make sure you're using the latest version of the checkpoint-kusto library. The default serializer was changed to `JsonStringSerializer` in v3.0.0.

## Questions?

For more help, see:
- `TUTORIAL_04_OPENAI.md` - Example using the new JSON serializer
- `examples/tutorial_04_openai_chatbot.py` - Working code example
