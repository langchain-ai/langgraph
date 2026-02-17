# langgraph-checkpoint-conformance

Conformance test suite for [LangGraph](https://github.com/langchain-ai/langgraph) checkpointer implementations.

Validates that a `BaseCheckpointSaver` subclass correctly implements the checkpoint storage contract — blob round-trips, metadata preservation, namespace isolation, incremental channel updates, and more.

## Installation

```bash
pip install langgraph-checkpoint-conformance
```

## Quick start

Register your checkpointer with `@checkpointer_test` and run `validate()`:

```python
import asyncio
from langgraph.checkpoint.conformance import checkpointer_test, validate

@checkpointer_test(name="MyCheckpointer")
async def my_checkpointer():
    saver = MyCheckpointer(...)
    yield saver
    # cleanup runs after yield

async def main():
    report = await validate(my_checkpointer)
    report.print_report()
    assert report.passed_all_base()

asyncio.run(main())
```

Or in a pytest test:

```python
import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate

@checkpointer_test(name="MyCheckpointer")
async def my_checkpointer():
    yield MyCheckpointer(...)

@pytest.mark.asyncio
async def test_conformance():
    report = await validate(my_checkpointer)
    report.print_report()
    assert report.passed_all_base()
```

## Capabilities

The suite tests **base** capabilities (required) and **extended** capabilities (optional, auto-detected):

| Capability | Required | Method |
|---|---|---|
| `put` | yes | `aput` |
| `put_writes` | yes | `aput_writes` |
| `get_tuple` | yes | `aget_tuple` |
| `list` | yes | `alist` |
| `delete_thread` | yes | `adelete_thread` |
| `delete_for_runs` | no | `adelete_for_runs` |
| `copy_thread` | no | `acopy_thread` |
| `prune` | no | `aprune` |

Extended capabilities are detected by checking whether the method is overridden from `BaseCheckpointSaver`. If not overridden, those tests are skipped.

## Options

### Progress output

```python
from langgraph.checkpoint.conformance.report import ProgressCallbacks

# Dot-style progress (. per pass, F per fail)
report = await validate(my_checkpointer, progress=ProgressCallbacks.default())

# Verbose (per-test names + stacktraces on failure)
report = await validate(my_checkpointer, progress=ProgressCallbacks.verbose())
```

### Skip capabilities

```python
@checkpointer_test(name="MyCheckpointer", skip_capabilities={"prune"})
async def my_checkpointer():
    yield MyCheckpointer(...)
```

### Run specific capabilities

```python
report = await validate(my_checkpointer, capabilities={"put", "list"})
```

### Lifespan (one-time setup/teardown)

For expensive setup like database creation:

```python
async def db_lifespan():
    await create_database()
    yield
    await drop_database()

@checkpointer_test(name="PostgresSaver", lifespan=db_lifespan)
async def pg_checkpointer():
    async with PostgresSaver.from_conn_string(CONN_STRING) as saver:
        yield saver
```
