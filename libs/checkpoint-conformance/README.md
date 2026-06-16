# LangGraph Checkpoint Conformance

[![PyPI - Version](https://img.shields.io/pypi/v/langgraph-checkpoint-conformance?label=%20)](https://pypi.org/project/langgraph-checkpoint-conformance/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langgraph-checkpoint-conformance)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langgraph-checkpoint-conformance)](https://pypistats.org/packages/langgraph-checkpoint-conformance)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

To help you ship LangGraph apps to production faster, check out [LangSmith](https://www.langchain.com/langsmith).
[LangSmith](https://www.langchain.com/langsmith) is a unified developer platform for building, testing, and monitoring LLM applications.

## Quick Install

```bash
uv add langgraph-checkpoint-conformance
```

## 🤔 What is this?

This library provides a conformance test suite for [LangGraph](https://github.com/langchain-ai/langgraph) checkpointer implementations. It validates that a `BaseCheckpointSaver` subclass correctly implements the checkpoint storage contract — blob round-trips, metadata preservation, namespace isolation, incremental channel updates, and more.

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/langgraph/). For conceptual guides on persistence and memory, see the [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/overview).

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
| `delta_channel_history` | no | `aget_delta_channel_history` |

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

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
