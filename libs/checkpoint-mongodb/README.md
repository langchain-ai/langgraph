# langgraph-checkpoint-mongodb

`langgraph-checkpoint-mongodb` is a library that provides MongoDB-based checkpoint saving functionality for LangGraph. It allows saving and retrieving checkpoints in MongoDB databases asynchronously or synchronously, enabling efficient state tracking in LangGraph-based applications.

## Installation

To install the library, you can use `pip`:

```bash
pip install langgraph-checkpoint-mongodb
```

## Usage

### 1. Asynchronous Checkpoint Saver

The `AsyncMongoDBSaver` allows saving and retrieving checkpoints asynchronously. Here's an example:

```python
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: Literal["nyc", "sf"]):
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

tools = [get_weather]  # List of tools to be used by the agent
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key="your_api_key")

# Create an AsyncMongoDBSaver instance and use it as a checkpointer
async with AsyncMongoDBSaver.from_conn_info(
    url="mongodb://localhost:27017", db_name="checkpoints"
) as checkpointer:

    # Create a React agent using the model and tools, along with the MongoDB checkpointer
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)

    # Example configuration for the agent's execution
    config = {"configurable": {"thread_id": "1"}}

    # Invoke the agent with a query
    res = await graph.ainvoke({"messages": [("human", "what's the weather in sf")]}, config)

    # Retrieve the latest checkpoint
    latest_checkpoint = await checkpointer.get(config)
    latest_checkpoint_tuple = await checkpointer.get_tuple(config)

    # List all checkpoint tuples
    checkpoint_tuples = await checkpointer.list(config)
```

### 2. Synchronous Checkpoint Saver

The `MongoDBSaver` allows saving and retrieving checkpoints synchronously. Here's an example:

```python
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: Literal["nyc", "sf"]):
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

tools = [get_weather]  # List of tools to be used by the agent
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key="your_api_key")

# Create a MongoDBSaver instance and use it as a checkpointer
with MongoDBSaver.from_conn_info(
    url="mongodb://localhost:27017", db_name="checkpoints"
) as checkpointer:

    # Create a React agent using the model and tools, along with the MongoDB checkpointer
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)

    # Example configuration for the agent's execution
    config = {"configurable": {"thread_id": "1"}}

    # Invoke the agent with a query
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    # Retrieve the latest checkpoint
    latest_checkpoint = checkpointer.get(config)
    latest_checkpoint_tuple = checkpointer.get_tuple(config)

    # List all checkpoint tuples
    checkpoint_tuples = list(checkpointer.list(config))
```

## Configuration

### `AsyncMongoDBSaver` Configuration

- `url`: The connection URL to the MongoDB database (e.g., `"mongodb://localhost:27017"`).
- `db_name`: The name of the database where checkpoints will be stored.

### `MongoDBSaver` Configuration

- `url`: The connection URL to the MongoDB database (e.g., `"mongodb://localhost:27017"`).
- `db_name`: The name of the database where checkpoints will be stored.

## Development

To contribute to the development of this library, you can set up the development environment using Poetry:

```bash
poetry install
```

Run tests using `pytest`:

```bash
poetry run pytest
```

## License

This library is licensed under the MIT License.

## Example of Checkpointing Flow

1. The agent performs a task (e.g., answering a question) and produces a result.
2. The checkpoint is saved in MongoDB after each task execution.
3. The agent can later retrieve the checkpoint to resume its state, making the system resilient to failures.

For further information on LangGraph or to explore other tools, refer to the [LangGraph repository](https://www.github.com/langchain-ai/langgraph).
