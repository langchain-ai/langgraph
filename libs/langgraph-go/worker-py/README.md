# LangGraph Worker Python gRPC Server

This directory contains a Python implementation of the gRPC server defined in `server.proto`. The server implements the `Worker` service which provides methods for streaming nodes and invoking reducers.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Compile the Protocol Buffer definition to generate Python code:

```bash
python compile_proto.py
```

This will generate the necessary Python modules in the `stubs` directory.

## Server Implementation

The server implementation is in `grpc_server.py`. It provides:

- A `WorkerServicer` class that implements the `Worker` service defined in the proto file
- Methods to register handlers for nodes and reducers
- Helper methods to create write and error events

## Running the Server

To run the server:

```bash
python grpc_server.py [port]
```

By default, the server listens on port 50051.

## Customizing the Server

To customize the server behavior, modify the `register_handlers` function in `grpc_server.py` to register your own node and reducer handlers.

Example:

```python
def register_handlers(servicer: WorkerServicer):
    # Custom node handler
    def my_node_handler(inputs, config, path):
        # Process inputs and return results
        return {"output": b"Processed result"}
    
    # Register the handler
    servicer.register_node_handler("my_node", my_node_handler)
```

## Protocol Buffer Definition

The Protocol Buffer definition in `server.proto` defines:

- `Config`: Configuration for checkpoints
- `PregelExecutableTask`: Task information for execution
- `Event`: Output events (write or error)
- `Worker` service: Service with methods for streaming nodes and invoking reducers
