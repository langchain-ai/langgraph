# LangGraph CLI

The official command-line interface for LangGraph, providing tools to create, develop, and deploy LangGraph applications.

## Installation

Install via pip:
```bash
pip install langgraph-cli
```

For development mode with hot reloading:
```bash
pip install "langgraph-cli[inmem]"
```

## Commands

### `langgraph new` 🌱
Create a new LangGraph project from a template
```bash
langgraph new [PATH] --template TEMPLATE_NAME
```

### `langgraph dev` 🏃‍♀️
Run LangGraph API server in development mode with hot reloading
```bash
langgraph dev [OPTIONS]
  --host TEXT                 Host to bind to (default: 127.0.0.1)
  --port INTEGER             Port to bind to (default: 2024)
  --no-reload               Disable auto-reload
  --debug-port INTEGER      Enable remote debugging
  --no-browser             Skip opening browser window
  -c, --config FILE        Config file path (default: langgraph.json)
```

### `langgraph up` 🚀
Launch LangGraph API server in Docker
```bash
langgraph up [OPTIONS]
  -p, --port INTEGER        Port to expose (default: 8123)
  --wait                   Wait for services to start
  --watch                  Restart on file changes
  --verbose               Show detailed logs
  -c, --config FILE       Config file path
  -d, --docker-compose    Additional services file
```

### `langgraph build`
Build a Docker image for your LangGraph application
```bash
langgraph build -t IMAGE_TAG [OPTIONS]
  --platform TEXT          Target platforms (e.g., linux/amd64,linux/arm64)
  --pull / --no-pull      Use latest/local base image
  -c, --config FILE       Config file path
```

### `langgraph dockerfile`
Generate a Dockerfile for custom deployments
```bash
langgraph dockerfile SAVE_PATH [OPTIONS]
  -c, --config FILE       Config file path
```

## Configuration

The CLI uses a `langgraph.json` configuration file with these key settings:

```json
{
  "dependencies": ["langchain_openai", "./your_package"],  // Required: Package dependencies
  "graphs": {
    "my_graph": "./your_package/file.py:graph"            // Required: Graph definitions
  },
  "env": "./.env",                                        // Optional: Environment variables
  "python_version": "3.11",                               // Optional: Python version (3.11/3.12)
  "pip_config_file": "./pip.conf",                        // Optional: pip configuration
  "dockerfile_lines": []                                  // Optional: Additional Dockerfile commands
}
```

See the [full documentation](https://langchain-ai.github.io/langgraph/cloud/reference/cli/) for detailed configuration options.

## Development

To develop the CLI itself:

1. Clone the repository
2. Navigate to the CLI directory: `cd libs/cli`
3. Install development dependencies: `uv pip install`
4. Make your changes to the CLI code
5. Test your changes:
   ```bash
   # Run CLI commands directly
   uv run langgraph --help
   
   # Or use the examples
   cd examples
   uv pip install
   uv run langgraph dev  # or other commands
   ```

## License

This project is licensed under the terms specified in the repository's LICENSE file.
