## Components

The LangGraph Platform consists of components that work together to support the development, deployment, debugging, and monitoring of LangGraph applications:

- [LangGraph Server](./langgraph_server.md): The server defines an opinionated API and architecture that incorporates best practices for deploying agentic applications, allowing you to focus on building your agent logic rather than developing server infrastructure.
- [LangGraph CLI](./langgraph_cli.md): LangGraph CLI is a command-line interface that helps to interact with a local LangGraph
- [LangGraph Studio](./langgraph_studio.md): LangGraph Studio is a specialized IDE that can connect to a LangGraph Server to enable visualization, interaction, and debugging of the application locally.
- [Python/JS SDK](./sdk.md): The Python/JS SDK provides a programmatic way to interact with deployed LangGraph Applications.
- [Remote Graph](../how-tos/use-remote-graph.md): A RemoteGraph allows you to interact with any deployed LangGraph application as though it were running locally.
- [LangGraph control plane](./langgraph_control_plane.md): The LangGraph Control Plane refers to the Control Plane UI where users create and update LangGraph Servers and the Control Plane APIs that support the UI experience.
- [LangGraph data plane](./langgraph_data_plane.md): The LangGraph Data Plane refers to LangGraph Servers, the corresponding infrastructure for each server, and the "listener" application that continuously polls for updates from the LangGraph Control Plane.

![LangGraph components](img/lg_platform.png)
