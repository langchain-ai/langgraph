# LangGraph Data Plane

The term "data plane" is used broadly to refer to [LangGraph Servers](./langgraph_server.md) (deployments), the corresponding infrastructure for each server, and the "listener" application that continuously polls for updates from the [LangGraph Control Plane](./langgraph_control_plane.md).

## Server Infrastructure

In addition to the [LangGraph Server](./langgraph_server.md) itself, the following infrastructure for each server are also included in the broad defintion of "data plane":

- [Postgres](../concepts/platform_architecture.md#how-we-use-postgres)
- [Redis](../concepts/platform_architecture.md#how-we-use-redis)
- Secrets store
- Autoscalers

See [LangGraph Platform Architecture](../concepts/platform_architecture.md) for more details.

## "Listener" Application

The data plane "listener" application periodically calls [Control Plane APIs](../concepts/langgraph_control_plane.md#control-plane-api) to:

- Determine if new deployments should be created.
- Determine if existing deployments should be updated (i.e. new revisions).
- Determine if existing deployments should be deleted.

In other words, the data plane "listener" reads the latest state of the control plane (desired state) and takes action to reconcile outstanding deployments (current state) to match the latest state.
