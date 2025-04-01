# LangGraph Data Plane

The term "data plane" is used broadly to refer to [LangGraph Server](./langgraph_server.md) instances (deployments), the corresponding database (Postgres) for each deployment, and the "listener" application that continuously polls for updates from the [LangGraph Control Plane](./langgraph_control_plane.md).

The data plane "listener" application periodically calls LangGraph Control Plane APIs to:

- Determine if new deployments should be created.
- Determine if existing deployments should be updated (i.e. new revisions).
- Determine if existing deployments should be deleted.

In other words, the data plane "listener" reads the latest state of the control plane (desired state) and takes action to reconcile outstanding deployments (current state) to match the latest state.
