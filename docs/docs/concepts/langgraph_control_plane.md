# LangGraph Control Plane

The term "control plane" is used broadly to refer to the Control Plane UI where users create and update [LangGraph Server](./langgraph_server.md) instances (deployments) and the Control Plane APIs that support the UI experience.

When a user makes an update through the Control Plane UI, the update is stored in the control plane state. The [LangGraph Data Plane](./langgraph_data_plane.md) "listener" application polls for these updates by calling the Control Plane APIs.

## Control Plane UI

From the Control Plane UI, a user can:

- View a list of outstanding deployments.
- View details of an individual deployment.
- Create a new deployment.
- Update an existing deployment.
- Update environment variables for an existing deployment.
- Delete an existing deployment.

The Control Plane UI is embeded in [LangSmith](https://docs.smith.langchain.com/langgraph_cloud).

## Control Plane API

This section describes concepts of the LangGraph Control Plane API. LangGraph Control Plane APIs are used to create, update, and delete deployments. However, they are not publicly accessible.

### Deployment

A deployment is an instance of a LangGraph Server. A single deployment can have many revisions.

### Revision

A revision is an iteration of a deployment. When a new deployment is created, an initial revision is automatically created. To deploy code changes or update environment variables for a deployment, a new revision must be created.

### Environment Variable

Environment variables are set for a deployment. All environment variables are stored as secrets (i.e. saved in a secrets store).
