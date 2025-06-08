---
search:
  boost: 2
---

# LangGraph Control Plane

The term "control plane" is used broadly to refer to the control plane UI where users create and update [LangGraph Servers](./langgraph_server.md) (deployments) and the control plane APIs that support the UI experience.

When a user makes an update through the control plane UI, the update is stored in the control plane state. The [LangGraph Data Plane](./langgraph_data_plane.md) "listener" application polls for these updates by calling the control plane APIs.

## Control Plane UI

From the control plane UI, you can:

- View a list of outstanding deployments.
- View details of an individual deployment.
- Create a new deployment.
- Update a deployment.
- Update environment variables for a deployment.
- View build and server logs of a deployment.
- Delete a deployment.

The Control Plane UI is embedded in [LangSmith](https://docs.smith.langchain.com/langgraph_cloud).

## Control Plane API

This section describes data model of the control plane API. The API is used to create, update, and delete deployments. However, they are not publicly accessible.

### Deployment

A deployment is an instance of a LangGraph Server. A single deployment can have many revisions.

### Revision

A revision is an iteration of a deployment. When a new deployment is created, an initial revision is automatically created. To deploy code changes or update environment variables for a deployment, a new revision must be created.

### Environment Variable

Environment variables are set for a deployment. All environment variables are stored as secrets (i.e. saved in a secrets store).

## Control Plane Features

This section describes various features of the control plane.

### Deployment Types

For simplicity, the control plane offers two deployment types with different resource allocations: `Development` and `Production`.

| **Deployment Type** | **CPU/Memory**  | **Scaling**         | **Database**                                                                     |
|---------------------|-----------------|---------------------|----------------------------------------------------------------------------------|
| Development         | 1 CPU, 1 GB RAM | Up to 1 container   | 10 GB disk, no backups                                                           |
| Production          | 2 CPU, 2 GB RAM | Up to 10 containers | Autoscaling disk, automatic backups, highly available (multi-zone configuration) |

CPU and memory resources are per container.

!!! warning "Immutable Deployment Type"

    Once a deployment is created, the deployment type cannot be changed.

!!! info "Resource Customization"
    For `Production` type deployments, resources can be manually increased on a case-by-case basis depending on use case and capacity constraints. Contact support@langchain.dev to request an increase in resources.

    For `Development` types deployments, database disk size can be manually increased on a case-by-case basis depending on use case and capacity constraints. For most use cases, [TTLs](../how-tos/ttl/configure_ttl.md) should be configured to manage disk usage. Contact support@langchain.dev to request an increase in resources.

    Resources for [Self-Hosted Data Plane](../concepts/langgraph_self_hosted_data_plane.md) and [Self-Hosted Control Plane](../concepts/langgraph_self_hosted_control_plane.md) deployments can be fully customized.

### Database Provisioning

The control plane and [LangGraph Data Plane](./langgraph_data_plane.md) "listener" application coordinate to automatically create a Postgres database for each deployment. The database serves as the [persistence layer](../concepts/persistence.md) for the deployment.

When implementing a LangGraph application, a [checkpointer](../concepts/persistence.md#checkpointer-libraries) does not need to be configured by the developer. Instead, a checkpointer is automatically configured for the graph. Any checkpointer configured for a graph will be replaced by the one that is automatically configured.

There is no direct access to the database. All access to the database occurs through the [LangGraph Server](../concepts/langgraph_server.md).

The database is never deleted until the deployment itself is deleted.

!!! info
    A custom Postgres instance can be configured for [Self-Hosted Data Plane](../concepts/langgraph_self_hosted_data_plane.md) and [Self-Hosted Control Plane](../concepts/langgraph_self_hosted_control_plane.md) deployments.

### Asynchronous Deployment

Infrastructure for deployments and revisions are provisioned and deployed asynchronously. They are not deployed immediately after submission. Currently, deployment can take up to several minutes.

- When a new deployment is created, a new database is created for the deployment. Database creation is a one-time step. This step contributes to a longer deployment time for the initial revision of the deployment.
- When a subsequent revision is created for a deployment, there is no database creation step. The deployment time for a subsequent revision is significantly faster compared to the deployment time of the initial revision.
- The deployment process for each revision contains a build step, which can take up to a few minutes.

The control plane and [LangGraph Data Plane](./langgraph_data_plane.md) "listener" application coordinate to achieve asynchronous deployments.

### LangSmith Integration

A [LangSmith](https://docs.smith.langchain.com/) tracing project is automatically created for each deployment. The tracing project has the same name as the deployment. When creating a deployment, the `LANGCHAIN_TRACING` and `LANGSMITH_API_KEY`/`LANGCHAIN_API_KEY` environment variables do not need to be specified; they are set automatically by the control plane.

When a deployment is deleted, the traces and the tracing project are not deleted.
