# LangGraph Data Plane

The term "data plane" is used broadly to refer to [LangGraph Servers](./langgraph_server.md) (deployments), the corresponding infrastructure for each server, and the "listener" application that continuously polls for updates from the [LangGraph Control Plane](./langgraph_control_plane.md).

## Server Infrastructure

In addition to the [LangGraph Server](./langgraph_server.md) itself, the following infrastructure for each server are also included in the broad definition of "data plane":

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

## Data Plane Features

This section describes various features of the data plane.

### Lite vs Enterprise

There are two versions of the LangGraph Server: `Lite` and `Enterprise`.

The `Lite` version is a limited version of the LangGraph Server that you can run locally or in a self-hosted manner (up to 1 million nodes executed per year). `Lite` is only available for the [Standalone Container](../concepts/langgraph_standalone_container.md) deployment option.

The `Enterprise` version is the full version of the LangGraph Server. To use the `Enterprise` version, you must acquire a license key that you will need to specify when running the Docker image. To acquire a license key, please email sales@langchain.dev. `Enterprise` is available for [Cloud SaaS](../concepts/langgraph_cloud.md), [Self-Hosted Data Plane](../concepts/langgraph_self_hosted_data_plane.md), and [Self-Hosted Control Plane](../concepts/langgraph_self_hosted_control_plane.md) deployment options.

Feature Differences:

|       | Lite       | Enterprise |
|-------|------------|------------|
| [Cron Jobs](../concepts/langgraph_server.md#cron-jobs) |❌|✅|
| [Custom Authentication](../concepts/auth.md) |❌|✅|

### Autoscaling

[`Production` type](../concepts/langgraph_control_plane.md#deployment-types) deployments automatically scale up to 10 containers. Scaling is based on the current request load for a single container. Specifically, the autoscaling implementation scales the deployment so that each container is processing about 10 concurrent requests. For example... 

- If the deployment is processing 20 concurrent requests, the deployment will scale up from 1 container to 2 containers (20 requests / 2 containers = 10 requests per container).
- If a deployment of 2 containers is processing 10 requests, the deployment will scale down from 2 containers to 1 container (10 requests / 1 container = 10 requests per container).

10 concurrent requests per container is the target threshold. However, 10 concurrent requests per container is not a hard limit. The number of concurrent requests can exceed 10 if there is a sudden burst of requests.

Scale down actions are delayed for 30 minutes before any action is taken. In other words, if the autoscaling implementation decides to scale down a deployment, it will first wait for 30 minutes before scaling down. After 30 minutes, the concurrency metric is recomputed and the deployment will scale down if the concurrency metric has met the target threshold. Otherwise, the deployment remains scaled up. This "cool down" period ensures that deployments do not scale up and down too frequently.

In the future, the autoscaling implementation may evolve to accommodate other metrics such as background run queue size.

### Static IP Addresses

!!! info "Only for Cloud SaaS"
    Static IP addresses are only available for [Cloud SaaS](../concepts/langgraph_cloud.md).

All traffic from deployments created after January 6th 2025 will come through a NAT gateway. This NAT gateway will have several static IP addresses depending on the data region. Refer to the table below for the list of static IP addresses:

| US             | EU             |
|----------------|----------------|
| 35.197.29.146  | 34.13.192.67   |
| 34.145.102.123 | 34.147.105.64  |
| 34.169.45.153  | 34.90.22.166   |
| 34.82.222.17   | 34.147.36.213  |
| 35.227.171.135 | 34.32.137.113  | 
| 34.169.88.30   | 34.91.238.184  |
| 34.19.93.202   | 35.204.101.241 |
| 34.19.34.50    | 35.204.48.32   |
