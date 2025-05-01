---
search:
  boost: 2
---

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

[`Production` type](../concepts/langgraph_control_plane.md#deployment-types) deployments automatically scale up to 10 containers. Scaling is based on 3 metrics:

1. CPU utilization
1. Memory utilization
1. Number of pending (in progress) [runs](../concepts/langgraph_server.md#runs)

For CPU utilization, the autoscaler targets 75% utilization. This means the autoscaler will scale the number of containers up or down to ensure that CPU utilization is at or near 75%. For memory utilization, the autoscaler targets 75% utilization as well.

For number of pending runs, the autoscaler targets 10 pending runs. For example, if the current number of containers is 1, but the number of pending runs in 20, the autoscaler will scale up the deployment to 2 containers (20 pending runs / 2 containers = 10 pending runs per container).

Each metric is computed independently and the autoscaler will determine the scaling action based on the metric that results in the most number of containers.

Scale down actions are delayed for 30 minutes before any action is taken. In other words, if the autoscaler decides to scale down a deployment, it will first wait for 30 minutes before scaling down. After 30 minutes, the metrics are recomputed and the deployment will scale down if the recomputed metrics result in a lower number of containers than the current number. Otherwise, the deployment remains scaled up. This "cool down" period ensures that deployments do not scale up and down too frequently.

### Static IP Addresses

!!! info "Only for Cloud SaaS"
    Static IP addresses are only available for [Cloud SaaS](../concepts/langgraph_cloud.md) deployments.

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

### Custom Postgres

!!! info "Only for Self-Hosted Data Plane and Self-Hosted Control Plane"
    Custom Postgres instances are only available for [Self-Hosted Data Plane](../concepts/langgraph_self_hosted_data_plane.md) and [Self-Hosted Control Plane](../concepts/langgraph_self_hosted_control_plane.md) deployments.

A custom Postgres instance can be used instead of the [one automatically created by the control plane](./langgraph_control_plane.md#database-provisioning). Specify the [`POSTGRES_URI_CUSTOM`](../cloud/reference/env_var.md#postgres_uri_custom) environment variable to use a custom Postgres instance.

Multiple deployments can share the same Postgres instance. For example, for `Deployment A`, `POSTGRES_URI_CUSTOM` can be set to `postgres://<user>:<password>@/<database_name_1>?host=<hostname_1>` and for `Deployment B`, `POSTGRES_URI_CUSTOM` can be set to `postgres://<user>:<password>@/<database_name_2>?host=<hostname_1>`. `<database_name_1>` and `database_name_2` are different databases within the same instance, but `<hostname_1>` is shared. **The same database cannot be used for separate deployments**.

### Custom Redis

!!! info "Only for Self-Hosted Data Plane and Self-Hosted Control Plane"
    Custom Redis instances are only available for [Self-Hosted Data Plane](../concepts/langgraph_self_hosted_data_plane.md) and [Self-Hosted Control Plane](../concepts/langgraph_self_hosted_control_plane.md) deployments.

A custom Redis instance can be used instead of the one automatically created by the control plane. Specify the [REDIS_URI_CUSTOM](../cloud/reference/env_var.md#redis_uri_custom) environment variable to use a custom Redis instance.


Multiple deployments can share the same Redis instance. For example, for `Deployment A`, `REDIS_URI_CUSTOM` can be set to `redis://<hostname_1>:<port>/1` and for `Deployment B`, `REDIS_URI_CUSTOM` can be set to `redis://<hostname_1>:<port>/2`. `1` and `2` are different database numbers within the same instance, but `<hostname_1>` is shared. **The same database number cannot be used for separate deployments**.

### LangSmith Tracing

LangGraph Server is automatically configured to send traces to LangSmith. See the table below for details with respect to each deployment option.

| Cloud SaaS | Self-Hosted Data Plane | Self-Hosted Control Plane | Standalone Container |
|------------|------------------------|---------------------------|----------------------|
| Required<br><br>Trace to LangSmith SaaS. | Optional<br><br>Disable tracing or trace to LangSmith SaaS. | Optional<br><br>Disable tracing or trace to Self-Hosted LangSmith. | Optional<br><br>Disable tracing, trace to LangSmith SaaS, or trace to Self-Hosted LangSmith. |

### Telemetry

LangGraph Server is automatically configured to report telemetry metadata for billing purposes. See the table below for details with respect to each deployment option.

| Cloud SaaS | Self-Hosted Data Plane | Self-Hosted Control Plane | Standalone Container |
|------------|------------------------|---------------------------|----------------------|
| Telemetry sent to LangSmith SaaS. | Telemetry sent to LangSmith SaaS. | Self-reported usage (audit) for air-gapped license key.<br><br>Telemetry sent to LangSmith SaaS for LangGraph Platform License Key. | Self-reported usage (audit) for air-gapped license key.<br><br>Telemetry sent to LangSmith SaaS for LangGraph Platform License Key. |

### Licensing

LangGraph Server is automatically configured to perform license key validation. See the table below for details with respect to each deployment option.

| Cloud SaaS | Self-Hosted Data Plane | Self-Hosted Control Plane | Standalone Container |
|------------|------------------------|---------------------------|----------------------|
| LangSmith API Key validated against LangSmith SaaS. | LangSmith API Key validated against LangSmith SaaS. | Air-gapped license key or LangGraph Platform License Key validated against LangSmith SaaS. | Air-gapped license key or LangGraph Platform License Key validated against LangSmith SaaS. |
