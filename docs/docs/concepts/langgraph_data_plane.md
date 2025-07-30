---
search:
  boost: 2
---

# LangGraph Data Plane

The term "data plane" is used broadly to refer to [LangGraph Servers](./langgraph_server.md) (deployments), the corresponding infrastructure for each server, and the "listener" application that continuously polls for updates from the [LangGraph Control Plane](./langgraph_control_plane.md).

## Server Infrastructure

In addition to the [LangGraph Server](./langgraph_server.md) itself, the following infrastructure components for each server are also included in the broad definition of "data plane":

- Postgres
- Redis
- Secrets store
- Autoscalers

## "Listener" Application

The data plane "listener" application periodically calls [control plane APIs](../concepts/langgraph_control_plane.md#control-plane-api) to:

- Determine if new deployments should be created.
- Determine if existing deployments should be updated (i.e. new revisions).
- Determine if existing deployments should be deleted.

In other words, the data plane "listener" reads the latest state of the control plane (desired state) and takes action to reconcile outstanding deployments (current state) to match the latest state.

## Postgres

Postgres is the persistence layer for all user, run, and long-term memory data in a LangGraph Server. This stores both checkpoints (see more info [here](./persistence.md)), server resources (threads, runs, assistants and crons), as well as items saved in the long-term memory store (see more info [here](./persistence.md#memory-store)).

## Redis

Redis is used in each LangGraph Server as a way for server and queue workers to communicate, and to store ephemeral metadata. No user or run data is stored in Redis.

### Communication

All runs in a LangGraph Server are executed by a pool of background workers that are part of each deployment. In order to enable some features for those runs (such as cancellation and output streaming) we need a channel for two-way communication between the server and the worker handling a particular run. We use Redis to organize that communication.

1. A Redis list is used as a mechanism to wake up a worker as soon as a new run is created. Only a sentinel value is stored in this list, no actual run information. The run information is then retrieved from Postgres by the worker.
2. A combination of a Redis string and Redis PubSub channel is used for the server to communicate a run cancellation request to the appropriate worker.
3. A Redis PubSub channel is used by the worker to broadcast streaming output from an agent while the run is being handled. Any open `/stream` request in the server will subscribe to that channel and forward any events to the response as they arrive. No events are stored in Redis at any time.

### Ephemeral metadata

Runs in a LangGraph Server may be retried for specific failures (currently only for transient Postgres errors encountered during the run). In order to limit the number of retries (currently limited to 3 attempts per run) we record the attempt number in a Redis string when it is picked up. This contains no run-specific info other than its ID, and expires after a short delay.

## Data Plane Features

This section describes various features of the data plane.

### Data Region

!!! info "Only for Cloud SaaS"
    Data regions are only applicable for [Cloud SaaS](../concepts/langgraph_cloud.md) deployments.

Deployments can be created in 2 data regions: US and EU

The data region for a deployment is implied by the data region of the LangSmith organization where the deployment is created. Deployments and the underlying database for the deployments cannot be migrated between data regions.

### Autoscaling

[`Production` type](../concepts/langgraph_control_plane.md#deployment-types) deployments automatically scale up to 10 containers. Scaling is based on 3 metrics:

1. CPU utilization
1. Memory utilization
1. Number of pending (in progress) [runs](./assistants.md#execution)

For CPU utilization, the autoscaler targets 75% utilization. This means the autoscaler will scale the number of containers up or down to ensure that CPU utilization is at or near 75%. For memory utilization, the autoscaler targets 75% utilization as well.

For number of pending runs, the autoscaler targets 10 pending runs. For example, if the current number of containers is 1, but the number of pending runs in 20, the autoscaler will scale up the deployment to 2 containers (20 pending runs / 2 containers = 10 pending runs per container).

Each metric is computed independently and the autoscaler will determine the scaling action based on the metric that results in the largest number of containers.

Scale down actions are delayed for 30 minutes before any action is taken. In other words, if the autoscaler decides to scale down a deployment, it will first wait for 30 minutes before scaling down. After 30 minutes, the metrics are recomputed and the deployment will scale down if the recomputed metrics result in a lower number of containers than the current number. Otherwise, the deployment remains scaled up. This "cool down" period ensures that deployments do not scale up and down too frequently.

### Static IP Addresses

!!! info "Only for Cloud SaaS"
Static IP addresses are only available for [Cloud SaaS](../concepts/langgraph_cloud.md) deployments.

All traffic from deployments created after January 6th 2025 will come through a NAT gateway. This NAT gateway will have several static IP addresses depending on the data region. Refer to the table below for the list of static IP addresses:

| US             | EU             |
| -------------- | -------------- |
| 35.197.29.146  | 34.13.192.67   |
| 34.145.102.123 | 34.147.105.64  |
| 34.169.45.153  | 34.90.22.166   |
| 34.82.222.17   | 34.147.36.213  |
| 35.227.171.135 | 34.32.137.113  |
| 34.169.88.30   | 34.91.238.184  |
| 34.19.93.202   | 35.204.101.241 |
| 34.19.34.50    | 35.204.48.32   |

### Custom Postgres

!!! info
Custom Postgres instances are only available for [Self-Hosted Data Plane](../concepts/langgraph_self_hosted_data_plane.md) and [Self-Hosted Control Plane](../concepts/langgraph_self_hosted_control_plane.md) deployments.

A custom Postgres instance can be used instead of the [one automatically created by the control plane](./langgraph_control_plane.md#database-provisioning). Specify the [`POSTGRES_URI_CUSTOM`](../cloud/reference/env_var.md#postgres_uri_custom) environment variable to use a custom Postgres instance.

Multiple deployments can share the same Postgres instance. For example, for `Deployment A`, `POSTGRES_URI_CUSTOM` can be set to `postgres://<user>:<password>@/<database_name_1>?host=<hostname_1>` and for `Deployment B`, `POSTGRES_URI_CUSTOM` can be set to `postgres://<user>:<password>@/<database_name_2>?host=<hostname_1>`. `<database_name_1>` and `database_name_2` are different databases within the same instance, but `<hostname_1>` is shared. **The same database cannot be used for separate deployments**.

### Custom Redis

!!! info
Custom Redis instances are only available for [Self-Hosted Data Plane](../concepts/langgraph_self_hosted_control_plane.md) and [Self-Hosted Control Plane](../concepts/langgraph_self_hosted_control_plane.md) deployments.

A custom Redis instance can be used instead of the one automatically created by the control plane. Specify the [REDIS_URI_CUSTOM](../cloud/reference/env_var.md#redis_uri_custom) environment variable to use a custom Redis instance.

Multiple deployments can share the same Redis instance. For example, for `Deployment A`, `REDIS_URI_CUSTOM` can be set to `redis://<hostname_1>:<port>/1` and for `Deployment B`, `REDIS_URI_CUSTOM` can be set to `redis://<hostname_1>:<port>/2`. `1` and `2` are different database numbers within the same instance, but `<hostname_1>` is shared. **The same database number cannot be used for separate deployments**.

### LangSmith Tracing

LangGraph Server is automatically configured to send traces to LangSmith. See the table below for details with respect to each deployment option.

| Cloud SaaS                               | Self-Hosted Data Plane                                      | Self-Hosted Control Plane                                          | Standalone Container                                                                         |
| ---------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| Required<br><br>Trace to LangSmith SaaS. | Optional<br><br>Disable tracing or trace to LangSmith SaaS. | Optional<br><br>Disable tracing or trace to Self-Hosted LangSmith. | Optional<br><br>Disable tracing, trace to LangSmith SaaS, or trace to Self-Hosted LangSmith. |

### Telemetry

LangGraph Server is automatically configured to report telemetry metadata for billing purposes. See the table below for details with respect to each deployment option.

| Cloud SaaS                        | Self-Hosted Data Plane            | Self-Hosted Control Plane                                                                                                           | Standalone Container                                                                                                                |
| --------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Telemetry sent to LangSmith SaaS. | Telemetry sent to LangSmith SaaS. | Self-reported usage (audit) for air-gapped license key.<br><br>Telemetry sent to LangSmith SaaS for LangGraph Platform License Key. | Self-reported usage (audit) for air-gapped license key.<br><br>Telemetry sent to LangSmith SaaS for LangGraph Platform License Key. |

### Licensing

LangGraph Server is automatically configured to perform license key validation. See the table below for details with respect to each deployment option.

| Cloud SaaS                                          | Self-Hosted Data Plane                              | Self-Hosted Control Plane                                                                  | Standalone Container                                                                       |
| --------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| LangSmith API Key validated against LangSmith SaaS. | LangSmith API Key validated against LangSmith SaaS. | Air-gapped license key or LangGraph Platform License Key validated against LangSmith SaaS. | Air-gapped license key or LangGraph Platform License Key validated against LangSmith SaaS. |
