# Self-Hosted Data Plane

## Overview

LangGraph Platform's Self-Hosted Data Plane deployment option is a "hybrid" model for deployemnt where we manage the control plane (e.g. LangGraph Platform UI, deployment APIs) in our cloud and you manage the data plane (e.g. containers, Postgres, Redis) in your cloud.

|                   | Control Plane     | Data Plane |
|-------------------|-------------------|------------|
| **What is it?** | <ul><li>LangGraph Platform UI for creating deployments and revisions</li><li>Control plane APIs for creating deployments and revisions</li></ul> | <ul><li>Data plane listene for reconciling deployments with control plane state</li><li>LangGraph server container(s)</li><li>Postgres, Redis</li></ul> |
| **Where it's hosted?** | LangChain's cloud | Your cloud |
| **Who provisions and manages it?** | LangChain | You |

## Architecture

![Self-Hosted Data Plane Architecture](./img/self_hosted_data_plane_architecture.png)

## Compute Platforms

### Kubernetes

The Self-Hosted Data Plane deployment option supports deploying data plane infrastructure to any Kubernetes cluster.

### Amazon ECS

Coming soon...

## Miscellaneous

Miscellaneous options/details about the Self-Hosted Data Plane deployment option.

|                   | Description |
|-------------------|-------------|
| **Lite vs Enterprise** | Enterprise |
| **Tracing** | Trace to LangSmith SaaS |
| **Licensing** | LangSmith API Key validated against LangSmith SaaS |
| **Telemetry** | Telemetry sent to LangSmith SaaS |

## Related

* [How to deploy the Self-Hosted Data Plane](../cloud/deployment/self_hosted_data_plane.md)
