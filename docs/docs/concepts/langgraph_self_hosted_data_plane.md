---
search:
  boost: 2
---

# Self-Hosted Data Plane (Beta)

To deploy a [LangGraph Server](../concepts/langgraph_server.md), follow the how-to guide for [how to deploy the Self-Hosted Data Plane](../cloud/deployment/self_hosted_data_plane.md).

## Overview

LangGraph Platform's Self-Hosted Data Plane deployment option is a "hybrid" model for deployment where we manage the [control plane](./langgraph_control_plane.md) in our cloud and you manage the [data plane](./langgraph_data_plane.md) in your cloud.

|                   | [Control Plane](../concepts/langgraph_control_plane.md) | [Data Plane](../concepts/langgraph_data_plane.md) |
|-------------------|-------------------|------------|
| **What is it?** | <ul><li>Control Plane UI for creating deployments and revisions</li><li>Control Plane APIs for creating deployments and revisions</li></ul> | <ul><li>Data plane "listener" for reconciling deployments with control plane state</li><li>LangGraph Servers</li><li>Postgres, Redis, etc</li></ul> |
| **Where is it hosted?** | LangChain's cloud | Your cloud |
| **Who provisions and manages it?** | LangChain | You |

## Architecture

![Self-Hosted Data Plane Architecture](./img/self_hosted_data_plane_architecture.png)

## Compute Platforms

### Kubernetes

The Self-Hosted Data Plane deployment option supports deploying data plane infrastructure to any Kubernetes cluster.

### Amazon ECS

Coming soon...
