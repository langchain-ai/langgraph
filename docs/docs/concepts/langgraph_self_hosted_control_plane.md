---
search:
  boost: 2
---

# Self-Hosted Control Plane (Beta)

To deploy a [LangGraph Server](../concepts/langgraph_server.md), follow the how-to guide for [how to deploy the Self-Hosted Control Plane](../cloud/deployment/self_hosted_control_plane.md).

## Overview

The Self-Hosted Control Plane deployment option is a fully self-hosted model for deployment where you manage the [control plane](./langgraph_control_plane.md) and [data plane](./langgraph_data_plane.md) in your cloud (this option implies that the data plane is self-hosted).

|                   | [Control Plane](../concepts/langgraph_control_plane.md) | [Data Plane](../concepts/langgraph_data_plane.md) |
|-------------------|-------------------|------------|
| **What is it?** | <ul><li>Control Plane UI for creating deployments and revisions</li><li>Control Plane APIs for creating deployments and revisions</li></ul> | <ul><li>Data plane "listener" for reconciling deployments with control plane state</li><li>LangGraph Servers</li><li>Postgres, Redis, etc</li></ul> |
| **Where is it hosted?** | Your cloud | Your cloud |
| **Who provisions and manages it?** | You | You |

## Architecture

![Self-Hosted Control Plane Architecture](./img/self_hosted_control_plane_architecture.png)

## Compute Platforms

### Kubernetes

The Self-Hosted Control Plane deployment option supports deploying control plane and data plane infrastructure to any Kubernetes cluster.
