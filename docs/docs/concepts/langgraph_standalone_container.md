---
search:
  boost: 2
---

# Standalone Container

To deploy a [LangGraph Server](../concepts/langgraph_server.md), follow the how-to guide for [how to deploy a Standalone Container](../cloud/deployment/standalone_container.md).

## Overview

The Standalone Container deployment option is the least restrictive model for deployment. There is no [control plane](./langgraph_control_plane.md). [Data plane](./langgraph_data_plane.md) infrastructure is managed by you.

|                   | [Control plane](../concepts/langgraph_control_plane.md) | [Data plane](../concepts/langgraph_data_plane.md) |
|-------------------|-------------------|------------|
| **What is it?** | n/a | <ul><li>LangGraph Servers</li><li>Postgres, Redis, etc</li></ul> |
| **Where is it hosted?** | n/a | Your cloud |
| **Who provisions and manages it?** | n/a | You |

!!! warning

      LangGraph Platform should not be deployed in serverless environments. Scale to zero may cause task loss and scaling up will not work reliably.

## Architecture

![Standalone Container](./img/langgraph_platform_deployment_architecture.png)

## Compute Platforms

### Kubernetes

The Standalone Container deployment option supports deploying data plane infrastructure to a Kubernetes cluster.

### Docker

The Standalone Container deployment option supports deploying data plane infrastructure to any Docker-supported compute platform.
