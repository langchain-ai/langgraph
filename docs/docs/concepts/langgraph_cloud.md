---
search:
  boost: 2
---

# Cloud SaaS

To deploy a [LangGraph Server](../concepts/langgraph_server.md), follow the how-to guide for [how to deploy to Cloud SaaS](../cloud/deployment/cloud.md).

## Overview

The Cloud SaaS deployment option is a fully managed model for deployment where we manage the [control plane](./langgraph_control_plane.md) and [data plane](./langgraph_data_plane.md) in our cloud.

|                                    | [Control plane](../concepts/langgraph_control_plane.md)                                                                                     | [Data plane](../concepts/langgraph_data_plane.md)                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What is it?**                    | <ul><li>Control plane UI for creating deployments and revisions</li><li>Control plane APIs for creating deployments and revisions</li></ul> | <ul><li>Data plane "listener" for reconciling deployments with control plane state</li><li>LangGraph Servers</li><li>Postgres, Redis, etc</li></ul> |
| **Where is it hosted?**            | LangChain's cloud                                                                                                                           | LangChain's cloud                                                                                                                                   |
| **Who provisions and manages it?** | LangChain                                                                                                                                   | LangChain                                                                                                                                           |

## Architecture

![Cloud SaaS](./img/self_hosted_control_plane_architecture.png)
