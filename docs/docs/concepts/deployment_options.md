# Deployment Options

!!! info "Prerequisites"

    - [LangGraph Platform](./langgraph_platform.md)
    - [LangGraph Server](./langgraph_server.md)
    - [LangGraph Platform Plans](./plans.md)

## Overview

There are 4 main options for deploying with the LangGraph Platform:

1. **[Cloud SaaS](#cloud-saas)**: Available for **Plus** and **Enterprise** plans.

1. **[Self-Hosted Data Plane](#self-hosted-data-plane)**: Available for the **Enterprise** plan.

1. **[Self-Hosted Control Plane](#self-hosted-control-plane)**: Available for the **Enterprise** plan.

1. **[Standalone Container](#standalone-container)**: Available for all plans.

Please see the [LangGraph Platform Plans](./plans.md) for more information on the different plans.

The guide below will explain the differences between the deployment options.

## Cloud SaaS

The [Cloud SaaS](./langgraph_cloud.md) deployment option is hosted as part of [LangSmith](https://smith.langchain.com/). This option provides a simple way to deploy and manage your LangGraph applications.

Connect your GitHub repository to the platform and deploy your LangGraph application from the LangGraph Platform UI.

For more information, please see:

* [Cloud SaaS Conceptual Guide](./langgraph_cloud.md)
* [How to deploy to Cloud SaaS](../cloud/deployment/cloud.md)

## Self-Hosted Data Plane

The [Self-Hosted Data Plane](./langgraph_self_hosted_data_plane.md) deployment option is a "hybrid" model for deployemnt where we manage the control plane (e.g. LangGraph Platform UI, deployment APIs) in our cloud and you manage the data plane (e.g. containers, Postgres, Redis) in your cloud. This option provides a way to securely manage your data plane infrastucture, while offloading control plane management to us.

Build a Docker image using the [LangGraph CLI]() and deploy your LangGraph application from the LangGraph Platform UI.

Supported Compute Platforms

- Kubernetes
- Amazon ECS (coming soon!)

For more information, please see:

* [Self-Hosted Data Plane Conceptual Guide](./langgraph_self_hosted_data_plane.md)
* [How to deploy the Self-Hosted Data Plane](../cloud/deployment/self_hosted_data_plane.md)

## Self-Hosted Control Plane

The [Self-Hosted Control Plane](./langgraph_self_hosted_control_plane.md) deployment option is a fully self-hosted model for deployment where you manage the control plane (e.g. LangGraph Platform UI, deployment APIs) and data plane (e.g. containers, Postgres, Redis) in your cloud. This options give you full control and responsibility of the control plane and data plane.

Build a Docker image using the [LangGraph CLI]() and deploy your LangGraph application from the LangGraph Platform UI.

Supported Compute Platforms

- Kubernetes

For more information, please see:

* [Self-Hosted Control Plane Conceptual Guide](./langgraph_self_hosted_control_plane.md)
* [How to deploy the Self-Hosted Control Plane](../cloud/deployment/self_hosted_control_plane.md)

## Standalone Container

The [Standalone Container](./langgraph_standalone_container.md) deployment is the least restrictive model for deployment. Deploy standalone instances of a LangGraph application in your cloud.

Build a Docker image using the [LangGraph CLI] and deploy your LangGraph application using any container deployment tooling of your choice.

For more information, please see:

* [Sandalone Container Conceptual Guide](./langgraph_standalone_container.md)
* [How to deploy a Standalone Container](../cloud/deployment/standalone_container.md)

## Related

For more information, please see:

* [LangGraph Platform plans](./plans.md)
* [LangGraph Platform pricing](https://www.langchain.com/langgraph-platform-pricing)
* [Deployment how-to guides](../how-tos/index.md#deployment)
