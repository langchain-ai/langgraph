---
search:
  boost: 2
---

# Deployment Options

## Free deployment

[Local](../tutorials/langgraph-platform/local-server.md): Deploy for local testing and development.

## Production deployment

There are 4 main options for deploying with the [LangGraph Platform](langgraph_platform.md):

1. [Cloud SaaS](#cloud-saas)

1. [Self-Hosted Data Plane](#self-hosted-data-plane)

1. [Self-Hosted Control Plane](#self-hosted-control-plane)

1. [Standalone Container](#standalone-container)


A quick comparison:

|                      | **Cloud SaaS** | **Self-Hosted Data Plane** | **Self-Hosted Control Plane** | **Standalone Container** |
|----------------------|----------------|----------------------------|-------------------------------|--------------------------|
| **[Control plane UI/API](../concepts/langgraph_control_plane.md)** | Yes | Yes | Yes | No |
| **CI/CD** | Managed internally by platform | Managed externally by you | Managed externally by you | Managed externally by you |
| **Data/compute residency** | LangChain's cloud | Your cloud | Your cloud | Your cloud |
| **LangSmith compatibility** | Trace to LangSmith SaaS | Trace to LangSmith SaaS | Trace to Self-Hosted LangSmith | Optional tracing |
| **[Pricing](https://www.langchain.com/pricing-langgraph-platform)** | Plus | Enterprise | Enterprise | Enterprise |

## Cloud SaaS

The [Cloud SaaS](./langgraph_cloud.md) deployment option is a fully managed model for deployment where we manage the [control plane](./langgraph_control_plane.md) and [data plane](./langgraph_data_plane.md) in our cloud. This option provides a simple way to deploy and manage your LangGraph Servers.

Connect your GitHub repositories to the platform and deploy your LangGraph Servers from the [control plane UI](./langgraph_control_plane.md#control-plane-ui). The build process (i.e. CI/CD) is managed internally by the platform.

For more information, please see:

* [Cloud SaaS Conceptual Guide](./langgraph_cloud.md)
* [How to deploy to Cloud SaaS](../cloud/deployment/cloud.md)

## Self-Hosted Data Plane

!!! info "Important"
    The Self-Hosted Data Plane deployment option requires an [Enterprise](../concepts/plans.md) plan.

The [Self-Hosted Data Plane](./langgraph_self_hosted_data_plane.md) deployment option is a "hybrid" model for deployment where we manage the [control plane](./langgraph_control_plane.md) in our cloud and you manage the [data plane](./langgraph_data_plane.md) in your cloud. This option provides a way to securely manage your data plane infrastructure, while offloading control plane management to us.

Build a Docker image using the [LangGraph CLI](./langgraph_cli.md) and deploy your LangGraph Server from the [control plane UI](./langgraph_control_plane.md#control-plane-ui).

Supported Compute Platforms: [Kubernetes](https://kubernetes.io/), [Amazon ECS](https://aws.amazon.com/ecs/) (coming soon!)

For more information, please see:

* [Self-Hosted Data Plane Conceptual Guide](./langgraph_self_hosted_data_plane.md)
* [How to deploy the Self-Hosted Data Plane](../cloud/deployment/self_hosted_data_plane.md)

## Self-Hosted Control Plane

!!! info "Important"
    The Self-Hosted Control Plane deployment option requires an [Enterprise](../concepts/plans.md) plan.

The [Self-Hosted Control Plane](./langgraph_self_hosted_control_plane.md) deployment option is a fully self-hosted model for deployment where you manage the [control plane](./langgraph_control_plane.md) and [data plane](./langgraph_data_plane.md) in your cloud. This option gives you full control and responsibility of the control plane and data plane infrastructure.

Build a Docker image using the [LangGraph CLI](./langgraph_cli.md) and deploy your LangGraph Server from the [control plane UI](./langgraph_control_plane.md#control-plane-ui).

Supported Compute Platforms: [Kubernetes](https://kubernetes.io/)

For more information, please see:

* [Self-Hosted Control Plane Conceptual Guide](./langgraph_self_hosted_control_plane.md)
* [How to deploy the Self-Hosted Control Plane](../cloud/deployment/self_hosted_control_plane.md)

## Standalone Container

The [Standalone Container](./langgraph_standalone_container.md) deployment option is the least restrictive model for deployment. Deploy standalone instances of a LangGraph Server in your cloud, using any of the [available](./plans.md) license options.

Build a Docker image using the [LangGraph CLI](./langgraph_cli.md) and deploy your LangGraph Server using the container deployment tooling of your choice. Images can be deployed to any compute platform.

For more information, please see:

* [Standalone Container Conceptual Guide](./langgraph_standalone_container.md)
* [How to deploy a Standalone Container](../cloud/deployment/standalone_container.md)

## Related

For more information, please see:

* [LangGraph Platform plans](./plans.md)
* [LangGraph Platform pricing](https://www.langchain.com/langgraph-platform-pricing)
