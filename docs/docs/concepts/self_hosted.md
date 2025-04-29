---
search:
  boost: 2
---

# Self-Hosted

!!! note Prerequisites

    - [LangGraph Platform](./langgraph_platform.md)
    - [Deployment Options](./deployment_options.md)

## Versions

There are two versions of the self-hosted deployment: [Self-Hosted Data Plane](./deployment_options.md#self-hosted-data-plane) and [Self-Hosted Control Plane](./deployment_options.md#self-hosted-control-plane).

### Self-Hosted Data Plane

The [Self-Hosted Data Plane](./langgraph_self_hosted_data_plane.md) deployment option is a "hybrid" model for deployment where we manage the [control plane](./langgraph_control_plane.md) in our cloud and you manage the [data plane](./langgraph_data_plane.md) in your cloud. This option provides a way to securely manage your data plane infrastructure, while offloading control plane management to us.

When using the Self-Hosted Data Plane version, you authenticate with a [LangSmith](https://smith.langchain.com/) API key.

### Self-Hosted Control Plane

The [Self-Hosted Control Plane](./langgraph_self_hosted_control_plane.md) deployment option is a fully self-hosted model for deployment where you manage the [control plane](./langgraph_control_plane.md) and [data plane](./langgraph_data_plane.md) in your cloud. This option give you full control and responsibility of the control plane and data plane infrastructure.

## Requirements

- You use `langgraph-cli` and/or [LangGraph Studio](./langgraph_studio.md) app to test graph locally.
- You use `langgraph build` command to build image.

## How it works

- Deploy Redis and Postgres instances on your own infrastructure.
- Build the docker image for [LangGraph Server](./langgraph_server.md) using the [LangGraph CLI](./langgraph_cli.md).
- Deploy a web server that will run the docker image and pass in the necessary environment variables.

!!! warning "Note"

    The LangGraph Platform Deployments view is optionally available for Self-Hosted LangGraph deployments. With one click, self-hosted LangGraph deployments can be deployed in the same Kubernetes cluster where a self-hosted LangSmith instance is deployed.

For step-by-step instructions, see [How to set up a self-hosted deployment of LangGraph](../how-tos/deploy-self-hosted.md).

## Helm Chart

If you would like to deploy LangGraph Cloud on Kubernetes, you can use this [Helm chart](https://github.com/langchain-ai/helm/blob/main/charts/langgraph-cloud/README.md).

## Related

- [How to set up a self-hosted deployment of LangGraph](../how-tos/deploy-self-hosted.md).
