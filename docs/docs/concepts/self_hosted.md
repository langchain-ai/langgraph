# Self-Hosted

!!! note Prerequisites

    - [LangGraph Platform](./langgraph_platform.md)
    - [Deployment Options](./deployment_options.md)

## Versions

There are two versions of the self-hosted deployment: [Self-Hosted Enterprise](./deployment_options.md#self-hosted-enterprise) and [Self-Hosted Lite](./deployment_options.md#self-hosted-lite).

### Self-Hosted Lite

The Self-Hosted Lite version is a limited version of LangGraph Platform that you can run locally or in a self-hosted manner (up to 1 million nodes executed per year).

When using the Self-Hosted Lite version, you authenticate with a [LangSmith](https://smith.langchain.com/) API key.

### Self-Hosted Enterprise

The Self-Hosted Enterprise version is the full version of LangGraph Platform.

To use the Self-Hosted Enterprise version, you must acquire a license key that you will need to pass in when running the Docker image. To acquire a license key, please email sales@langchain.dev.

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
