# How to Self-Host LangGraph Cloud API

!!! warning "Enterprise License Required"
    Self-hosting LangGraph Cloud API requires a license key. Please contact sales@langchain.dev for more details.

LangGraph Cloud APIs can be self-hosted with a valid LangGraph Cloud license key. Self-hosted deployments are built with Docker and deployed with Helm (on Kubernetes) or with Docker Compose. Ensure that the [Docker CLI](https://docs.docker.com/engine/reference/commandline/cli/) is installed.

## Build Docker Image

1. Follow the [How-to Guide](setup.md) for setting up a LangGraph application for deployment.
1. Install the [LangGraph CLI](../reference/cli.md#installation).
1. Run the following LangGraph CLI `build` command to build a Docker image. Specify the image tag (`-t`) and other desired [options](../reference/cli.md#build).

        langgraph build -t tag_name

## Self-Host on Kubernetes

The public Helm chart for LangGraph Cloud is available [here](https://github.com/langchain-ai/helm/tree/main/charts/langgraph-cloud).

1. Ensure that the [Helm client](https://github.com/helm/helm?tab=readme-ov-file#install) is installed.
1. Publish the built Docker image to a respository that can be accessed by the target Kubernetes cluster.
1. Follow [these instructions](https://github.com/langchain-ai/helm/tree/main/charts/langgraph-cloud#readme) to configure the Helm chart and deploy to Kubernetes.

## Self-Host with Docker

Docker Compose can be used to deploy LangGraph Cloud to the compute infrastructure of your choice (e.g. VM).
