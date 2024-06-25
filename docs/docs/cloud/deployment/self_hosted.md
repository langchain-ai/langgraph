# How to Self-Host LangGraph Cloud API

LangGraph Cloud APIs can be self-hosted with a valid LangGraph Cloud license key. Self-hosted deployments are built with Docker and Helm. Ensure that both the [Docker CLI](https://docs.docker.com/engine/reference/commandline/cli/) and [Helm client](https://github.com/helm/helm?tab=readme-ov-file#install) are installed.

## Build Image

1. Follow the [How-to Guide](setup.md) for setting up a LangGraph application for deployment.
1. Install the [LangGraph CLI](../reference/cli.md#installation).
1. Run the following LangGraph CLI `build` command to build a Docker image. Specify the image tag (`-t`) and other desired [options](../reference/cli.md#build).

        langgraph build -t tag_name

## Configure Helm Chart

The public Helm chart for LangGraph Cloud is available [here](https://github.com/langchain-ai/helm/tree/main/charts/langgraph-cloud). Follow [these instructions](https://github.com/langchain-ai/helm/tree/main/charts/langgraph-cloud#readme) for configuring your Helm chart and deploying to Kubernetes.
