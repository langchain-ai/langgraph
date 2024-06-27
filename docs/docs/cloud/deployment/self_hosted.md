# How to Self-Host LangGraph Cloud API

!!! warning "Enterprise License Required"
    Self-hosting LangGraph Cloud API requires a license key. Please contact sales@langchain.dev for more details.

LangGraph Cloud APIs can be self-hosted with a valid LangGraph Cloud license key. Self-hosted deployments are built with Docker and deployed with Helm (on Kubernetes) or with Docker Compose. Ensure that the [Docker CLI](https://docs.docker.com/engine/reference/commandline/cli/) is installed.

LangGraph Cloud license key should be passed to the service as an environment variable named LANGGRAPH_CLOUD_LICENSE_KEY.

## Build Docker Image

1. Follow the [How-to Guide](setup.md) for setting up a LangGraph application for deployment. Your LangGraph application will vary from the example in the How-to Guide. However, ensure that the [LangGraph API configuration file](../reference/cli.md#configuration-file) is created.
1. Install the [LangGraph CLI](../reference/cli.md#installation).
1. Run the following LangGraph CLI `build` command to build a Docker image. Specify the image tag (`-t`) and other desired [options](../reference/cli.md#build).

        langgraph build -t tag_name

!!! info "Build Platform"
    When building the Docker image, ensure that the image is built for the platform of the target Kubernetes cluster: `langgraph build -t tag_name --platform linux/amd64,linux/arm64`

## Self-Host on Kubernetes

This section is for self-hosting LangGraph Cloud API on Kubernetes via Helm. A Kubernetes cluster must be provisioned before proceeding with these steps. The public Helm chart for LangGraph Cloud is available [here](https://github.com/langchain-ai/helm/tree/main/charts/langgraph-cloud).

1. Publish the built Docker image to a repository that can be accessed by the target Kubernetes cluster.
1. Ensure that the [Helm client](https://github.com/helm/helm?tab=readme-ov-file#install) is installed.
1. Make note of all environment variables that are needed for the application. These values will need to be set in the Helm `values` YAML configuration.
1. Follow [these instructions](https://github.com/langchain-ai/helm/tree/main/charts/langgraph-cloud#readme) to configure the Helm chart and deploy to Kubernetes.

## Self-Host with Docker

!!! warning "Under Construction"
    This section of the documentation is in progress.

Docker Compose can be used to deploy LangGraph Cloud to the compute infrastructure of your choice (e.g. VM).
