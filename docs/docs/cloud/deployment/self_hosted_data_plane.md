# How to Deploy Self-Hosted Data Plane

## Kubernetes

### Requirements

- You use `langgraph-cli` and/or LangGraph Desktop app to test your application locally.
- You use `langgraph build` command to build a Docker image and then push it to a registry your Kubernetes cluster has access to.

### Setup

1. We provide you a [Helm chart]() which you run to setup your Kubernetes cluster. This chart contains a few important components.
    1. `langgraph-listener`: This is a service that listens to our control plane for changes to your deployments and provision/updates downstream CRDS.
    1. `LangGraphPlatform CRD`: A CRD for LangGraph platform deployments. This contains the spec for managing an instance of a specific LangGraph platform deployment.
    1. `langgraph-platform-operator`: This operator handles changes to your LangGraph Platform CRDS.
1. You give us your LangSmith organization id. We will enable the Self-Hosted Data Plane for your organization.
1. You create a LangGraph Cloud Project in `smith.langchain.com` providing...
    1. Image path for your LangGraph Cloud image
1. The `langgraph-listener` sees there is a revision that needs to be deployed in your Kubernetes cluster and provisions necessary resources.

## Amazon ECS

Coming soon...

## Concepts

Follow the conceptual guide for [Self-Hosted Data Plane Conceptual Guide](../../concepts/langgraph_self_hosted_data_plane.md).
