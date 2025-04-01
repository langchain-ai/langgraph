# How to Deploy Self-Hosted Data Plane

Before deploying, review the [conceptual guide for the Self-Hosted Data Plane](../../concepts/langgraph_self_hosted_data_plane.md) deployment option.

## Prerequisites

1. You use the [LangGraph CLI](../../concepts/langgraph_cli.md) to [test your application locally](./test_locally.md).
1. You use the [LangGraph CLI](../../concepts/langgraph_cli.md) to build a Docker image (i.e. `langgraph build`) and push it to a registry your Kubernetes cluster or Amazon ECS cluster has access to.

## Kubernetes

1. We provide you a [Helm chart](https://github.com/langchain-ai/helm/tree/main/charts/langgraph-dataplane) which you run to setup your Kubernetes cluster. This chart contains a few important components.
    1. `langgraph-listener`: This is a service that listens to LangChain's [control plane](../../concepts/langgraph_control_plane.md) for changes to your deployments and creates/updates downstream CRDs.
    1. `LangGraphPlatform CRD`: A CRD for LangGraph Platform deployments. This contains the spec for managing an instance of a LangGraph Platform deployment.
    1. `langgraph-platform-operator`: This operator handles changes to your LangGraph Platform CRDs.
1. You give us your LangSmith organization ID. We will enable the Self-Hosted Data Plane for your organization.
1. You create a deployment from the [Control Plane UI](../../concepts/langgraph_control_plane.md#control-plane-ui).

## Amazon ECS

Coming soon!
