# How to Deploy Self-Hosted Data Plane

Before deploying, review the [conceptual guide for the Self-Hosted Data Plane](../../concepts/langgraph_self_hosted_data_plane.md) deployment option.

!!! info "Important"
    The Self-Hosted Data Plane deployment option requires an [Enterprise](../../concepts/plans.md) plan.

## Prerequisites

1. Use the [LangGraph CLI](../../concepts/langgraph_cli.md) to [test your application locally](../../tutorials/langgraph-platform/local-server.md).
1. Use the [LangGraph CLI](../../concepts/langgraph_cli.md) to build a Docker image (i.e. `langgraph build`) and push it to a registry your Kubernetes cluster or Amazon ECS cluster has access to.

## Kubernetes

### Prerequisites
1. `KEDA` is installed on your cluster.

        helm repo add kedacore https://kedacore.github.io/charts
        helm install keda kedacore/keda --namespace keda --create-namespace

1. A valid `Ingress` controller is installed on your cluster.
1. You have slack space in your cluster for multiple deployments. `Cluster-Autoscaler` is recommended to automatically provision new nodes.
1. You will need to enable egress to two control plane URLs. The listener polls these endpoints for deployments:

        https://api.host.langchain.com
        https://api.smith.langchain.com

### Setup

1. You give us your LangSmith organization ID. We will enable the Self-Hosted Data Plane for your organization.
1. We provide you a [Helm chart](https://github.com/langchain-ai/helm/tree/main/charts/langgraph-dataplane) which you run to setup your Kubernetes cluster. This chart contains a few important components.
    1. `langgraph-listener`: This is a service that listens to LangChain's [control plane](../../concepts/langgraph_control_plane.md) for changes to your deployments and creates/updates downstream CRDs.
    1. `LangGraphPlatform CRD`: A CRD for LangGraph Platform deployments. This contains the spec for managing an instance of a LangGraph Platform deployment.
    1. `langgraph-platform-operator`: This operator handles changes to your LangGraph Platform CRDs.
1. Configure your `langgraph-dataplane-values.yaml` file.

        config:
          langsmithApiKey: "" # API Key of your Workspace
          langsmithWorkspaceId: "" # Workspace ID
          hostBackendUrl: "https://api.host.langchain.com" # Only override this if on EU
          smithBackendUrl: "https://api.smith.langchain.com" # Only override this if on EU

1. Deploy `langgraph-dataplane` Helm chart.

        helm repo add langchain https://langchain-ai.github.io/helm/
        helm repo update
        helm upgrade -i langgraph-dataplane langchain/langgraph-dataplane --values langgraph-dataplane-values.yaml

1. If successful, you will see two services start up in your namespace.

        NAME                                          READY   STATUS              RESTARTS   AGE
        langgraph-dataplane-listener-7fccd788-wn2dx   0/1     Running             0          9s
        langgraph-dataplane-redis-0                   0/1     ContainerCreating   0          9s

1. You create a deployment from the [control plane UI](../../concepts/langgraph_control_plane.md#control-plane-ui).

## Amazon ECS

Coming soon!
