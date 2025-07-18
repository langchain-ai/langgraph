# How to Deploy Self-Hosted Control Plane

Before deploying, review the [conceptual guide for the Self-Hosted Control Plane](../../concepts/langgraph_self_hosted_control_plane.md) deployment option.

!!! info "Important"
    The Self-Hosted Control Plane deployment option requires an [Enterprise](../../concepts/plans.md) plan.

## Prerequisites

1. You are using Kubernetes.
1. You have self-hosted LangSmith deployed.
1. Use the [LangGraph CLI](../../concepts/langgraph_cli.md) to [test your application locally](../../tutorials/langgraph-platform/local-server.md).
1. Use the [LangGraph CLI](../../concepts/langgraph_cli.md) to build a Docker image (i.e. `langgraph build`) and push it to a registry your Kubernetes cluster has access to.
1. `KEDA` is installed on your cluster.

         helm repo add kedacore https://kedacore.github.io/charts 
         helm install keda kedacore/keda --namespace keda --create-namespace
1. Ingress Configuration
    1. You must set up an ingress for your LangSmith instance. All agents will be deployed as Kubernetes services behind this ingress.
    1. You can use this guide to [set up an ingress](https://docs.smith.langchain.com/self_hosting/configuration/ingress) for your instance.
1. You have slack space in your cluster for multiple deployments. `Cluster-Autoscaler` is recommended to automatically provision new nodes.
1. A valid Dynamic PV provisioner or PVs available on your cluster. You can verify this by running:

        kubectl get storageclass

1. Egress to `https://beacon.langchain.com` from your network. This is required for license verification and usage reporting if not running in air-gapped mode. See the [Egress documentation](../../cloud/deployment/egress.md) for more details.

## Setup

1. As part of configuring your Self-Hosted LangSmith instance, you enable the `langgraphPlatform` option. This will provision a few key resources.
    1. `listener`: This is a service that listens to the [control plane](../../concepts/langgraph_control_plane.md) for changes to your deployments and creates/updates downstream CRDs.
    1. `LangGraphPlatform CRD`: A CRD for LangGraph Platform deployments. This contains the spec for managing an instance of a LangGraph platform deployment.
    1. `operator`: This operator handles changes to your LangGraph Platform CRDs.
    1. `host-backend`: This is the [control plane](../../concepts/langgraph_control_plane.md).
1. Two additional images will be used by the chart. Use the images that are specified in the latest release.

        hostBackendImage:
          repository: "docker.io/langchain/hosted-langserve-backend"
          pullPolicy: IfNotPresent
        operatorImage:
          repository: "docker.io/langchain/langgraph-operator"
          pullPolicy: IfNotPresent

1. In your config file for langsmith (usually `langsmith_config.yaml`, enable the `langgraphPlatform` option. Note that you must also have a valid ingress setup:

        config:
          langgraphPlatform:
            enabled: true
            langgraphPlatformLicenseKey: "YOUR_LANGGRAPH_PLATFORM_LICENSE_KEY"
1. In your `values.yaml` file, configure the `hostBackendImage` and `operatorImage` options (if you need to mirror images)

1. You can also configure base templates for your agents by overriding the base templates [here](https://github.com/langchain-ai/helm/blob/main/charts/langsmith/values.yaml#L898).
1. You create a deployment from the [control plane UI](../../concepts/langgraph_control_plane.md#control-plane-ui).
