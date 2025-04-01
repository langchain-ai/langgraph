# How to Deploy Self-Hosted Control Plane

Before deploying, review the [conceptual guide for the Self-Hosted Control Plane](../../concepts/langgraph_self_hosted_control_plane.md) deployment option.

## Requirements

- You are using Kubernetes already
- You have already deployed/are deploying LangSmith self-hosted
- You use `langgraph-cli` and/or LangGraph Desktop app to test graph locally
- You use `langgraph build` command to build image and then push it to a registry your cluster has access to.Requirements
- `KEDA` installed on your cluster

         helm repo add kedacore https://kedacore.github.io/charts 
         helm install keda kedacore/keda --namespace keda --create-namespac

- Ingress Configuration (Recommended)
    - `Ingress Nginx` to serve as a reverse proxy for your deployment

            helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
            helm repo update
            helm install ingress-nginx ingress-nginx/ingress-nginx

    - A root domain that will suffix all domains for your workloads. E.g `us.langgraph.app`
    - Wildcard certificates to terminate TLS for your deployments
    - If you do not set this up, you will need to provision domains/certs for each of your deployments.

- Slack space in your cluster for additional deployments. `Cluster-Autoscaler` recommended to automatically provision new nodes

## Setup

1. As part of configuring your self-hosted LangSmith setup, you enable the langgraphPlatform. This will provision a few key resources.
    1. `listener`: This is a service that listens to our control plane for changes to your deployments and provision/updates downstream CRDS.
    1. `LangGraphPlatform CRD`: A CRD for LangGraph platform deployments. This contains the spec for managing an instance of a specific LangGraph platform deployment.
    1. `operator`: This operator handles changes to your LangGraph Platform CRDS.
    1. `host-backend`: This service handles api requests for the control plane.
1. Two additional images will be used by the chart:

        hostBackendImage:
          repository: "docker.io/langchain/hosted-langserve-backend"
          pullPolicy: IfNotPresent
          tag: "0.9.80"
        operatorImage:
          repository: "docker.io/langchain/langgraph-operator"
          pullPolicy: IfNotPresent
          tag: "aa9dff4"

1. In your values file, configure this like so:

        config:
          langgraphPlatform:
            enabled: true
            langgraphPlatformLicenseKey: "YOUR_LANGGRAPH_PLATFORM_LICENSE_KEY"
            rootDomain: "YOUR_ROOT_DOMAIN"

1. You can also configure base templates for your agents by overriding the base templates here: [values.yaml](https://github.com/langchain-ai/helm/blob/main/charts/langsmith/values.yaml#L898)
1. You create a LangGraph Cloud Project in your instance providing
    1. Image path for your LangGraph Cloud image
1. The `langgraph-listener` sees there is a revision that needs to be deployed in your Kubernetes cluster and provisions necessary resources
1. You use your LangSmith instance/existing cluster monitoring to ensure your deployments are healthy/running smoothly.
