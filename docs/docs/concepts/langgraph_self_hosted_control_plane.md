# Self-Hosted Control Plane

## Overview

The Self-Hosted Control Plane deployment option is a fully self-hosted model for deployment where you manage the control plane and data plane in your cloud.

|                   | Control Plane     | Data Plane |
|-------------------|-------------------|------------|
| **What is it?** | <ul><li>LangGraph Platform UI for creating deployments and revisions</li><li>Control plane APIs for creating deployments and revisions</li></ul> | <ul><li>Data plane listene for reconciling deployments with control plane state</li><li>LangGraph server container(s)</li><li>Postgres, Redis</li></ul> |
| **Where it's hosted?** | Your cloud | Your cloud |
| **Who provisions and manages it?** | You | You |

## Architecture

![Self-Hosted Control Plane Architecture](./img/self_hosted_control_plane_architecture.png)

## Compute Platforms

### Kubernetes

The Self-Hosted Control Plane deployment option supports deploying control plane and data plane infrastructure to any Kubernetes cluster.

## Miscellaneous

Miscellaneous options/details about the Self-Hosted Control Plane deployment option.

|                   | Description |
|-------------------|-------------|
| **Lite vs Enterprise** | Enterprise |
| **Tracing** | Trace to Self-Hosted LangSmith |
| **Licensing** | Air-gapped license key or LangGraph Platform License Key |
| **Telemetry** | Self-reported usage (audit) for air-gapped license key.</br></br>Telemetry sent to LangSmith SaaS for LangGraph Platform License Key. |

## Related

* [How to deploy the Self-Hosted Control Plane](../cloud/deployment/self_hosted_control_plane.md)
