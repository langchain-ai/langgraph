# Standalone Container

## Overview

The Standalone Container deployment option is the least restrictive model for deployment. There is no [control plane](./langgraph_control_plane.md). [Data plane](./langgraph_data_plane.md) infrastructure is managed by you.

|                   | Control Plane     | Data Plane |
|-------------------|-------------------|------------|
| **What is it?** | n/a | <ul><li>LangGraph server container(s)</li><li>Postgres, Redis</li></ul> |
| **Where it's hosted?** | n/a | Your cloud |
| **Who provisions and manages it?** | n/a | You |

## Architecture

![Standalone Container](./img/langgraph_platform_deployment_architecture.png)

## Compute Platforms

### Kubernetes

The Standalone Container deployment option supports deploying data plane infrastructure to a Kubernetes cluster.

### Docker

The Standalone Container deployment option supports deploying data plane infrastructure to any Docker-supported compute platform.

## Miscellaneous

Miscellaneous options/details about the Standalone Container deployment option.

|                   | Description |
|-------------------|-------------|
| **Lite vs Enterprise** | Lite or Enterprise |
| **Custom Postgres/Redis** | Yes |
| **Tracing** | Optional</br></br>Disable tracing, trace to LangSmith SaaS, or trace Self-Hosted LangSmith. |
| **Licensing** | Air-gapped license key or LangGraph Platform License Key |
| **Telemetry** | Self-reported usage (audit) for air-gapped license key.</br></br>Telemetry sent to LangSmith SaaS for LangGraph Platform License Key. |

## Deployment

Follow the how-to guide for [how to deploy a Standalone Container](../cloud/deployment/standalone_container.md).
