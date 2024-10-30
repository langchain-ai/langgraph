# LangGraph Cloud

!!! info "Prerequisites"
    - [LangGraph Platform](./langgraph_platform.md)
    - [LangGraph Server](./langgraph_server.md)

## Overview

LangGraph Cloud is a managed service that provides a scalable and secure environment for deploying LangGraph APIs. It is designed to work seamlessly with your LangGraph API regardless of how it is defined, what tools it uses, or any dependencies. LangGraph Cloud provides a simple way to deploy and manage your LangGraph API in the cloud.

## Deployment

A **deployment** is an instance of a LangGraph API. A single deployment can have many [revisions](#revision). When a deployment is created, all the necessary infrastructure (e.g. database, containers, secrets store) are automatically provisioned. See the [architecture diagram](#architecture) below for more details.

See the [how-to guide](../cloud/deployment/cloud.md#create-new-deployment) for creating a new deployment.

## Revision

A revision is an iteration of a [deployment](#deployment). When a new deployment is created, an initial revision is automatically created. To deploy new code changes or update environment variable configurations for a deployment, a new revision must be created. When a revision is created, a new container image is built automatically.

See the [how-to guide](../cloud/deployment/cloud.md#create-new-revision) for creating a new revision.

## Asynchronous Deployment

Infrastructure for [deployments](#deployment) and [revisions](#revision) are provisioned and deployed asynchronously. They are not deployed immediately after submission. Currently, deployment can take up to several minutes.

## Architecture

!!! warning "Subject to Change"
The LangGraph Cloud deployment architecture may change in the future.

A high-level diagram of a LangGraph Cloud deployment.

![diagram](img/langgraph_cloud_architecture.png)

## Related

- [Deployment Options](./deployment_options.md)