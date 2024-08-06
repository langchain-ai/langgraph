# Cloud Concepts

This page describes the high-level concepts of the LangGraph Cloud deployment.

## Deployment

A deployment is an instance of a LangGraph API. A single deployment can have many [revisions](#revision). When a deployment is created, all of the necessary infrastructure (e.g. database, containers, secrets store) are automatically provisioned. See the [architecture diagram](#architecture) below for more details.

See the [how-to guide](../deployment/cloud.md#create-new-deployment) for creating a new deployment.

## Revision

A revision is an iteration of a [deployment](#deployment). When a new deployment is created, an initial revision is automatically created. To deploy new code changes or update environment variable configurations for a deployment, a new revision must be created. When a revision is created, a new container image is built automatically.

See the [how-to guide](../deployment/cloud.md#create-new-revision) for creating a new revision.

## Asynchronous Deployment

Infrastructure for [deployments](#deployment) and [revisions](#revision) are provisioned and deployed asynchronously. They are not deployed immediately after submission. Currently, deployment can take up to several minutes.

## Architecture

!!! warning "Subject to Change"
    The LangGraph Cloud deployment architecture may change in the future.

A high-level diagram of a LangGraph Cloud deployment.

![diagram](langgraph_cloud_architecture.png)
