---
search:
  boost: 2
---

# Deployment

Get started deploying your LangGraph applications locally or on the cloud with
[LangGraph Platform](../concepts/langgraph_platform.md).

## Get Started 🚀 {#quick-start}

- [LangGraph Server Quickstart](../tutorials/langgraph-platform/local-server.md): Launch a LangGraph server locally and interact with it using REST API and LangGraph Studio Web UI.
- [LangGraph Template Quickstart](../concepts/template_applications.md): Start building with LangGraph Platform using a template application.
- [Deploy with LangGraph Cloud Quickstart](../cloud/quick_start.md): Deploy a LangGraph app using LangGraph Cloud.


## Deployment Options

- <a href="../../concepts/langgraph_cloud/">Cloud SaaS<sup>(Beta)</sup></a>: Connect to your GitHub repositories and deploy LangGraph Servers to LangChain's cloud. We manage everything.
- <a href="../../concepts/langgraph_self_hosted_data_plane/">Self-Hosted Data Plane<sup>(Beta)</sup></a>: Create deployments from the [Control Plane UI](../concepts/langgraph_control_plane.md#control-plane-ui) and deploy LangGraph Servers to your cloud. We manage the [control plane](../concepts/langgraph_control_plane.md), you manage the deployments.
- <a href="../../concepts/langgraph_self_hosted_control_plane/">Self-Hosted Control Plane<sup>(Beta)</sup></a>: Create deployments from a self-hosted [Control Plane UI](../concepts/langgraph_control_plane.md#control-plane-ui) and deploy LangGraph Servers to your cloud. You manage everything.
- [Standalone Container](../concepts/langgraph_standalone_container.md): Deploy LangGraph Server Docker images however you like.

A quick comparison...

|                      | **Cloud SaaS** | **Self-Hosted [Data Plane](../concepts/langgraph_data_plane.md)** | **Self-Hosted [Control Plane](../concepts/langgraph_control_plane.md)** | **Standalone Container** |
|----------------------|----------------|----------------------------|-------------------------------|--------------------------|
| **[Control Plane UI/API](../concepts/langgraph_control_plane.md)** | Yes | Yes | Yes | No |
| **CI/CD** | Managed internally by platform | Managed externally by you | Managed externally by you | Managed externally by you |
| **Data/Compute Residency** | LangChain’s cloud | Your cloud | Your cloud | Your cloud |
| **Required Permissions** | None | See details [here](). | See details [here](). | None |
| **LangSmith Compatibility** | Trace to LangSmith SaaS | Trace to LangSmith SaaS | Trace to Self-Hosted LangSmith | Optional tracing |
| **[Pricing](https://www.langchain.com/pricing-langgraph-platform)** | Plus | Enterprise | Enterprise | Developer |
