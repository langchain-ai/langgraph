---
search:
  boost: 2
---

# Deployment 🚀

Get started deploying your LangGraph applications locally or on the cloud with
[LangGraph Platform](../concepts/langgraph_platform.md).

## Deployment options

- [Local](../cloud/deployment/test_locally.md): Deploy for local testing and development. 
- [Cloud SaaS<sup>(Beta)</sup>](../../concepts/langgraph_cloud/): Connect to your GitHub repositories and deploy LangGraph Servers to LangChain's cloud. We manage everything.
- [Self-Hosted Data Plane<sup>(Beta)</sup>](../../concepts/langgraph_self_hosted_data_plane/): Create deployments from the [Control Plane UI](../concepts/langgraph_control_plane.md#control-plane-ui) and deploy LangGraph Servers to your cloud. We manage the [control plane](../concepts/langgraph_control_plane.md), you manage the deployments.
- [Self-Hosted Control Plane<sup>(Beta)</sup>](../../concepts/langgraph_self_hosted_control_plane/): Create deployments from a self-hosted [Control Plane UI](../concepts/langgraph_control_plane.md#control-plane-ui) and deploy LangGraph Servers to your cloud. You manage everything.
- [Standalone Container](../concepts/langgraph_standalone_container.md) / Self-Hosted Lite: Deploy LangGraph Server Docker images however you like.
- Self-Hosted Lite: A limited version of Standalone Container for deployments unlikely to see more that 1 million node executions per year and that do not need crons and other enterprise features.

A quick comparison:

|                      | **Cloud SaaS** | **Self-Hosted Data Plane** | **Self-Hosted Control Plane** | **Standalone Container** | **Self-Hosted Lite** |
|----------------------|----------------|----------------------------|-------------------------------|--------------------------| ---------------------|
| **[Control Plane UI/API](../concepts/langgraph_control_plane.md)** | Yes | Yes | Yes | No | |
| **CI/CD** | Managed internally by platform | Managed externally by you | Managed externally by you | Managed externally by you | |
| **Data/compute residency** | LangChain’s cloud | Your cloud | Your cloud | Your cloud | |
| **Required permissions** | None | See details [here](). | See details [here](). | None | |
| **LangSmith compatibility** | Trace to LangSmith SaaS | Trace to LangSmith SaaS | Trace to Self-Hosted LangSmith | Optional tracing | |
| **[Server version compatibility](../concepts/langgraph_server/#server-versions)** | Enterprise | Enterprise | Enterprise | Lite, Enterprise | |
| **[Pricing](https://www.langchain.com/pricing-langgraph-platform)** | Plus | Enterprise | Enterprise | Developer | Free with LangSmith |

For more information, see 
