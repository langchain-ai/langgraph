---
search:
  boost: 2
---

# Deployment 🚀

Get started deploying your LangGraph applications locally:

- [Local](../cloud/deployment/test_locally.md): Deploy for local testing and development. 
- Self-Hosted Lite: A limited version of Standalone Container for deployments unlikely to see more that 1 million node executions per year and that do not need crons and other enterprise features. Self-Hosted Lite is free with a LangSmith API key.

## Other deployment options

Additionally, you can deploy to production with [LangGraph Platform](../concepts/langgraph_platform.md):

- [Cloud SaaS](../concepts/langgraph_cloud.md): Connect to your GitHub repositories and deploy LangGraph Servers to LangChain's cloud. We manage everything.
- [Self-Hosted Data Plane<sup>(Beta)</sup>](../concepts/self_hosted.md#self-hosted-data-plane-beta): Create deployments from the [Control Plane UI](../concepts/langgraph_control_plane.md#control-plane-ui) and deploy LangGraph Servers to your cloud. We manage the [control plane](../concepts/langgraph_control_plane.md), you manage the deployments.
- [Self-Hosted Control Plane<sup>(Beta)</sup>](../concepts/self_hosted.md#self-hosted-control-plane-beta): Create deployments from a self-hosted [Control Plane UI](../concepts/langgraph_control_plane.md#control-plane-ui) and deploy LangGraph Servers to your cloud. You manage everything.
- [Standalone Container](../concepts/langgraph_standalone_container.md): Deploy LangGraph Server Docker images however you like.

For more information, see [Deployment options](../concepts/deployment_options.md)
