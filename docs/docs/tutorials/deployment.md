---
search:
  boost: 2
---

# Deployment 🚀

There are two free options for deploying LangGraph applications via the LangGraph Server:

- [Local](./langgraph-platform/local-server.md): Deploy for local testing and development. 
- [Standalone Container (Lite)](../concepts/langgraph_standalone_container.md): A limited version of Standalone Container for deployments unlikely to see more that 1 million node executions per year and that do not need crons and other enterprise features. Standalone Container (Lite) deployment option is free with a LangSmith API key.

## Other deployment options

Additionally, you can deploy to production with [LangGraph Platform](../concepts/langgraph_platform.md):

- [Cloud SaaS](../concepts/langgraph_cloud.md): Connect your GitHub repositories and deploy LangGraph Servers within LangChain's cloud. *We manage everything.*
- [Self-Hosted Data Plane<sup>(Beta)</sup>](../concepts/langgraph_self_hosted_data_plane.md): Create deployments from the [Control Plane UI](../concepts/langgraph_control_plane.md#control-plane-ui) and deploy LangGraph Servers to **your** cloud. *We manage the [control plane](../concepts/langgraph_control_plane.md). You manage the deployments.*
- [Self-Hosted Control Plane<sup>(Beta)</sup>](../concepts/langgraph_self_hosted_control_plane.md): Create deployments from a self-hosted [Control Plane UI](../concepts/langgraph_control_plane.md#control-plane-ui) and deploy LangGraph Servers to **your** cloud. *You manage everything.*
- [Standalone Container](../concepts/langgraph_standalone_container.md): Deploy LangGraph Server Docker images however you like.

For more information, see [Deployment options](../concepts/deployment_options.md).
