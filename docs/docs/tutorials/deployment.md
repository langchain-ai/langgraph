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

- [Cloud SaaS](../concepts/langgraph_cloud.md): Connect GitHub repositories to LangSmith and deploy LangGraph servers to LangChain's cloud. We manage everything.
- [Self-Hosted Data Plane](../concepts/langgraph_self_hosted_data_plane.md): Manage deployments from LangSmith, but deploy LangGraph servers to your cloud. We manage LangSmith, you manage the deployments. 
- [Self-Hosted Control Plane](../concepts/langgraph_self_hosted_control_plane.md): Manage deployments from Self-Hosted LangSmith and deploy LangGraph servers to your cloud. You manage everything.
- [Standalone Container](../concepts/langgraph_standalone_container.md): Deploy LangGraph server Docker images however you like.

### Quick Comparison
|                      | **Cloud SaaS** | **Self-Hosted Data Plane** | **Self-Hosted Control Plane** | **Standalone Container** |
|----------------------|----------------|----------------------------|-------------------------------|--------------------------|
| **Description**          | User connects GitHub repository to LangSmith and deploys via the Deployments UI. | User builds a Docker image using the LangGraph CLI. User deploys via the Deployments UI to their cloud. | User builds a Docker image using the LangGraph CLI. User deploys via the Deployments UI to their cloud. | User builds a Docker image using the LangGraph CLI. User deploys a standalone instance of the LangGraph server using any container deployment tooling. |
| **LangSmith**            | Requires LangSmith SaaS | Requires LangSmith SaaS | Requires Self-Hosted LangSmith | Optional |
| **Deployments UI**       | Yes | Yes | Yes | No |
| **CI/CD**                | Build process is managed internally by the platform. | User builds image and manages CI/CD workflow externally. | User builds image and manages CI/CD workflow externally. | User builds image and manages CI/CD workflow externally. |
| **Data Residency**       | LangChain’s cloud | User’s cloud | User’s cloud | User’s cloud |
| **Required Permissions** | None | See details [here](). | See details [here](). | None |