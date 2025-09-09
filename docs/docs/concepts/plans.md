---
search:
  boost: 2
---

# LangGraph Platform Plans


## Overview
LangGraph Platform is a solution for deploying agentic applications in production.
There are three different plans for using it.

- **Developer**: All [LangSmith](https://smith.langchain.com/) users have access to this plan. You can sign up for this plan simply by creating a LangSmith account. This gives you access to the [local deployment](./deployment_options.md#free-deployment) option.
- **Plus**: All [LangSmith](https://smith.langchain.com/) users with a [Plus account](https://docs.smith.langchain.com/administration/pricing) have access to this plan. You can sign up for this plan simply by upgrading your LangSmith account to the Plus plan type. This gives you access to the [Cloud](./deployment_options.md#cloud-saas) deployment option.
- **Enterprise**: This is separate from LangSmith plans. You can sign up for this plan by [contacting our sales team](https://www.langchain.com/contact-sales). This gives you access to all [deployment options](./deployment_options.md).


## Plan Details

|                                                                  | Developer                                   | Plus                                                  | Enterprise                                          |
|------------------------------------------------------------------|---------------------------------------------|-------------------------------------------------------|-----------------------------------------------------|
| Deployment Options                                               | Local                          | Cloud SaaS                                         | <ul><li>Cloud SaaS</li><li>Self-Hosted Data Plane</li><li>Self-Hosted Control Plane</li><li>Standalone Container</li></ul> |
| Usage                                                            | Free | See [Pricing](https://www.langchain.com/langgraph-platform-pricing) | Custom                                              |
| APIs for retrieving and updating state and conversational history | ✅                                           | ✅                                                     | ✅                                                   |
| APIs for retrieving and updating long-term memory                | ✅                                           | ✅                                                     | ✅                                                   |
| Horizontally scalable task queues and servers                    | ✅                                           | ✅                                                     | ✅                                                   |
| Real-time streaming of outputs and intermediate steps            | ✅                                           | ✅                                                     | ✅                                                   |
| Assistants API (configurable templates for LangGraph apps)       | ✅                                           | ✅                                                     | ✅                                                   |
| Cron scheduling                                                  | --                                          | ✅                                                     | ✅                                                   |
| LangGraph Studio for prototyping                                 | 	✅                                         | ✅                                                    | ✅                                                  |
| Authentication & authorization to call the LangGraph APIs        | --                                          | Coming Soon!                                          | Coming Soon!                                        |
| Smart caching to reduce traffic to LLM API                       | --                                          | Coming Soon!                                          | Coming Soon!                                        |
| Publish/subscribe API for state                                  | --                                          | Coming Soon!                                          | Coming Soon!                                        |
| Scheduling prioritization                                        | --                                          | Coming Soon!                                          | Coming Soon!                                        |

For pricing information, see [LangGraph Platform Pricing](https://www.langchain.com/langgraph-platform-pricing).

## Related

For more information, please see:

* [Deployment Options conceptual guide](./deployment_options.md)
* [LangGraph Platform Pricing](https://www.langchain.com/langgraph-platform-pricing)
* [LangSmith Plans](https://docs.smith.langchain.com/administration/pricing)
