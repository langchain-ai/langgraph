# Deployment Options

!!! info "Prerequisites"

    - [LangGraph Platform](./langgraph_platform.md)

## Overview

There are several deployment options for LangGraph Platform:

|                                | Free | Self Hosted | Bring Your Own Cloud | Cloud |
|--------------------------------|------|-------------|----------------------|-------|
| Who manages the infrastructure | You  | You         | Us                   | Us    |
| With whom does the data reside | You  | You         | You                  | Us    |
| LangGraph Studio Included?     | ❌    | ✅           | ✅                    | ✅     |
| Assistants Included?           | ❌    | ✅           | ✅                    | ✅     |


### Free

All you need in order to use the free version of LangGraph Platform is a [LangSmith](https://smith.langchain.com/) API key.

You need to add this as an environment variable when running LangGraph Platform. It should be provided as `LANGSMITH_API_KEY=...`.

LangGraph Platform will provide a one-time check when starting up to ensure that it is a valid LangSmith key.

The free version of LangGraph Platform does not have access to some features that the other versions have, namely LangGraph Studio and Assistants.

### Cloud

The Cloud version of LangGraph Platform is hosted as part of [LangSmith](https://smith.langchain.com/).
This deployment option provides a seamless integration with GitHub to easily deploy your code from there.
It also integrates seamlessly with LangSmith for observability and testing.

While in beta, the Cloud version of LangGraph Platform is available to all users of LangSmith on the [Plus or Enterprise plans](https://docs.smith.langchain.com/administration/pricing).

### Self Hosted

The Self Hosted version of LangGraph Platform can be set up in the same way as the free version.
The only difference is that rather than specifying a LangSmith API key, you pass in a license key.

This license key gives you access to all LangGraph Platform features, like LangGraph Studio and Assistants.

This is a paid offering. Please contact sales@langchain.dev for pricing.

Please see the [Self Hosted Deployment](../how-tos/deployment/self_hosted.md) guide for more information on how to set up the Self Hosted version of LangGraph Platform.

### Bring your own cloud

This combines the best of both worlds for Cloud and Self Hosted.
We manage the infrastructure, so you don't have to, but the infrastructure all runs within your cloud.

This is currently only available on AWS.

This is a paid offering. Please contact sales@langchain.dev for pricing.