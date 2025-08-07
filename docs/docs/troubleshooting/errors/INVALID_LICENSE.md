# INVALID_LICENSE

This error is raised when license verification fails while attempting to start a self-hosted LangGraph Platform server. This error is specific to the LangGraph Platform and is not related to the open source libraries.

## When This Occurs

This error occurs when running a self-hosted deployment of LangGraph Platform without a valid enterprise license or API key.

## Troubleshooting

### Confirm deployment type

First, confirm the desired mode of deployment.

#### For Local Development

If you're just developing locally, you can use the lightweight in-memory server by running `langgraph dev`.
See the [local server](../../tutorials/langgraph-platform/local-server.md) docs for more information.

#### For Managed LangGraph Platform

If you would like a fast managed environment, consider the [Cloud SaaS](../../concepts/langgraph_cloud.md) deployment option. This requires no additional license key.

#### For Standalone Container

For self-hosting, set the `LANGGRAPH_CLOUD_LICENSE_KEY` environment variable. If you are interested in an enterprise license key, please contact the LangChain support team.

For more information on deployment options and their features, see the [Deployment Options](../../concepts/deployment_options.md) documentation.


### Confirm credentials

If you have confirmed that you would like to self-host LangGraph Platform, please verify your credentials.

#### For Standalone Container

1. Confirm that you have provided a working `LANGGRAPH_CLOUD_LICENSE_KEY` environment variable in your deployment environment or `.env` file
2. Confirm the key is still valid and has not surpassed its expiration date