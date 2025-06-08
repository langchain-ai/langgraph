# Webhooks

Webhooks enable event-driven communication from your LangGraph Platform application to external services. For example, you may want to issue an update to a separate service once an API call to LangGraph Platform has finished running.

Many LangGraph Platform endpoints accept a `webhook` parameter. If this parameter is specified by an endpoint that can accept POST requests, LangGraph Platform will send a request at the completion of a run.

See the corresponding [how-to guide](../../cloud/how-tos/webhooks.md) for more detail.