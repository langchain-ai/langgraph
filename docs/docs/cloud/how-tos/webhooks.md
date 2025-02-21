# Using Webhooks

When working with LangGraph Cloud, you may want to use webhooks to receive updates after an API call completes. Webhooks are useful for triggering actions in your service once a run has finished processing. To implement this, you need to expose an endpoint that can accept `POST` requests and pass this endpoint as a `webhook` parameter in your API request.

Currently, the SDK does not provide built-in support for defining webhook endpoints, but you can specify them manually using API requests.

## Supported Endpoints

The following API endpoints accept a `webhook` parameter:

| Operation | HTTP Method | Endpoint |
|-----------|------------|----------|
| Create Run | `POST` | `/thread/{thread_id}/runs` |
| Create Thread Cron | `POST` | `/thread/{thread_id}/runs/crons` |
| Stream Run | `POST` | `/thread/{thread_id}/runs/stream` |
| Wait Run | `POST` | `/thread/{thread_id}/runs/wait` |
| Create Cron | `POST` | `/runs/crons` |
| Stream Run Stateless | `POST` | `/runs/stream` |
| Wait Run Stateless | `POST` | `/runs/wait` |

In this guide, we’ll show how to trigger a webhook after streaming a run.

## Setting Up Your Assistant and Thread

Before making API calls, set up your assistant and thread.

=== "Python"
```python
from langgraph_sdk import get_client

client = get_client(url=<DEPLOYMENT_URL>)
assistant_id = "agent"
thread = await client.threads.create()
print(thread)
```

=== "JavaScript"
```js
import { Client } from "@langchain/langgraph-sdk";

const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
const assistantID = "agent";
const thread = await client.threads.create();
console.log(thread);
```

=== "CURL"
```bash
curl --request POST \
    --url <DEPLOYMENT_URL>/assistants/search \
    --header 'Content-Type: application/json' \
    --data '{ "limit": 10, "offset": 0 }' | jq -c 'map(select(.config == null or .config == {})) | .[0]' && \
curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}'
```

### Example Response
```json
{
    "thread_id": "9dde5490-2b67-47c8-aa14-4bfec88af217",
    "created_at": "2024-08-30T23:07:38.242730+00:00",
    "updated_at": "2024-08-30T23:07:38.242730+00:00",
    "metadata": {},
    "status": "idle",
    "config": {},
    "values": null
}
```

## Using a Webhook with a Graph Run

To use a webhook, specify the `webhook` parameter in your API request. When the run completes, LangGraph Cloud sends a `POST` request to the specified webhook URL.

For example, if your server listens for webhook events at `https://my-server.app/my-webhook-endpoint`, include this in your request:

=== "Python"
```python
input = { "messages": [{ "role": "user", "content": "Hello!" }] }

async for chunk in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id=assistant_id,
    input=input,
    stream_mode="events",
    webhook="https://my-server.app/my-webhook-endpoint"
):
    pass
```

=== "JavaScript"
```js
const input = { messages: [{ role: "human", content: "Hello!" }] };

const streamResponse = client.runs.stream(
  thread["thread_id"],
  assistantID,
  {
    input: input,
    webhook: "https://my-server.app/my-webhook-endpoint"
  }
);

for await (const chunk of streamResponse) {
  // Handle stream output
}
```

=== "CURL"
```bash
curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data '{
        "assistant_id": <ASSISTANT_ID>,
        "input": {"messages": [{"role": "user", "content": "Hello!"}]},
        "webhook": "https://my-server.app/my-webhook-endpoint"
    }'
```

## Webhook Payload

LangGraph Cloud sends webhook notifications in the format of a [Run](../../concepts/langgraph_server.md/#runs). See the [API Reference](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html#model/run) for details. The request payload includes run input, configuration, and other metadata in the `kwargs` field.

## Securing Webhooks

To ensure only authorized requests hit your webhook endpoint, consider adding a security token as a query parameter:

```
https://my-server.app/my-webhook-endpoint?token=YOUR_SECRET_TOKEN
```

Your server should extract and validate this token before processing requests.

## Testing Webhooks

You can test your webhook using online services like:

- **[Beeceptor](https://beeceptor.com/)** – Quickly create a test endpoint and inspect incoming webhook payloads.
- **[Webhook.site](https://webhook.site/)** – View, debug, and log incoming webhook requests in real time.

These tools help you verify that LangGraph Cloud is correctly triggering and sending webhooks to your service.

---

By following these steps, you can integrate webhooks into your LangGraph Cloud workflow, automating actions based on completed runs.
