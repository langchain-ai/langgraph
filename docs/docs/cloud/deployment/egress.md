# Egress for Subscription Metrics and Operational Metadata

> **Important: Self Hosted Only** 
> This section only applies to customers who are not running in offline mode and assumes you are using a self-hosted LangGraph Platform instance.
> This does not apply to SaaS or Hybrid deployments.

Self-Hosted LangGraph Platform instances store all information locally and will never send sensitive information outside of your network. We currently only track platform usage for billing purposes according to the entitlements in your order. In order to better remotely support our customers, we do require egress to `https://beacon.langchain.com`.

In the future, we will be introducing support diagnostics to help us ensure that the LangGraph Platform is running at an optimal level within your environment.

> **Warning**  
> **This will require egress to `https://beacon.langchain.com` from your network.**
> **If using an API key, you will also need to allow egress to `https://api.smith.langchain.com` or `https://eu.api.smith.langchain.com` for API key verification.**

Generally, data that we send to Beacon can be categorized as follows:

- **Subscription Metrics**
  - Subscription metrics are used to determine level of access and utilization of LangSmith. This includes, but are not limited to:
    - Nodes Executed
    - Runs Executed
    - License Key Verification
- **Operational Metadata**
  - This metadata will contain and collect the above subscription metrics to assist with remote support, allowing the LangChain team to diagnose and troubleshoot performance issues more effectively and proactively.

## Example Payloads

In an effort to maximize transparency, we provide sample payloads here:

### License Verification (If using an Enterprise License)

**Endpoint:**

`POST beacon.langchain.com/v1/beacon/verify`

**Request:**

```json
{
  "license": "<YOUR_LICENSE_KEY>"
}
```

**Response:**

```json
{
  "token": "Valid JWT" // Short-lived JWT token to avoid repeated license checks
}
```

### Api Key Verification (If using a LangSmith API Key)

**Endpoint:**
`POST api.smith.langchain.com/auth`

**Request:**

```json
"Headers": {
  X-Api-Key: <YOUR_API_KEY>
}
```

**Response:**

```json
{
  "org_config": {
    "org_id": "3a1c2b6f-4430-4b92-8a5b-79b8b567bbc1",
    ... // Additional organization details
  }
}
```

### Usage Reporting

**Endpoint:**

`POST beacon.langchain.com/v1/metadata/submit`

**Request:**

```json
{
  "license": "<YOUR_LICENSE_KEY>",
  "from_timestamp": "2025-01-06T09:00:00Z",
  "to_timestamp": "2025-01-06T10:00:00Z",
  "tags": {
    "langgraph.python.version": "0.1.0",
    "langgraph_api.version": "0.2.0",
    "langgraph.platform.revision": "abc123",
    "langgraph.platform.variant": "standard",
    "langgraph.platform.host": "host-1",
    "langgraph.platform.tenant_id": "3a1c2b6f-4430-4b92-8a5b-79b8b567bbc1",
    "langgraph.platform.project_id": "c5b5f53a-4716-4326-8967-d4f7f7799735",
    "langgraph.platform.plan": "enterprise",
    "user_app.uses_indexing": "true",
    "user_app.uses_custom_app": "false",
    "user_app.uses_custom_auth": "true",
    "user_app.uses_thread_ttl": "true",
    "user_app.uses_store_ttl": "false"
  },
  "measures": {
    "langgraph.platform.runs": 150,
    "langgraph.platform.nodes": 450
  },
  "logs": []
}
```

**Response:**

```json
"204 No Content"
```

## Our Commitment

LangChain will not store any sensitive information in the Subscription Metrics or Operational Metadata. Any data collected will not be shared with a third party. If you have any concerns about the data being sent, please reach out to your account team.
