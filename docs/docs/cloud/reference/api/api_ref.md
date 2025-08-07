# LangGraph Server API Reference

The LangGraph Server API reference is available within each deployment at the `/docs` endpoint (e.g. `http://localhost:8124/docs`).

Click <a href="/langgraph/cloud/reference/api/api_ref.html" target="_blank">here</a> to view the API reference.

## Authentication

For deployments to LangGraph Platform, authentication is required. Pass the `X-Api-Key` header with each request to the LangGraph Server. The value of the header should be set to a valid LangSmith API key for the organization where the LangGraph Server is deployed.

Example `curl` command:
```shell
curl --request POST \
  --url http://localhost:8124/assistants/search \
  --header 'Content-Type: application/json' \
  --header 'X-Api-Key: LANGSMITH_API_KEY' \
  --data '{
  "metadata": {},
  "limit": 10,
  "offset": 0
}'
```
