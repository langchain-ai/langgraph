# API Reference

The LangGraph Cloud API reference is available with each deployment at the `/docs` URL path (e.g. `http://localhost:8124/docs`).

Click <a href="/langgraph/cloud/reference/api/api_ref.html" target="_blank">here</a> to view the API reference.

## Authentication

For deployments to LangGraph Cloud, authentication is required. Pass the `X-Api-Key` header with each request to the LangGraph Cloud API. The value of the header should be set to a valid LangSmith API key for the organization where the API is deployed.

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
