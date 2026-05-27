# Thin layer on top of the latest published langgraph-api image.
#
# The base image bundles `langgraph_api`, `langgraph_runtime_postgres`,
# `langgraph_license`, `langgraph_grpc_common`, the Go `core-api-grpc`
# binary, and an entrypoint that starts both the gRPC server and uvicorn
# (`/storage/entrypoint.sh`). It also ships langgraph + langchain-core.
#
# We track the `latest-py3.12` tag rather than pinning a specific revision
# so CI surfaces upstream regressions early. If the base shifts under us,
# `docker compose build` will pick up the new digest on the next run.
#
# The image is the `licensed` variant, so it requires either a real
# `LANGSMITH_API_KEY` or a `LANGGRAPH_CLOUD_LICENSE_KEY` at runtime
# (passed through from the host shell / CI secrets, see docker-compose.yml).

FROM langchain/langgraph-api:latest-py3.12

# Graph dependencies not in the base image. `deepagents` is required for
# the deep_agent graph; the supervisor and researcher use a fake chat
# model (no `langchain-anthropic`) so no LLM API key is needed.
RUN pip install --no-cache-dir \
    "langchain>=1.3.0" \
    "deepagents>=0.6.2"

# Project graphs + registration config.
COPY graph/ /app/graph/
COPY langgraph.json /app/langgraph.json
