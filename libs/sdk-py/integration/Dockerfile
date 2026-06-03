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

# Swap the published langgraph *core* for this monorepo's local copy, so the
# server executes the langgraph under test rather than the latest release.
# We keep the rest of the base image (langgraph-api, runtime, Go core server)
# on `latest` — that still surfaces upstream regressions — but the core the
# server runs is now the PR's, which is what lets this suite catch core
# regressions (e.g. ensure_config / runtime changes) before they're published.
# `--no-deps` keeps the base image's already-compatible
# checkpoint/prebuilt/sdk; we only replace core. The local source comes from
# the `langgraph_src` additional build context (see docker-compose.yml).
COPY --from=langgraph_src pyproject.toml README.md LICENSE /opt/langgraph-src/
COPY --from=langgraph_src langgraph /opt/langgraph-src/langgraph/
RUN pip install --no-cache-dir --force-reinstall --no-deps /opt/langgraph-src

# Project graphs + registration config.
COPY graph/ /app/graph/
COPY langgraph.json /app/langgraph.json
