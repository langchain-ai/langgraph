FROM ghcr.io/astral-sh/uv:0.13.4 AS uv
FROM postgres:14.7
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_PREFERENCE=only-managed
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.14.3
ARG LANGGRAPH_REVISION
RUN test "${#LANGGRAPH_REVISION}" -eq 40 \
    && git clone https://github.com/langchain-ai/langgraph.git /workspace/langgraph \
    && git -C /workspace/langgraph checkout --detach "$LANGGRAPH_REVISION" \
    && test "$(git -C /workspace/langgraph rev-parse HEAD)" = "$LANGGRAPH_REVISION"
WORKDIR /workspace/langgraph/libs/langgraph
RUN uv sync --frozen --group dev --python 3.14.3 \
    && uv cache clean \
    && .venv/bin/python -c 'from langsmith.sandbox import SandboxClient; from langgraph.checkpoint.postgres import PostgresSaver' \
    && command -v initdb && command -v pg_ctl
ENTRYPOINT []
CMD ["/bin/bash"]
