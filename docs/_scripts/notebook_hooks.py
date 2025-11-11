"""mkdocs hooks for adding custom logic to documentation pipeline.

Lifecycle events: https://www.mkdocs.org/dev-guide/plugins/#events
"""

import json
import logging
import os
import posixpath
import re
from typing import Any, Dict

from bs4 import BeautifulSoup
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files, File
from mkdocs.structure.pages import Page

from _scripts.generate_api_reference_links import update_markdown_with_imports
from _scripts.handle_auto_links import _replace_autolinks
from _scripts.notebook_convert import convert_notebook

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)
DISABLED = os.getenv("DISABLE_NOTEBOOK_CONVERT") in ("1", "true", "True")


REDIRECT_MAP = {
    # lib redirects
    "how-tos/stream-values.ipynb": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "how-tos/stream-updates.ipynb": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "how-tos/streaming-content.ipynb": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "how-tos/stream-multiple.ipynb": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "how-tos/streaming-tokens-without-langchain.ipynb": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "how-tos/streaming-from-final-node.ipynb": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "how-tos/streaming-events-from-within-tools-without-langchain.ipynb": "https://docs.langchain.com/oss/python/langgraph/streaming",
    # graph-api
    "how-tos/state-reducers.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#define-and-update-state",
    "how-tos/sequence.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#create-a-sequence-of-steps",
    "how-tos/branching.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#create-branches",
    "how-tos/recursion-limit.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#create-and-control-loops",
    "how-tos/visualization.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#visualize-your-graph",
    "how-tos/input_output_schema.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#define-input-and-output-schemas",
    "how-tos/pass_private_state.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#pass-private-state-between-nodes",
    "how-tos/state-model.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#use-pydantic-models-for-graph-state",
    "how-tos/map-reduce.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#map-reduce-and-the-send-api",
    "how-tos/command.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#combine-control-flow-and-state-updates-with-command",
    "how-tos/configuration.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#add-runtime-configuration",
    "how-tos/node-retries.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#add-retry-policies",
    "how-tos/return-when-recursion-limit-hits.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#impose-a-recursion-limit",
    "how-tos/async.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api#async",
    # memory how-tos
    "how-tos/memory/manage-conversation-history.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    "how-tos/memory/delete-messages.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory#delete-messages",
    "how-tos/memory/add-summary-conversation-history.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory#summarize-messages",
    "how-tos/memory.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    "agents/memory.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    # subgraph how-tos
    "how-tos/subgraph-transform-state.ipynb": "https://docs.langchain.com/oss/python/langgraph/use-subgraphs#different-state-schemas",
    "how-tos/subgraphs-manage-state.ipynb": "https://docs.langchain.com/oss/python/langgraph/use-subgraphs#add-persistence",
    # persistence how-tos
    "how-tos/persistence_postgres.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory#use-in-production",
    "how-tos/persistence_mongodb.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory#use-in-production",
    "how-tos/persistence_redis.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory#use-in-production",
    "how-tos/subgraph-persistence.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory#use-with-subgraphs",
    "how-tos/cross-thread-persistence.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory#add-long-term-memory",
    "cloud/how-tos/copy_threads": "https://docs.langchain.com/langsmith/use-threads",
    "cloud/how-tos/check-thread-status": "https://docs.langchain.com/langsmith/use-threads",
    "cloud/concepts/threads.md": "https://docs.langchain.com/oss/python/langgraph/persistence#threads",
    "how-tos/persistence.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    # tool calling how-tos
    "how-tos/tool-calling-errors.ipynb": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "how-tos/pass-config-to-tools.ipynb": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "how-tos/pass-run-time-values-to-tools.ipynb": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "how-tos/update-state-from-tools.ipynb": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "agents/tools.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    # multi-agent how-tos
    "how-tos/agent-handoffs.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "how-tos/multi-agent-network.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "how-tos/multi-agent-multi-turn-convo.ipynb": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    # cloud redirects
    "cloud/index.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "cloud/how-tos/index.md": "https://docs.langchain.com/langsmith/home",
    "cloud/concepts/api.md": "https://docs.langchain.com/langsmith/agent-server",
    "cloud/concepts/cloud.md": "https://docs.langchain.com/langsmith/cloud",
    "cloud/faq/studio.md": "https://docs.langchain.com/langsmith/studio",
    "cloud/how-tos/human_in_the_loop_edit_state.md": "https://docs.langchain.com/langsmith/add-human-in-the-loop",
    "cloud/how-tos/human_in_the_loop_user_input.md": "https://docs.langchain.com/langsmith/add-human-in-the-loop",
    "concepts/platform_architecture.md": "https://docs.langchain.com/langsmith/cloud#architecture",
    # cloud streaming redirects
    "cloud/how-tos/stream_values.md": "https://docs.langchain.com/langsmith/streaming",
    "cloud/how-tos/stream_updates.md": "https://docs.langchain.com/langsmith/streaming",
    "cloud/how-tos/stream_messages.md": "https://docs.langchain.com/langsmith/streaming",
    "cloud/how-tos/stream_events.md": "https://docs.langchain.com/langsmith/streaming",
    "cloud/how-tos/stream_debug.md": "https://docs.langchain.com/langsmith/streaming",
    "cloud/how-tos/stream_multiple.md": "https://docs.langchain.com/langsmith/streaming",
    "cloud/concepts/streaming.md": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "agents/streaming.md": "https://docs.langchain.com/oss/python/langgraph/streaming",
    # prebuilt redirects
    "how-tos/create-react-agent.ipynb": "https://docs.langchain.com/oss/python/langchain/agents#basic-configuration",
    "how-tos/create-react-agent-memory.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    "how-tos/create-react-agent-system-prompt.ipynb": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    "how-tos/create-react-agent-structured-output.ipynb": "https://docs.langchain.com/oss/python/langchain/agents#structured-output",
    # misc
    "prebuilt.md": "https://docs.langchain.com/oss/python/langchain/agents",
    "reference/prebuilt.md": "https://reference.langchain.com/python/langgraph/agents/",
    "concepts/high_level.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "concepts/index.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "concepts/v0-human-in-the-loop.md": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "how-tos/index.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "tutorials/introduction.ipynb": "https://docs.langchain.com/oss/python/langgraph/overview",
    "agents/deployment.md": "https://docs.langchain.com/oss/python/langgraph/local-server",
    # deployment redirects
    "how-tos/deploy-self-hosted.md": "https://docs.langchain.com/langsmith/platform-setup",
    "concepts/self_hosted.md": "https://docs.langchain.com/langsmith/platform-setup",
    "tutorials/deployment.md": "https://docs.langchain.com/langsmith/deployments",
    # assistant redirects
    "cloud/how-tos/assistant_versioning.md": "https://docs.langchain.com/langsmith/configuration-cloud",
    "cloud/concepts/runs.md": "https://docs.langchain.com/langsmith/assistants#execution",
    # hitl redirects
    "how-tos/wait-user-input-functional.ipynb": "https://docs.langchain.com/oss/python/langgraph/functional-api",
    "how-tos/review-tool-calls-functional.ipynb": "https://docs.langchain.com/oss/python/langgraph/functional-api",
    "how-tos/create-react-agent-hitl.ipynb": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "agents/human-in-the-loop.md": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "how-tos/human_in_the_loop/dynamic_breakpoints.ipynb": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "concepts/breakpoints.md": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "how-tos/human_in_the_loop/breakpoints.md": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "cloud/how-tos/human_in_the_loop_breakpoint.md": "https://docs.langchain.com/langsmith/add-human-in-the-loop",
    "how-tos/human_in_the_loop/edit-graph-state.ipynb": "https://docs.langchain.com/oss/python/langgraph/use-time-travel",

    # LGP mintlify migration redirects
    "tutorials/auth/getting_started.md": "https://docs.langchain.com/langsmith/auth",
    "tutorials/auth/resource_auth.md": "https://docs.langchain.com/langsmith/resource-auth",
    "tutorials/auth/add_auth_server.md": "https://docs.langchain.com/langsmith/add-auth-server",
    "how-tos/use-remote-graph.md": "https://docs.langchain.com/langsmith/use-remote-graph",
    "how-tos/autogen-integration.md": "https://docs.langchain.com/langsmith/autogen-integration",
    "cloud/how-tos/use_stream_react.md": "https://docs.langchain.com/langsmith/use-stream-react",
    "cloud/how-tos/generative_ui_react.md": "https://docs.langchain.com/langsmith/generative-ui-react",
    "concepts/langgraph_platform.md": "https://docs.langchain.com/langsmith/deployments",
    "concepts/langgraph_components.md": "https://docs.langchain.com/langsmith/components",
    "concepts/langgraph_server.md": "https://docs.langchain.com/langsmith/agent-server",
    "concepts/langgraph_data_plane.md": "https://docs.langchain.com/langsmith/data-plane",
    "concepts/langgraph_control_plane.md": "https://docs.langchain.com/langsmith/control-plane",
    "concepts/langgraph_cli.md": "https://docs.langchain.com/langsmith/cli",
    "concepts/langgraph_studio.md": "https://docs.langchain.com/langsmith/studio",
    "cloud/how-tos/studio/quick_start.md": "https://docs.langchain.com/langsmith/quick-start-studio",
    "cloud/how-tos/invoke_studio.md": "https://docs.langchain.com/langsmith/use-studio#run-application",
    "cloud/how-tos/studio/manage_assistants.md": "https://docs.langchain.com/langsmith/use-studio#manage-assistants",
    "cloud/how-tos/threads_studio.md": "https://docs.langchain.com/langsmith/use-studio#manage-threads",
    "cloud/how-tos/iterate_graph_studio.md": "https://docs.langchain.com/langsmith/observability-studio#iterate-on-prompts",
    "cloud/how-tos/studio/run_evals.md": "https://docs.langchain.com/langsmith/observability-studio#run-experiments-over-a-dataset",
    "cloud/how-tos/clone_traces_studio.md": "https://docs.langchain.com/langsmith/observability-studio#debug-langsmith-traces",
    "cloud/how-tos/datasets_studio.md": "https://docs.langchain.com/langsmith/observability-studio#add-node-to-dataset",
    "concepts/sdk.md": "https://docs.langchain.com/langsmith/sdk",
    "concepts/plans.md": "https://langchain.com/pricing",
    "concepts/application_structure.md": "https://docs.langchain.com/langsmith/application-structure",
    "concepts/scalability_and_resilience.md": "https://docs.langchain.com/langsmith/scalability-and-resilience",
    "concepts/auth.md": "https://docs.langchain.com/langsmith/authentication-methods",
    "how-tos/auth/custom_auth.md": "https://docs.langchain.com/langsmith/custom-auth",
    "how-tos/auth/openapi_security.md": "https://docs.langchain.com/langsmith/openapi-security",
    "concepts/assistants.md": "https://docs.langchain.com/langsmith/assistants",
    "cloud/how-tos/configuration_cloud.md": "https://docs.langchain.com/langsmith/cloud",
    "cloud/how-tos/use_threads.md": "https://docs.langchain.com/langsmith/use-threads",
    "cloud/how-tos/background_run.md": "https://docs.langchain.com/langsmith/background-run",
    "cloud/how-tos/same-thread.md": "https://docs.langchain.com/langsmith/same-thread",
    "cloud/how-tos/stateless_runs.md": "https://docs.langchain.com/langsmith/stateless-runs",
    "cloud/how-tos/configurable_headers.md": "https://docs.langchain.com/langsmith/configurable-headers",
    "concepts/double_texting.md": "https://docs.langchain.com/langsmith/double-texting",
    "cloud/how-tos/interrupt_concurrent.md": "https://docs.langchain.com/langsmith/interrupt-concurrent",
    "cloud/how-tos/rollback_concurrent.md": "https://docs.langchain.com/langsmith/rollback-concurrent",
    "cloud/how-tos/reject_concurrent.md": "https://docs.langchain.com/langsmith/reject-concurrent",
    "cloud/how-tos/enqueue_concurrent.md": "https://docs.langchain.com/langsmith/enqueue-concurrent",
    "cloud/concepts/webhooks.md": "https://docs.langchain.com/langsmith/use-webhooks",
    "cloud/how-tos/webhooks.md": "https://docs.langchain.com/langsmith/use-webhooks",
    "cloud/concepts/cron_jobs.md": "https://docs.langchain.com/langsmith/cron-jobs",
    "cloud/how-tos/cron_jobs.md": "https://docs.langchain.com/langsmith/cron-jobs",
    "how-tos/http/custom_lifespan.md": "https://docs.langchain.com/langsmith/custom-lifespan",
    "how-tos/http/custom_middleware.md": "https://docs.langchain.com/langsmith/custom-middleware",
    "how-tos/http/custom_routes.md": "https://docs.langchain.com/langsmith/custom-routes",
    "cloud/concepts/data_storage_and_privacy.md": "https://docs.langchain.com/langsmith/data-storage-and-privacy",
    "cloud/deployment/semantic_search.md": "https://docs.langchain.com/langsmith/semantic-search",
    "how-tos/ttl/configure_ttl.md": "https://docs.langchain.com/langsmith/configure-ttl",
    "concepts/deployment_options.md": "https://docs.langchain.com/langsmith/platform-setup",
    "cloud/quick_start.md": "https://docs.langchain.com/langsmith/deployment-quickstart",
    "cloud/deployment/setup.md": "https://docs.langchain.com/langsmith/setup-app-requirements-txt",
    "cloud/deployment/setup_pyproject.md": "https://docs.langchain.com/langsmith/setup-pyproject",
    "cloud/deployment/setup_javascript.md": "https://docs.langchain.com/langsmith/setup-javascript",
    "cloud/deployment/custom_docker.md": "https://docs.langchain.com/langsmith/custom-docker",
    "cloud/deployment/graph_rebuild.md": "https://docs.langchain.com/langsmith/graph-rebuild",
    "concepts/langgraph_cloud.md": "https://docs.langchain.com/langsmith/cloud",
    "concepts/langgraph_self_hosted_data_plane.md": "https://docs.langchain.com/langsmith/hybrid",
    "concepts/langgraph_self_hosted_control_plane.md": "https://docs.langchain.com/langsmith/self-hosted",
    "concepts/langgraph_standalone_container.md": "https://docs.langchain.com/langsmith/self-hosted#standalone-server",
    "cloud/deployment/cloud.md": "https://docs.langchain.com/langsmith/cloud",
    "cloud/deployment/self_hosted_data_plane.md": "https://docs.langchain.com/langsmith/deploy-hybrid",
    "cloud/deployment/self_hosted_control_plane.md": "https://docs.langchain.com/langsmith/deploy-self-hosted-full-platform",
    "cloud/deployment/standalone_container.md": "https://docs.langchain.com/langsmith/deploy-standalone-server",
    "concepts/server-mcp.md": "https://docs.langchain.com/langsmith/server-mcp",
    "cloud/how-tos/human_in_the_loop_time_travel.md": "https://docs.langchain.com/langsmith/human-in-the-loop-time-travel",
    "cloud/how-tos/add-human-in-the-loop.md": "https://docs.langchain.com/langsmith/add-human-in-the-loop",
    "cloud/deployment/egress.md": "https://docs.langchain.com/langsmith/env-var",
    "cloud/how-tos/streaming.md": "https://docs.langchain.com/langsmith/streaming",
    "cloud/reference/api/api_ref.md": "https://docs.langchain.com/langsmith/server-api-ref",
    "cloud/reference/langgraph_server_changelog.md": "https://docs.langchain.com/langsmith/agent-server-changelog",
    "cloud/reference/api/api_ref_control_plane.md": "https://docs.langchain.com/langsmith/api-ref-control-plane",
    "cloud/reference/cli.md": "https://docs.langchain.com/langsmith/cli",
    "cloud/reference/env_var.md": "https://docs.langchain.com/langsmith/env-var",
    "troubleshooting/studio.md": "https://docs.langchain.com/langsmith/troubleshooting-studio",

    # LangGraph mintlify migration redirects
    "index.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "agents/agents.md": "https://docs.langchain.com/oss/python/langchain/agents",
    "concepts/why-langgraph.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "tutorials/get-started/1-build-basic-chatbot.md": "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "tutorials/get-started/2-add-tools.md": "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "tutorials/get-started/3-add-memory.md": "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "tutorials/get-started/4-human-in-the-loop.md": "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "tutorials/get-started/5-customize-state.md": "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "tutorials/get-started/6-time-travel.md": "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "tutorials/langsmith/local-server.md": "https://docs.langchain.com/oss/python/langgraph/local-server",
    "tutorials/workflows.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "concepts/agentic_concepts.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "guides/index.md": "https://docs.langchain.com/oss/python/langchain/overview",
    "agents/overview.md": "https://docs.langchain.com/oss/python/langchain/agents",
    "agents/run_agents.md": "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "concepts/low_level.md": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "how-tos/graph-api.md": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "concepts/functional_api.md": "https://docs.langchain.com/oss/python/langgraph/functional-api",
    "how-tos/use-functional-api.md": "https://docs.langchain.com/oss/python/langgraph/functional-api",
    "concepts/pregel.md": "https://docs.langchain.com/oss/python/langgraph/pregel",
    "concepts/streaming.md": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "how-tos/streaming.md": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "concepts/persistence.md": "https://docs.langchain.com/oss/python/langgraph/persistence",
    "concepts/durable_execution.md": "https://docs.langchain.com/oss/python/langgraph/durable-execution",
    "concepts/memory.md": "https://docs.langchain.com/oss/python/langgraph/memory",
    "how-tos/memory/add-memory.md": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    "agents/context.md": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    "agents/models.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "concepts/tools.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "how-tos/tool-calling.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "concepts/human_in_the_loop.md": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "how-tos/human_in_the_loop/add-human-in-the-loop.md": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "concepts/time-travel.md": "https://docs.langchain.com/oss/python/langgraph/persistence",
    "how-tos/human_in_the_loop/time-travel.md": "https://docs.langchain.com/oss/python/langgraph/use-time-travel",
    "concepts/subgraphs.md": "https://docs.langchain.com/oss/python/langgraph/use-subgraphs",
    "how-tos/subgraph.md": "https://docs.langchain.com/oss/python/langgraph/use-subgraphs",
    "concepts/multi_agent.md": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "agents/multi-agent.md": "https://docs.langchain.com/oss/python/langchain/multi-agent",
    "how-tos/multi_agent.md": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "concepts/mcp.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "agents/mcp.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "concepts/tracing.md": "https://docs.langchain.com/oss/python/langgraph/observability",
    "how-tos/enable-tracing.md": "https://docs.langchain.com/oss/python/langgraph/observability",
    "agents/evals.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "examples/index.md": "https://docs.langchain.com/oss/python/langgraph/case-studies",
    "concepts/template_applications.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "tutorials/rag/langgraph_agentic_rag.md": "https://docs.langchain.com/oss/python/langgraph/agentic-rag",
    "tutorials/multi_agent/agent_supervisor.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "tutorials/sql/sql-agent.md": "https://docs.langchain.com/oss/python/langgraph/sql-agent",
    "agents/ui.md": "https://docs.langchain.com/oss/python/langgraph/ui",
    "how-tos/run-id-langsmith.md": "https://docs.langchain.com/oss/python/langgraph/observability",
    "troubleshooting/errors/index.md": "https://docs.langchain.com/oss/python/langgraph/common-errors",
    "troubleshooting/errors/INVALID_CHAT_HISTORY.md": "https://docs.langchain.com/oss/python/langgraph/INVALID_CHAT_HISTORY",
    "troubleshooting/errors/INVALID_LICENSE.md": "https://docs.langchain.com/oss/python/langgraph/common-errors",
    "adopters.md": "https://docs.langchain.com/oss/python/langgraph/case-studies",
    "concepts/faq.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "agents/prebuilt.md": "https://docs.langchain.com/oss/python/langchain/agents",
    "reference/index.md": "https://reference.langchain.com/python/langgraph/",
    "reference/graphs.md": "https://reference.langchain.com/python/langgraph/graphs/",
    "reference/func.md": "https://reference.langchain.com/python/langgraph/func/",
    "reference/pregel.md": "https://reference.langchain.com/python/langgraph/pregel/",
    "reference/checkpoints.md": "https://reference.langchain.com/python/langgraph/checkpoints/",
    "reference/store.md": "https://reference.langchain.com/python/langgraph/store/",
    "reference/cache.md": "https://reference.langchain.com/python/langgraph/cache/",
    "reference/types.md": "https://reference.langchain.com/python/langgraph/types/",
    "reference/runtime.md": "https://reference.langchain.com/python/langgraph/runtime/",
    "reference/config.md": "https://reference.langchain.com/python/langgraph/config/",
    "reference/errors.md": "https://reference.langchain.com/python/langgraph/errors/",
    "reference/constants.md": "https://reference.langchain.com/python/langgraph/constants/",
    "reference/channels.md": "https://reference.langchain.com/python/langgraph/channels/",
    "reference/agents.md": "https://reference.langchain.com/python/langgraph/agents/",
    "reference/supervisor.md": "https://reference.langchain.com/python/langgraph/supervisor/",
    "reference/swarm.md": "https://reference.langchain.com/python/langgraph/swarm/",
    "reference/mcp.md": "https://reference.langchain.com/python/langgraph/mcp/",
    "cloud/reference/sdk/python_sdk_ref.md": "https://reference.langchain.com/python/langsmith/deployment/sdk/",
    "reference/remote_graph.md": "https://reference.langchain.com/python/langsmith/deployment/remote_graph/",

    # additional exclude-search entries from mkdocs.yml
    "additional-resources/index.md": "https://docs.langchain.com/oss/python/langchain/overview",
    "cloud/concepts/cron_jobs.md": "https://docs.langchain.com/langsmith/cron-jobs",
    "cloud/concepts/data_storage_and_privacy.md": "https://docs.langchain.com/langsmith/data-storage-and-privacy",
    "cloud/concepts/webhooks.md": "https://docs.langchain.com/langsmith/use-webhooks",
    "cloud/deployment/cloud.md": "https://docs.langchain.com/langsmith/cloud",
    "cloud/deployment/custom_docker.md": "https://docs.langchain.com/langsmith/custom-docker",
    "cloud/deployment/egress.md": "https://docs.langchain.com/langsmith/env-var",
    "cloud/deployment/graph_rebuild.md": "https://docs.langchain.com/langsmith/graph-rebuild",
    "cloud/deployment/self_hosted_control_plane.md": "https://docs.langchain.com/langsmith/platform-setup",
    "cloud/deployment/self_hosted_data_plane.md": "https://docs.langchain.com/langsmith/platform-setup",
    "cloud/deployment/semantic_search.md": "https://docs.langchain.com/langsmith/semantic-search",
    "cloud/deployment/setup_javascript.md": "https://docs.langchain.com/langsmith/setup-javascript",
    "cloud/deployment/setup_pyproject.md": "https://docs.langchain.com/langsmith/setup-pyproject",
    "cloud/deployment/setup.md": "https://docs.langchain.com/langsmith/setup-app-requirements-txt",
    "cloud/deployment/standalone_container.md": "https://docs.langchain.com/langsmith/docker",
    "cloud/how-tos/add-human-in-the-loop.md": "https://docs.langchain.com/langsmith/add-human-in-the-loop",
    "cloud/how-tos/background_run.md": "https://docs.langchain.com/langsmith/background-run",
    "cloud/how-tos/clone_traces_studio.md": "https://docs.langchain.com/langsmith/observability",
    "cloud/how-tos/configurable_headers.md": "https://docs.langchain.com/langsmith/configurable-headers",
    "cloud/how-tos/configuration_cloud.md": "https://docs.langchain.com/langsmith/configuration-cloud",
    "cloud/how-tos/cron_jobs.md": "https://docs.langchain.com/langsmith/cron-jobs",
    "cloud/how-tos/datasets_studio.md": "https://docs.langchain.com/langsmith/use-studio",
    "cloud/how-tos/enqueue_concurrent.md": "https://docs.langchain.com/langsmith/enqueue-concurrent",
    "cloud/how-tos/generative_ui_react.md": "https://docs.langchain.com/langsmith/generative-ui-react",
    "cloud/how-tos/human_in_the_loop_time_travel.md": "https://docs.langchain.com/langsmith/human-in-the-loop-time-travel",
    "cloud/how-tos/interrupt_concurrent.md": "https://docs.langchain.com/langsmith/interrupt-concurrent",
    "cloud/how-tos/invoke_studio.md": "https://docs.langchain.com/langsmith/use-studio",
    "cloud/how-tos/iterate_graph_studio.md": "https://docs.langchain.com/langsmith/use-studio",
    "cloud/how-tos/reject_concurrent.md": "https://docs.langchain.com/langsmith/reject-concurrent",
    "cloud/how-tos/rollback_concurrent.md": "https://docs.langchain.com/langsmith/rollback-concurrent",
    "cloud/how-tos/same-thread.md": "https://docs.langchain.com/langsmith/same-thread",
    "cloud/how-tos/stateless_runs.md": "https://docs.langchain.com/langsmith/stateless-runs",
    "cloud/how-tos/streaming.md": "https://docs.langchain.com/langsmith/streaming",
    "cloud/how-tos/studio/manage_assistants.md": "https://docs.langchain.com/langsmith/use-studio",
    "cloud/how-tos/studio/quick_start.md": "https://docs.langchain.com/langsmith/quick-start-studio",
    "cloud/how-tos/studio/run_evals.md": "https://docs.langchain.com/langsmith/observability",
    "cloud/how-tos/threads_studio.md": "https://docs.langchain.com/langsmith/use-threads",
    "cloud/how-tos/use_stream_react.md": "https://docs.langchain.com/langsmith/use-stream-react",
    "cloud/how-tos/use_threads.md": "https://docs.langchain.com/langsmith/use-threads",
    "cloud/how-tos/webhooks.md": "https://docs.langchain.com/langsmith/use-webhooks",
    "cloud/quick_start.md": "https://docs.langchain.com/langsmith/deployment-quickstart",
    "cloud/reference/api/api_ref_control_plane.md": "https://docs.langchain.com/langsmith/api-ref-control-plane",
    "cloud/reference/api/api_ref.md": "https://docs.langchain.com/langsmith/server-api-ref",
    "cloud/reference/cli.md": "https://docs.langchain.com/langsmith/cli",
    "cloud/reference/env_var.md": "https://docs.langchain.com/langsmith/env-var",
    "cloud/reference/langgraph_server_changelog.md": "https://docs.langchain.com/langsmith/agent-server-changelog",
    "cloud/reference/sdk/js_ts_sdk_ref.md": "https://reference.langchain.com/javascript/modules/langsmith.html",
    "concepts/application_structure.md": "https://docs.langchain.com/langsmith/application-structure",
    "concepts/assistants.md": "https://docs.langchain.com/langsmith/assistants",
    "concepts/auth.md": "https://docs.langchain.com/langsmith/auth",
    "concepts/deployment_options.md": "https://docs.langchain.com/langsmith/deployments",
    "concepts/double_texting.md": "https://docs.langchain.com/langsmith/double-texting",
    "concepts/faq.md": "https://docs.langchain.com/langsmith/faq",
    "concepts/langgraph_cli.md": "https://docs.langchain.com/langsmith/cli",
    "concepts/langgraph_cloud.md": "https://docs.langchain.com/langsmith/cloud",
    "concepts/langgraph_components.md": "https://docs.langchain.com/langsmith/components",
    "concepts/langgraph_control_plane.md": "https://docs.langchain.com/langsmith/control-plane",
    "concepts/langgraph_data_plane.md": "https://docs.langchain.com/langsmith/data-plane",
    "concepts/langgraph_platform.md": "https://docs.langchain.com/langsmith/home",
    "concepts/langgraph_self_hosted_control_plane.md": "https://docs.langchain.com/langsmith/platform-setup",
    "concepts/langgraph_self_hosted_data_plane.md": "https://docs.langchain.com/langsmith/platform-setup",
    "concepts/langgraph_server.md": "https://docs.langchain.com/langsmith/agent-server",
    "concepts/langgraph_standalone_container.md": "https://docs.langchain.com/langsmith/docker",
    "concepts/langgraph_studio.md": "https://docs.langchain.com/langsmith/studio",
    "concepts/plans.md": "https://docs.langchain.com/langsmith/home",
    "concepts/scalability_and_resilience.md": "https://docs.langchain.com/langsmith/scalability-and-resilience",
    "concepts/sdk.md": "https://docs.langchain.com/langsmith/sdk",
    "concepts/server-mcp.md": "https://docs.langchain.com/langsmith/server-mcp",
    "concepts/template_applications.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "concepts/why-langgraph.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "examples/index.md": "https://docs.langchain.com/oss/python/langgraph/case-studies",
    "guides/index.md": "https://docs.langchain.com/oss/python/langchain/overview",
    "how-tos/auth/custom_auth.md": "https://docs.langchain.com/langsmith/custom-auth",
    "how-tos/auth/openapi_security.md": "https://docs.langchain.com/langsmith/openapi-security",
    "how-tos/autogen-integration.md": "https://docs.langchain.com/langsmith/autogen-integration",
    "how-tos/http/custom_lifespan.md": "https://docs.langchain.com/langsmith/custom-lifespan",
    "how-tos/http/custom_middleware.md": "https://docs.langchain.com/langsmith/custom-middleware",
    "how-tos/http/custom_routes.md": "https://docs.langchain.com/langsmith/custom-routes",
    "how-tos/ttl/configure_ttl.md": "https://docs.langchain.com/langsmith/configure-ttl",
    "how-tos/use-remote-graph.md": "https://docs.langchain.com/langsmith/use-remote-graph",
    "index.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "snippets/chat_model_tabs.md": "https://docs.langchain.com/oss/python/langchain/overview",
    "troubleshooting/errors/GRAPH_RECURSION_LIMIT.md": "https://docs.langchain.com/oss/python/langgraph/GRAPH_RECURSION_LIMIT",
    "troubleshooting/errors/index.md": "https://docs.langchain.com/oss/python/langgraph/common-errors",
    "troubleshooting/errors/INVALID_CHAT_HISTORY.md": "https://docs.langchain.com/oss/python/langgraph/INVALID_CHAT_HISTORY",
    "troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE.md": "https://docs.langchain.com/oss/python/langgraph/INVALID_CONCURRENT_GRAPH_UPDATE",
    "troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE.md": "https://docs.langchain.com/oss/python/langgraph/INVALID_GRAPH_NODE_RETURN_VALUE",
    "troubleshooting/errors/INVALID_LICENSE.md": "https://docs.langchain.com/oss/python/langgraph/common-errors",
    "troubleshooting/errors/MULTIPLE_SUBGRAPHS.md": "https://docs.langchain.com/oss/python/langgraph/MULTIPLE_SUBGRAPHS",
    "troubleshooting/studio.md": "https://docs.langchain.com/langsmith/troubleshooting-studio",
    "tutorials/auth/add_auth_server.md": "https://docs.langchain.com/langsmith/add-auth-server",
    "tutorials/auth/getting_started.md": "https://docs.langchain.com/langsmith/auth",
    "tutorials/auth/resource_auth.md": "https://docs.langchain.com/langsmith/resource-auth",
    "agents/agents.md": "https://docs.langchain.com/oss/python/langchain/agents",
    "concepts/why-langgraph.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "tutorials/langsmith/local-server.md": "https://docs.langchain.com/oss/python/langgraph/local-server",
    "tutorials/workflows.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "concepts/agentic_concepts.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "guides/index.md": "https://docs.langchain.com/oss/python/langchain/overview",
    "agents/overview.md": "https://docs.langchain.com/oss/python/langchain/agents",
    "concepts/agentic_concepts.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "agents/run_agents.md": "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "concepts/low_level.md": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "how-tos/graph-api.md": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "concepts/functional_api.md": "https://docs.langchain.com/oss/python/langgraph/functional-api",
    "how-tos/use-functional-api.md": "https://docs.langchain.com/oss/python/langgraph/functional-api",
    "concepts/pregel.md": "https://docs.langchain.com/oss/python/langgraph/pregel",
    "concepts/streaming.md": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "how-tos/streaming.md": "https://docs.langchain.com/oss/python/langgraph/streaming",
    "concepts/persistence.md": "https://docs.langchain.com/oss/python/langgraph/persistence",
    "concepts/durable_execution.md": "https://docs.langchain.com/oss/python/langgraph/durable-execution",
    "concepts/memory.md": "https://docs.langchain.com/oss/python/langgraph/memory",
    "how-tos/memory/add-memory.md": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    "agents/context.md": "https://docs.langchain.com/oss/python/langgraph/add-memory",
    "agents/models.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "concepts/tools.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "how-tos/tool-calling.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "concepts/human_in_the_loop.md": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "how-tos/human_in_the_loop/add-human-in-the-loop.md": "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "concepts/time-travel.md": "https://docs.langchain.com/oss/python/langgraph/persistence",
    "how-tos/human_in_the_loop/time-travel.md": "https://docs.langchain.com/oss/python/langgraph/use-time-travel",
    "concepts/subgraphs.md": "https://docs.langchain.com/oss/python/langgraph/use-subgraphs",
    "how-tos/subgraph.md": "https://docs.langchain.com/oss/python/langgraph/use-subgraphs",
    "concepts/multi_agent.md": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "agents/multi-agent.md": "https://docs.langchain.com/oss/python/langchain/multi-agent",
    "how-tos/multi_agent.md": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "concepts/mcp.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "agents/mcp.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "concepts/tracing.md": "https://docs.langchain.com/oss/python/langgraph/observability",
    "how-tos/enable-tracing.md": "https://docs.langchain.com/oss/python/langgraph/observability",
    "agents/evals.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "examples/index.md": "https://docs.langchain.com/oss/python/langgraph/case-studies",
    "concepts/template_applications.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "tutorials/rag/langgraph_agentic_rag.md": "https://docs.langchain.com/oss/python/langgraph/agentic-rag",
    "tutorials/multi_agent/agent_supervisor.md": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "tutorials/sql/sql-agent.md": "https://docs.langchain.com/oss/python/langgraph/sql-agent",
    "agents/ui.md": "https://docs.langchain.com/oss/python/langgraph/ui",
    "how-tos/run-id-langsmith.md": "https://docs.langchain.com/oss/python/langgraph/observability",
    "troubleshooting/errors/index.md": "https://docs.langchain.com/oss/python/langgraph/common-errors",
    "troubleshooting/errors/GRAPH_RECURSION_LIMIT.md": "https://docs.langchain.com/oss/python/langgraph/GRAPH_RECURSION_LIMIT",
    "troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE.md": "https://docs.langchain.com/oss/python/langgraph/INVALID_CONCURRENT_GRAPH_UPDATE",
    "troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE.md": "https://docs.langchain.com/oss/python/langgraph/INVALID_GRAPH_NODE_RETURN_VALUE",
    "troubleshooting/errors/MULTIPLE_SUBGRAPHS.md": "https://docs.langchain.com/oss/python/langgraph/MULTIPLE_SUBGRAPHS",
    "troubleshooting/errors/INVALID_CHAT_HISTORY.md": "https://docs.langchain.com/oss/python/langgraph/INVALID_CHAT_HISTORY",
    "troubleshooting/errors/INVALID_LICENSE.md": "https://docs.langchain.com/oss/python/langgraph/common-errors",
    "adopters.md": "https://docs.langchain.com/oss/python/langgraph/case-studies",
    "concepts/faq.md": "https://docs.langchain.com/oss/python/langgraph/overview",
    "agents/prebuilt.md": "https://docs.langchain.com/oss/python/langchain/agents",
}


class NotebookFile(File):
    def is_documentation_page(self):
        return True


def on_files(files: Files, **kwargs: Dict[str, Any]):
    if DISABLED:
        return files
    new_files = Files([])
    for file in files:
        if file.src_path.endswith(".ipynb"):
            new_file = NotebookFile(
                path=file.src_path,
                src_dir=file.src_dir,
                dest_dir=file.dest_dir,
                use_directory_urls=file.use_directory_urls,
            )
            new_files.append(new_file)
        else:
            new_files.append(file)
    return new_files


def _add_path_to_code_blocks(markdown: str, page: Page) -> str:
    """Add the path to the code blocks."""
    code_block_pattern = re.compile(
        r"(?P<indent>[ \t]*)```(?P<language>\w+)[ ]*(?P<attributes>[^\n]*)\n"
        r"(?P<code>((?:.*\n)*?))"  # Capture the code inside the block using named group
        r"(?P=indent)```"  # Match closing backticks with the same indentation
    )

    def replace_code_block_header(match: re.Match) -> str:
        indent = match.group("indent")
        language = match.group("language")
        attributes = match.group("attributes").rstrip()

        if 'exec="on"' not in attributes:
            # Return original code block
            return match.group(0)

        code = match.group("code")
        return f'{indent}```{language} {attributes} path="{page.file.src_path}"\n{code}{indent}```'

    return code_block_pattern.sub(replace_code_block_header, markdown)


# Compiled regex patterns for better performance and readability


def _apply_conditional_rendering(md_text: str, target_language: str) -> str:
    if target_language not in {"python", "js"}:
        raise ValueError("target_language must be 'python' or 'js'")

    pattern = re.compile(
        r"(?P<indent>[ \t]*):::(?P<language>\w+)\s*\n"
        r"(?P<content>((?:.*\n)*?))"  # Capture the content inside the block
        r"(?P=indent)[ \t]*:::"  # Match closing with the same indentation + any additional whitespace
    )

    def replace_conditional_blocks(match: re.Match) -> str:
        """Keep active conditionals."""
        language = match.group("language")
        content = match.group("content")

        if language not in {"python", "js"}:
            # If the language is not supported, return the original block
            return match.group(0)

        if language == target_language:
            return content

        # If the language does not match, return an empty string
        return ""

    processed = pattern.sub(replace_conditional_blocks, md_text)
    return processed


def _highlight_code_blocks(markdown: str) -> str:
    """Find code blocks with highlight comments and add hl_lines attribute.

    Args:
        markdown: The markdown content to process.

    Returns:
        updated Markdown code with code blocks containing highlight comments
        updated to use the hl_lines attribute.
    """
    # Pattern to find code blocks with highlight comments and without
    # existing hl_lines for Python and JavaScript
    # Pattern to find code blocks with highlight comments, handling optional indentation
    code_block_pattern = re.compile(
        r"(?P<indent>[ \t]*)```(?P<language>\w+)[ ]*(?P<attributes>[^\n]*)\n"
        r"(?P<code>((?:.*\n)*?))"  # Capture the code inside the block using named group
        r"(?P=indent)```"  # Match closing backticks with the same indentation
    )

    def replace_highlight_comments(match: re.Match) -> str:
        indent = match.group("indent")
        language = match.group("language")
        code_block = match.group("code")
        attributes = match.group("attributes").rstrip()

        # Account for a case where hl_lines is manually specified
        if "hl_lines" in attributes:
            # Return original code block
            return match.group(0)

        lines = code_block.split("\n")
        highlighted_lines = []

        # Skip initial empty lines
        while lines and not lines[0].strip():
            lines.pop(0)

        lines_to_keep = []

        comment_syntax = (
            "# highlight-next-line"
            if language in ["py", "python"]
            else "// highlight-next-line"
        )

        for line in lines:
            if comment_syntax in line:
                count = len(lines_to_keep) + 1
                highlighted_lines.append(str(count))
            else:
                lines_to_keep.append(line)

        # Reconstruct the new code block
        new_code_block = "\n".join(lines_to_keep)

        # Construct the full code block that also includes
        # the fenced code block syntax.
        opening_fence = f"```{language}"

        if attributes:
            opening_fence += f" {attributes}"

        if highlighted_lines:
            opening_fence += f' hl_lines="{" ".join(highlighted_lines)}"'

        return (
            # The indent and opening fence
            f"{indent}{opening_fence}\n"
            # The indent and terminating \n is already included in the code block
            f"{new_code_block}"
            f"{indent}```"
        )

    # Replace all code blocks in the markdown
    markdown = code_block_pattern.sub(replace_highlight_comments, markdown)
    return markdown


def _save_page_output(markdown: str, output_path: str):
    """Save markdown content to a file, creating parent directories if needed.

    Args:
        markdown: The markdown content to save
        output_path: The file path to save to
    """
    # Create parent directories recursively if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the markdown content to the file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)


def _on_page_markdown_with_config(
    markdown: str,
    page: Page,
    *,
    add_api_references: bool = True,
    remove_base64_images: bool = False,
    **kwargs: Any,
) -> str:
    if DISABLED:
        return markdown

    if page.file.src_path.endswith(".ipynb"):
        # logger.info("Processing Jupyter notebook: %s", page.file.src_path)
        markdown = convert_notebook(page.file.abs_src_path)

    target_language = kwargs.get(
        "target_language",
        os.environ.get("TARGET_LANGUAGE", "python")
    )

    # Apply cross-reference preprocessing to all markdown content
    markdown = _replace_autolinks(markdown, page.file.src_path, default_scope=target_language)

    # Append API reference links to code blocks
    if add_api_references:
        markdown = update_markdown_with_imports(markdown, page.file.abs_src_path)
    # Apply highlight comments to code blocks
    markdown = _highlight_code_blocks(markdown)

    # Apply conditional rendering for code blocks
    markdown = _apply_conditional_rendering(markdown, target_language)

    # Add file path as an attribute to code blocks that are executable.
    # This file path is used to associate fixtures with the executable code
    # which can be used in CI to test the docs without making network requests.
    markdown = _add_path_to_code_blocks(markdown, page)

    if remove_base64_images:
        # Remove base64 encoded images from markdown
        markdown = re.sub(r"!\[.*?\]\(data:image/[^;]+;base64,[^)]+\)", "", markdown)

    return markdown


def on_page_markdown(markdown: str, page: Page, **kwargs: Dict[str, Any]):
    finalized_markdown = _on_page_markdown_with_config(
        markdown,
        page,
        add_api_references=True,
        **kwargs,
    )
    page.meta["original_markdown"] = finalized_markdown

    output_path = os.environ.get("MD_OUTPUT_PATH")
    if output_path:
        file_path = os.path.join(output_path, page.file.src_path)
        _save_page_output(finalized_markdown, file_path)

    return finalized_markdown


# redirects

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Redirecting...</title>
    <link rel="canonical" href="{url}">
    <meta name="robots" content="noindex">
    <script>var anchor=window.location.hash.substr(1);location.href="{url}"+(anchor?"#"+anchor:"")</script>
    <meta http-equiv="refresh" content="0; url={url}">
</head>
<body>
Redirecting...
</body>
</html>
"""


def _write_html(site_dir, old_path, new_path):
    """Write an HTML file in the site_dir with a meta redirect to the new page"""
    # Determine all relevant paths
    old_path_abs = os.path.join(site_dir, old_path)
    old_dir_abs = os.path.dirname(old_path_abs)

    # Create parent directories if they don't exist
    if not os.path.exists(old_dir_abs):
        os.makedirs(old_dir_abs)

    # Write the HTML redirect file in place of the old file
    content = HTML_TEMPLATE.format(url=new_path)
    with open(old_path_abs, "w", encoding="utf-8") as f:
        f.write(content)


def _inject_gtm(html: str) -> str:
    """Inject Google Tag Manager code into the HTML.

    Code to inject Google Tag Manager noscript tag immediately after <body>.

    This is done via hooks rather than via a template because the MkDocs material
    theme does not seem to allow placing the code immediately after the <body> tag
    without modifying the template files directly.

    Args:
        html: The HTML content to modify.

    Returns:
        The modified HTML content with GTM code injected.
    """
    # Code was copied from Google Tag Manager setup instructions.
    gtm_code = """
<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-T35S4S46"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->
"""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body
    if body:
        # Insert the GTM code as raw HTML at the top of <body>
        body.insert(0, BeautifulSoup(gtm_code, "html.parser"))
        return str(soup)
    else:
        return html  # fallback if no <body> found


def _inject_markdown_into_html(html: str, page: Page) -> str:
    """Inject the original markdown content into the HTML page as JSON."""
    original_markdown = page.meta.get("original_markdown", "")
    if not original_markdown:
        return html
    markdown_data = {
        "markdown": original_markdown,
        "title": page.title or "Page Content",
        "url": page.url or "",
    }

    # Properly escape the JSON for HTML
    json_content = json.dumps(markdown_data, ensure_ascii=False)

    json_content = (
        json_content.replace("</", "\\u003c/")
        .replace("<script", "\\u003cscript")
        .replace("</script", "\\u003c/script")
    )

    script_content = (
        f'<script id="page-markdown-content" '
        f'type="application/json">{json_content}</script>'
    )

    # Insert before </head> if it exists, otherwise before </body>
    if "</head>" not in html:
        raise ValueError(
            "HTML does not contain </head> tag. Cannot inject markdown content."
        )
    return html.replace("</head>", f"{script_content}</head>")


def on_post_page(html: str, page: Page, config: MkDocsConfig) -> str:
    """Inject Google Tag Manager noscript tag immediately after <body>.

    Args:
        html: The HTML output of the page.
        page: The page instance.
        config: The MkDocs configuration object.

    Returns:
        modified HTML output with GTM code injected.
    """
    html = _inject_markdown_into_html(html, page)
    return _inject_gtm(html)


# Create HTML files for redirects after site dir has been built
def on_post_build(config):
    use_directory_urls = config.get("use_directory_urls")
    site_dir = config["site_dir"]

    # Track which paths have explicit redirects
    redirected_paths = set()

    # Process explicit redirects from REDIRECT_MAP
    for page_old, page_new in REDIRECT_MAP.items():
        # Convert .ipynb to .md for path calculation
        page_old = page_old.replace(".ipynb", ".md")

        # Calculate the HTML path for the old page (whether it exists or not)
        if use_directory_urls:
            # With directory URLs: /path/to/page/ becomes /path/to/page/index.html
            if page_old.endswith(".md"):
                old_html_path = page_old[:-3] + "/index.html"
            else:
                old_html_path = page_old + "/index.html"
        else:
            # Without directory URLs: /path/to/page.md becomes /path/to/page.html
            if page_old.endswith(".md"):
                old_html_path = page_old[:-3] + ".html"
            else:
                old_html_path = page_old + ".html"

        # Track this path as redirected
        redirected_paths.add(old_html_path)

        if isinstance(page_new, str) and page_new.startswith("http"):
            # Handle external redirects
            _write_html(site_dir, old_html_path, page_new)
        else:
            # Handle internal redirects
            page_new = page_new.replace(".ipynb", ".md")
            page_new_before_hash, hash, suffix = page_new.partition("#")

            # Try to get the new path using File class, but fallback to manual calculation
            try:
                new_html_path = File(page_new_before_hash, "", "", True).url
                new_html_path = (
                    posixpath.relpath(new_html_path, start=posixpath.dirname(old_html_path))
                    + hash
                    + suffix
                )
            except:
                # Fallback: calculate relative path manually
                if use_directory_urls:
                    if page_new_before_hash.endswith(".md"):
                        new_html_path = page_new_before_hash[:-3] + "/"
                    else:
                        new_html_path = page_new_before_hash + "/"
                else:
                    if page_new_before_hash.endswith(".md"):
                        new_html_path = page_new_before_hash[:-3] + ".html"
                    else:
                        new_html_path = page_new_before_hash + ".html"
                new_html_path += hash + suffix

            _write_html(site_dir, old_html_path, new_html_path)

    # Create root index.html redirect
    root_redirect_html = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Redirecting to LangGraph Documentation</title>
    <link rel="canonical" href="https://docs.langchain.com/oss/python/langgraph/overview">
    <meta name="robots" content="noindex">
    <script>var anchor=window.location.hash.substr(1);location.href="https://docs.langchain.com/oss/python/langgraph/overview"+(anchor?"#"+anchor:"")</script>
    <meta http-equiv="refresh" content="0; url=https://docs.langchain.com/oss/python/langgraph/overview">
</head>
<body>
<h1>Documentation has moved</h1>
<p>The LangGraph documentation has moved to <a href="https://docs.langchain.com/oss/python/langgraph/overview">docs.langchain.com</a>.</p>
<p>Redirecting you now...</p>
</body>
</html>
"""

    root_index_path = os.path.join(site_dir, "index.html")
    with open(root_index_path, "w", encoding="utf-8") as f:
        f.write(root_redirect_html)

    # Create server-side catch-all redirect file for Netlify/Cloudflare Pages
    # This handles any pages not explicitly mapped in REDIRECT_MAP
    # Note: This won't work on GitHub Pages, but kept for potential future use
    redirects_content = """# Netlify/Cloudflare Pages redirect rules
# Specific redirects are handled by individual HTML redirect pages
# This is the catch-all for any unmapped pages

# Exclude reference docs from catch-all
/reference/*  200

# Catch-all: redirect any page not explicitly mapped
/*  https://docs.langchain.com/oss/python/langgraph/overview  301
"""

    redirects_path = os.path.join(site_dir, "_redirects")
    with open(redirects_path, "w", encoding="utf-8") as f:
        f.write(redirects_content)
