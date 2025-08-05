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
    "how-tos/stream-values.ipynb": "how-tos/streaming.md#stream-graph-state",
    "how-tos/stream-updates.ipynb": "how-tos/streaming.md#stream-graph-state",
    "how-tos/streaming-content.ipynb": "how-tos/streaming.md",
    "how-tos/stream-multiple.ipynb": "how-tos/streaming.md#stream-multiple-nodes",
    "how-tos/streaming-tokens-without-langchain.ipynb": "how-tos/streaming.md#use-with-any-llm",
    "how-tos/streaming-from-final-node.ipynb": "how-tos/streaming-specific-nodes.ipynb",
    "how-tos/streaming-events-from-within-tools-without-langchain.ipynb": "how-tos/streaming-events-from-within-tools.ipynb#example-without-langchain",
    # graph-api
    "how-tos/state-reducers.ipynb": "how-tos/graph-api.md#define-and-update-state",
    "how-tos/sequence.ipynb": "how-tos/graph-api.md#create-a-sequence-of-steps",
    "how-tos/branching.ipynb": "how-tos/graph-api.md#create-branches",
    "how-tos/recursion-limit.ipynb": "how-tos/graph-api.md#create-and-control-loops",
    "how-tos/visualization.ipynb": "how-tos/graph-api.md#visualize-your-graph",
    "how-tos/input_output_schema.ipynb": "how-tos/graph-api.md#define-input-and-output-schemas",
    "how-tos/pass_private_state.ipynb": "how-tos/graph-api.md#pass-private-state-between-nodes",
    "how-tos/state-model.ipynb": "how-tos/graph-api.md#use-pydantic-models-for-graph-state",
    "how-tos/map-reduce.ipynb": "how-tos/graph-api.md#map-reduce-and-the-send-api",
    "how-tos/command.ipynb": "how-tos/graph-api.md#combine-control-flow-and-state-updates-with-command",
    "how-tos/configuration.ipynb": "how-tos/graph-api.md#add-runtime-configuration",
    "how-tos/node-retries.ipynb": "how-tos/graph-api.md#add-retry-policies",
    "how-tos/return-when-recursion-limit-hits.ipynb": "how-tos/graph-api.md#impose-a-recursion-limit",
    "how-tos/async.ipynb": "how-tos/graph-api.md#async",
    # memory how-tos
    "how-tos/memory/manage-conversation-history.ipynb": "how-tos/memory/add-memory.md",
    "how-tos/memory/delete-messages.ipynb": "how-tos/memory/add-memory.md#delete-messages",
    "how-tos/memory/add-summary-conversation-history.ipynb": "how-tos/memory/add-memory.md#summarize-messages",
    "how-tos/memory.ipynb": "how-tos/memory/add-memory.md",
    "agents/memory.ipynb": "how-tos/memory/add-memory.md",
    # subgraph how-tos
    "how-tos/subgraph-transform-state.ipynb": "how-tos/subgraph.md#different-state-schemas",
    "how-tos/subgraphs-manage-state.ipynb": "how-tos/subgraph.md#add-persistence",
    # persistence how-tos
    "how-tos/persistence_postgres.ipynb": "how-tos/memory/add-memory.md#use-in-production",
    "how-tos/persistence_mongodb.ipynb": "how-tos/memory/add-memory.md#use-in-production",
    "how-tos/persistence_redis.ipynb": "how-tos/memory/add-memory.md#use-in-production",
    "how-tos/subgraph-persistence.ipynb": "how-tos/memory/add-memory.md#use-with-subgraphs",
    "how-tos/cross-thread-persistence.ipynb": "how-tos/memory/add-memory.md#add-long-term-memory",
    "cloud/how-tos/copy_threads": "cloud/how-tos/use_threads",
    "cloud/how-tos/check-thread-status": "cloud/how-tos/use_threads",
    "cloud/concepts/threads.md": "concepts/persistence.md#threads",
    "how-tos/persistence.ipynb": "how-tos/memory/add-memory.md",
    # tool calling how-tos
    "how-tos/tool-calling-errors.ipynb": "how-tos/tool-calling.ipynb#handle-errors",
    "how-tos/pass-config-to-tools.ipynb": "how-tos/tool-calling.ipynb#access-config",
    "how-tos/pass-run-time-values-to-tools.ipynb": "how-tos/tool-calling.ipynb#read-state",
    "how-tos/update-state-from-tools.ipynb": "how-tos/tool-calling.ipynb#update-state",
    "agents/tools.md": "how-tos/tool-calling.md",
    # multi-agent how-tos
    "how-tos/agent-handoffs.ipynb": "how-tos/multi_agent.md#handoffs",
    "how-tos/multi-agent-network.ipynb": "how-tos/multi_agent.md#use-in-a-multi-agent-system",
    "how-tos/multi-agent-multi-turn-convo.ipynb": "how-tos/multi_agent.md#multi-turn-conversation",
    # cloud redirects
    "cloud/index.md": "index.md",
    "cloud/how-tos/index.md": "concepts/langgraph_platform",
    "cloud/concepts/api.md": "concepts/langgraph_server.md",
    "cloud/concepts/cloud.md": "concepts/langgraph_cloud.md",
    "cloud/faq/studio.md": "concepts/langgraph_studio.md#studio-faqs",
    "cloud/how-tos/human_in_the_loop_edit_state.md": "cloud/how-tos/add-human-in-the-loop.md",
    "cloud/how-tos/human_in_the_loop_user_input.md": "cloud/how-tos/add-human-in-the-loop.md",
    "concepts/platform_architecture.md": "concepts/langgraph_cloud#architecture",
    # cloud streaming redirects
    "cloud/how-tos/stream_values.md": "https://docs.langchain.com/langgraph-platform/streaming",
    "cloud/how-tos/stream_updates.md": "https://docs.langchain.com/langgraph-platform/streaming",
    "cloud/how-tos/stream_messages.md": "https://docs.langchain.com/langgraph-platform/streaming",
    "cloud/how-tos/stream_events.md": "https://docs.langchain.com/langgraph-platform/streaming",
    "cloud/how-tos/stream_debug.md": "https://docs.langchain.com/langgraph-platform/streaming",
    "cloud/how-tos/stream_multiple.md": "https://docs.langchain.com/langgraph-platform/streaming",
    "cloud/concepts/streaming.md": "concepts/streaming.md",
    "agents/streaming.md": "how-tos/streaming.md",
    # prebuilt redirects
    "how-tos/create-react-agent.ipynb": "agents/agents.md#basic-configuration",
    "how-tos/create-react-agent-memory.ipynb": "agents/memory.md",
    "how-tos/create-react-agent-system-prompt.ipynb": "agents/context.md#prompts",
    "how-tos/create-react-agent-structured-output.ipynb": "agents/agents.md#structured-output",
    # misc
    "prebuilt.md": "agents/prebuilt.md",
    "reference/prebuilt.md": "reference/agents.md",
    "concepts/high_level.md": "index.md",
    "concepts/index.md": "index.md",
    "concepts/v0-human-in-the-loop.md": "concepts/human-in-the-loop.md",
    "how-tos/index.md": "index.md",
    "tutorials/introduction.ipynb": "concepts/why-langgraph.md",
    "agents/deployment.md": "tutorials/langgraph-platform/local-server.md",
    # deployment redirects
    "how-tos/deploy-self-hosted.md": "cloud/deployment/self_hosted_data_plane.md",
    "concepts/self_hosted.md": "concepts/langgraph_self_hosted_data_plane.md",
    "tutorials/deployment.md": "concepts/deployment_options.md",
    # assistant redirects
    "cloud/how-tos/assistant_versioning.md": "cloud/how-tos/configuration_cloud.md",
    "cloud/concepts/runs.md": "concepts/assistants.md#execution",
    # hitl redirects
    "how-tos/wait-user-input-functional.ipynb": "how-tos/use-functional-api.md",
    "how-tos/review-tool-calls-functional.ipynb": "how-tos/use-functional-api.md",
    "how-tos/create-react-agent-hitl.ipynb": "how-tos/human_in_the_loop/add-human-in-the-loop.md",
    "agents/human-in-the-loop.md": "how-tos/human_in_the_loop/add-human-in-the-loop.md",
    "how-tos/human_in_the_loop/dynamic_breakpoints.ipynb": "how-tos/human_in_the_loop/breakpoints.md",
    "concepts/breakpoints.md": "concepts/human_in_the_loop.md",
    "how-tos/human_in_the_loop/breakpoints.md": "how-tos/human_in_the_loop/add-human-in-the-loop.md",
    "cloud/how-tos/human_in_the_loop_breakpoint.md": "cloud/how-tos/add-human-in-the-loop.md",
    "how-tos/human_in_the_loop/edit-graph-state.ipynb": "how-tos/human_in_the_loop/time-travel.md",

    # LGP mintlify migration redirects
    "tutorials/auth/getting_started.md": "https://docs.langchain.com/langgraph-platform/auth",
    "tutorials/auth/resource_auth.md": "https://docs.langchain.com/langgraph-platform/resource-auth",
    "tutorials/auth/add_auth_server.md": "https://docs.langchain.com/langgraph-platform/add-auth-server",
    "how-tos/use-remote-graph.md": "https://docs.langchain.com/langgraph-platform/use-remote-graph",
    "how-tos/autogen-integration.md": "https://docs.langchain.com/langgraph-platform/autogen-integration",
    "cloud/how-tos/use_stream_react.md": "https://docs.langchain.com/langgraph-platform/use-stream-react",
    "cloud/how-tos/generative_ui_react.md": "https://docs.langchain.com/langgraph-platform/generative-ui-react",
    "concepts/langgraph_platform.md": "https://docs.langchain.com/langgraph-platform/index",
    "concepts/langgraph_components.md": "https://docs.langchain.com/langgraph-platform/components",
    "concepts/langgraph_server.md": "https://docs.langchain.com/langgraph-platform/langgraph-server",
    "concepts/langgraph_data_plane.md": "https://docs.langchain.com/langgraph-platform/data-plane",
    "concepts/langgraph_control_plane.md": "https://docs.langchain.com/langgraph-platform/control-plane",
    "concepts/langgraph_cli.md": "https://docs.langchain.com/langgraph-platform/langgraph-cli",
    "concepts/langgraph_studio.md": "https://docs.langchain.com/langgraph-platform/langgraph-studio",
    "cloud/how-tos/studio/quick_start.md": "https://docs.langchain.com/langgraph-platform/quick-start-studio",
    "cloud/how-tos/invoke_studio.md": "https://docs.langchain.com/langgraph-platform/invoke-studio",
    "cloud/how-tos/studio/manage_assistants.md": "https://docs.langchain.com/langgraph-platform/manage-assistants-studio",
    "cloud/how-tos/threads_studio.md": "https://docs.langchain.com/langgraph-platform/threads-studio",
    "cloud/how-tos/iterate_graph_studio.md": "https://docs.langchain.com/langgraph-platform/iterate-graph-studio",
    "cloud/how-tos/studio/run_evals.md": "https://docs.langchain.com/langgraph-platform/run-evals-studio",
    "cloud/how-tos/clone_traces_studio.md": "https://docs.langchain.com/langgraph-platform/clone-traces-studio",
    "cloud/how-tos/datasets_studio.md": "https://docs.langchain.com/langgraph-platform/datasets-studio",
    "concepts/sdk.md": "https://docs.langchain.com/langgraph-platform/sdk",
    "concepts/plans.md": "https://docs.langchain.com/langgraph-platform/plans",
    "concepts/application_structure.md": "https://docs.langchain.com/langgraph-platform/application-structure",
    "concepts/scalability_and_resilience.md": "https://docs.langchain.com/langgraph-platform/scalability-and-resilience",
    "concepts/auth.md": "https://docs.langchain.com/langgraph-platform/auth",
    "how-tos/auth/custom_auth.md": "https://docs.langchain.com/langgraph-platform/custom-auth",
    "how-tos/auth/openapi_security.md": "https://docs.langchain.com/langgraph-platform/openapi-security",
    "concepts/assistants.md": "https://docs.langchain.com/langgraph-platform/assistants",
    "cloud/how-tos/configuration_cloud.md": "https://docs.langchain.com/langgraph-platform/configuration-cloud",
    "cloud/how-tos/use_threads.md": "https://docs.langchain.com/langgraph-platform/use-threads",
    "cloud/how-tos/background_run.md": "https://docs.langchain.com/langgraph-platform/background-run",
    "cloud/how-tos/same-thread.md": "https://docs.langchain.com/langgraph-platform/same-thread",
    "cloud/how-tos/stateless_runs.md": "https://docs.langchain.com/langgraph-platform/stateless-runs",
    "cloud/how-tos/configurable_headers.md": "https://docs.langchain.com/langgraph-platform/configurable-headers",
    "concepts/double_texting.md": "https://docs.langchain.com/langgraph-platform/double-texting",
    "cloud/how-tos/interrupt_concurrent.md": "https://docs.langchain.com/langgraph-platform/interrupt-concurrent",
    "cloud/how-tos/rollback_concurrent.md": "https://docs.langchain.com/langgraph-platform/rollback-concurrent",
    "cloud/how-tos/reject_concurrent.md": "https://docs.langchain.com/langgraph-platform/reject-concurrent",
    "cloud/how-tos/enqueue_concurrent.md": "https://docs.langchain.com/langgraph-platform/enqueue-concurrent",
    "cloud/concepts/webhooks.md": "https://docs.langchain.com/langgraph-platform/use-webhooks",
    "cloud/how-tos/webhooks.md": "https://docs.langchain.com/langgraph-platform/use-webhooks",
    "cloud/concepts/cron_jobs.md": "https://docs.langchain.com/langgraph-platform/cron-jobs",
    "cloud/how-tos/cron_jobs.md": "https://docs.langchain.com/langgraph-platform/cron-jobs",
    "how-tos/http/custom_lifespan.md": "https://docs.langchain.com/langgraph-platform/custom-lifespan",
    "how-tos/http/custom_middleware.md": "https://docs.langchain.com/langgraph-platform/custom-middleware",
    "how-tos/http/custom_routes.md": "https://docs.langchain.com/langgraph-platform/custom-routes",
    "cloud/concepts/data_storage_and_privacy.md": "https://docs.langchain.com/langgraph-platform/data-storage-and-privacy",
    "cloud/deployment/semantic_search.md": "https://docs.langchain.com/langgraph-platform/semantic-search",
    "how-tos/ttl/configure_ttl.md": "https://docs.langchain.com/langgraph-platform/configure-ttl",
    "concepts/deployment_options.md": "https://docs.langchain.com/langgraph-platform/deployment-options",
    "cloud/quick_start.md": "https://docs.langchain.com/langgraph-platform/deployment-quickstart",
    "cloud/deployment/setup.md": "https://docs.langchain.com/langgraph-platform/setup-app-requirements-txt",
    "cloud/deployment/setup_pyproject.md": "https://docs.langchain.com/langgraph-platform/setup-pyproject",
    "cloud/deployment/setup_javascript.md": "https://docs.langchain.com/langgraph-platform/setup-javascript",
    "cloud/deployment/custom_docker.md": "https://docs.langchain.com/langgraph-platform/custom-docker",
    "cloud/deployment/graph_rebuild.md": "https://docs.langchain.com/langgraph-platform/graph-rebuild",
    "concepts/langgraph_cloud.md": "https://docs.langchain.com/langgraph-platform/cloud",
    "concepts/langgraph_self_hosted_data_plane.md": "https://docs.langchain.com/langgraph-platform/hybrid",
    "concepts/langgraph_self_hosted_control_plane.md": "https://docs.langchain.com/langgraph-platform/self-hosted",
    "concepts/langgraph_standalone_container.md": "https://docs.langchain.com/langgraph-platform/self-hosted#data-plane-only",
    "cloud/deployment/cloud.md": "https://docs.langchain.com/langgraph-platform/cloud",
    "cloud/deployment/self_hosted_data_plane.md": "https://docs.langchain.com/langgraph-platform/deploy-hybrid",
    "cloud/deployment/self_hosted_control_plane.md": "https://docs.langchain.com/langgraph-platform/deploy-self-hosted-full-platform",
    "cloud/deployment/standalone_container.md": "https://docs.langchain.com/langgraph-platform/deploy-data-plane-only",
    "concepts/server-mcp.md": "https://docs.langchain.com/langgraph-platform/server-mcp",
    "cloud/how-tos/human_in_the_loop_time_travel.md": "https://docs.langchain.com/langgraph-platform/human-in-the-loop-time-travel",
    "cloud/how-tos/add-human-in-the-loop.md": "https://docs.langchain.com/langgraph-platform/add-human-in-the-loop",
    "cloud/deployment/egress.md": "https://docs.langchain.com/langgraph-platform/env-var",
    "cloud/how-tos/streaming.md": "https://docs.langchain.com/langgraph-platform/streaming",
    "cloud/reference/api/api_ref.md": "https://docs.langchain.com/langgraph-platform/server-api-ref",
    "cloud/reference/langgraph_server_changelog.md": "https://docs.langchain.com/langgraph-platform/langgraph-server-changelog",
    "cloud/reference/api/api_ref_control_plane.md": "https://docs.langchain.com/langgraph-platform/api-ref-control-plane",
    "cloud/reference/cli.md": "https://docs.langchain.com/langgraph-platform/cli",
    "cloud/reference/env_var.md": "https://docs.langchain.com/langgraph-platform/env-var",
    "troubleshooting/studio.md": "https://docs.langchain.com/langgraph-platform/troubleshooting-studio",
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
        
        if isinstance(page_new, str) and page_new.startswith("http"):
            # Handle external redirects
            _write_html(config["site_dir"], old_html_path, page_new)
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
            
            _write_html(config["site_dir"], old_html_path, new_html_path)
