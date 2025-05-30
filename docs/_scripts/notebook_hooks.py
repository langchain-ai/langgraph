"""mkdocs hooks for adding custom logic to documentation pipeline.

Lifecycle events: https://www.mkdocs.org/dev-guide/plugins/#events
"""

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
    "how-tos/state-reducers.ipynb": "how-tos/graph-api#define-and-update-state",
    "how-tos/sequence.ipynb": "how-tos/graph-api#create-a-sequence-of-steps",
    "how-tos/branching.ipynb": "how-tos/graph-api#create-branches",
    "how-tos/recursion-limit.ipynb": "how-tos/graph-api#create-and-control-loops",
    "how-tos/visualization.ipynb": "how-tos/graph-api#visualize-your-graph",
    "how-tos/input_output_schema.ipynb": "how-tos/graph-api#define-input-and-output-schemas",
    "how-tos/pass_private_state.ipynb": "how-tos/graph-api#pass-private-state-between-nodes",
    "how-tos/state-model.ipynb": "how-tos/graph-api#use-pydantic-models-for-graph-state",
    "how-tos/map-reduce.ipynb": "how-tos/graph-api/#map-reduce-and-the-send-api",
    "how-tos/command.ipynb": "how-tos/graph-api/#combine-control-flow-and-state-updates-with-command",
    "how-tos/configuration.ipynb": "how-tos/graph-api/#add-runtime-configuration",
    "how-tos/node-retries.ipynb": "how-tos/graph-api/#add-retry-policies",
    "how-tos/return-when-recursion-limit-hits.ipynb": "how-tos/graph-api/#impose-a-recursion-limit",
    "how-tos/async.ipynb": "how-tos/graph-api/#async",
    # memory how-tos
    "how-tos/memory/manage-conversation-history.ipynb": "how-tos/memory.ipynb",
    "how-tos/memory/delete-messages.ipynb": "how-tos/memory.ipynb#delete-messages",
    "how-tos/memory/add-summary-conversation-history.ipynb": "how-tos/memory.ipynb#summarize-messages",
    # subgraph how-tos
    "how-tos/subgraph-transform-state.ipynb": "how-tos/subgraph.ipynb#different-state-schemas",
    "how-tos/subgraphs-manage-state.ipynb": "how-tos/subgraph.ipynb#add-persistence",
    # persistence how-tos
    "how-tos/persistence_postgres.ipynb": "how-tos/persistence.ipynb#use-in-production",
    "how-tos/persistence_mongodb.ipynb": "how-tos/persistence.ipynb#use-in-production",
    "how-tos/persistence_redis.ipynb": "how-tos/persistence.ipynb#use-in-production",
    "how-tos/subgraph-persistence.ipynb": "how-tos/persistence.ipynb#use-with-subgraphs",
    "how-tos/cross-thread-persistence.ipynb": "how-tos/persistence.ipynb#add-long-term-memory",
    "cloud/how-tos/copy_threads": "cloud/how-tos/use_threads",
    # tool calling how-tos
    "how-tos/tool-calling-errors.ipynb": "how-tos/tool-calling.ipynb#handle-errors",
    "how-tos/pass-config-to-tools.ipynb": "how-tos/tool-calling.ipynb#access-config",
    "how-tos/pass-run-time-values-to-tools.ipynb": "how-tos/tool-calling.ipynb#read-state",
    "how-tos/update-state-from-tools.ipynb": "how-tos/tool-calling.ipynb#update-state",
    # multi-agent how-tos
    "how-tos/agent-handoffs.ipynb": "how-tos/multi_agent.ipynb#handoffs",
    "how-tos/multi-agent-network.ipynb": "how-tos/multi_agent.ipynb#use-in-a-multi-agent-system",
    "how-tos/multi-agent-multi-turn-convo.ipynb": "how-tos/multi_agent.ipynb#multi-turn-conversation",
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
    "cloud/how-tos/stream_values.md": "cloud/how-tos/streaming.md#stream-graph-state",
    "cloud/how-tos/stream_updates.md": "cloud/how-tos/streaming.md#stream-graph-state",
    "cloud/how-tos/stream_messages.md": "cloud/how-tos/streaming.md#messages",
    "cloud/how-tos/stream_events.md": "cloud/how-tos/streaming.md#stream-events",
    "cloud/how-tos/stream_debug.md": "cloud/how-tos/streaming.md#debug",
    "cloud/how-tos/stream_multiple.md": "cloud/how-tos/streaming.md#stream-multiple-modes",
    # prebuit redirects
    "how-tos/create-react-agent.ipynb": "agents/agents.md#basic-configuration",
    "how-tos/create-react-agent-memory.ipynb": "agents/memory.md",
    "how-tos/create-react-agent-system-prompt.ipynb": "agents/context.md#prompts",
    "how-tos/create-react-agent-hitl.ipynb": "agents/human-in-the-loop.md",
    "how-tos/create-react-agent-structured-output.ipynb": "agents/agents.md#structured-output",
    # Time-travel
    "how-tos/human_in_the_loop/edit-graph-state.ipynb": "how-tos/human_in_the_loop/time-travel.ipynb",
    # breakpoints
    "how-tos/human_in_the_loop/dynamic_breakpoints.ipynb": "how-tos/human_in_the_loop/breakpoints.ipynb",
    # misc
    "prebuilt.md": "agents/prebuilt.md",
    "reference/prebuilt.md": "reference/agents.md",
    "concepts/high_level.md": "index.md",
    "concepts/index.md": "index.md",
    "concepts/v0-human-in-the-loop.md": "concepts/human-in-the-loop.md",
    "how-tos/index.md": "index.md",
    "tutorials/introduction.ipynb": "concepts/why-langgraph.md",
    # deployment redirects
    "how-tos/deploy-self-hosted.md": "cloud/deployment/self_hosted_data_plane.md",
    "concepts/self_hosted.md": "concepts/langgraph_self_hosted_data_plane.md",
    # assistant redirects
    "cloud/how-tos/assistant_versioning.md": "cloud/how-tos/configuration_cloud.md",
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
            opening_fence += f" hl_lines=\"{' '.join(highlighted_lines)}\""

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

    # Append API reference links to code blocks
    if add_api_references:
        markdown = update_markdown_with_imports(markdown, page.file.abs_src_path)
    # Apply highlight comments to code blocks
    markdown = _highlight_code_blocks(markdown)

    # Add file path as an attribute to code blocks that are executable.
    # This file path is used to associate fixtures with the executable code
    # which can be used in CI to test the docs without making network requests.
    markdown = _add_path_to_code_blocks(markdown, page)

    if remove_base64_images:
        # Remove base64 encoded images from markdown
        markdown = re.sub(r"!\[.*?\]\(data:image/[^;]+;base64,[^)]+\)", "", markdown)

    return markdown


def on_page_markdown(markdown: str, page: Page, **kwargs: Dict[str, Any]):
    return _on_page_markdown_with_config(
        markdown,
        page,
        add_api_references=True,
        **kwargs,
    )


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


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    """Inject Google Tag Manager noscript tag immediately after <body>.

    Args:
        output: The HTML output of the page.
        page: The page instance.
        config: The MkDocs configuration object.

    Returns:
        modified HTML output with GTM code injected.
    """
    return _inject_gtm(output)


# Create HTML files for redirects after site dir has been built
def on_post_build(config):
    use_directory_urls = config.get("use_directory_urls")
    for page_old, page_new in REDIRECT_MAP.items():
        page_old = page_old.replace(".ipynb", ".md")
        page_new = page_new.replace(".ipynb", ".md")
        page_new_before_hash, hash, suffix = page_new.partition("#")
        old_html_path = File(page_old, "", "", use_directory_urls).dest_path.replace(
            os.sep, "/"
        )
        new_html_path = File(page_new_before_hash, "", "", True).url
        new_html_path = (
            posixpath.relpath(new_html_path, start=posixpath.dirname(old_html_path))
            + hash
            + suffix
        )
        _write_html(config["site_dir"], old_html_path, new_html_path)
