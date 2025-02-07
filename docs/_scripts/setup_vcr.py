# A list of patterns that, if found in a code block, will cause us to leave that block unchanged.
import os


BLOCKLIST_COMMANDS = (
    "WebBaseLoader",  # avoid caching web pages
    "draw_mermaid_png",  # avoid generating mermaid images via API
)

_assets_dir = os.path.join(os.path.dirname(__file__), "assets")


def is_magic_command(line: str) -> bool:
    """Return True if the line starts with a magic or shell command."""
    stripped = line.strip()
    return stripped.startswith("%") or stripped.startswith("!")


def is_comment(line: str) -> bool:
    """Return True if the line is a comment."""
    return line.strip().startswith("#")


def has_blocklisted_command_in_code(code: str) -> bool:
    """Return True if any blocklisted pattern is found in the code."""
    for pattern in BLOCKLIST_COMMANDS:
        if pattern in code:
            return True
    return False


def add_vcr_setup_to_markdown(content: str, session_id: str) -> str:
    """Prepend a Python code block to the Markdown content that sets up VCR.

    The code block defines VCR configuration and helper functions. You can
    call this function once on your Markdown file so that later Python code blocks
    (which are wrapped with a cassette context manager) can use the VCR setup.
    """
    with open(os.path.join(_assets_dir, "vcr_setup_preamble.py"), "r") as f:
        vcr_setup_code = f.read()
    # Wrap the setup code in a Python code block (using triple backticks)
    vcr_setup_block = (
        f'```python exec="on" session="{session_id}" \n{vcr_setup_code}\n```\n\n'
    )
    return vcr_setup_block + content


def wrap_python_code_block_with_vcr(
    code_block: str, cassette_prefix: str, block_id: int
) -> str:
    """Wrap a Python code block (as a string) with a VCR cassette context manager.

    The function checks for trivial or problematic cases (for example, if the block is empty,
    contains only magic commands, or has blocklisted commands) and, if appropriate, wraps the
    code with a `with custom_vcr.use_cassette(...):` statement.

    Args:
        code_block: The original Python code block (without the markdown fences).
        cassette_prefix: A prefix that will be used to generate a unique cassette filename.
        block_id: A unique identifier (e.g. an integer) for this code block.

    Returns:
        The transformed code block as a string.
    """
    lines = code_block.splitlines()

    # If the block is empty (or only whitespace), return it unchanged.
    if not any(line.strip() for line in lines):
        return code_block

    # If the code contains any blocklisted command, skip wrapping.
    if has_blocklisted_command_in_code(code_block):
        return code_block

    # Check for magic commands.
    magic_flags = [is_magic_command(line) for line in lines if line.strip()]
    if all(magic_flags):
        # All lines are magic commands; do not wrap.
        return code_block

    if any(magic_flags) and not all(magic_flags):
        # Mixed magic and non-magic code is not supported.
        raise ValueError(
            "Cannot process code blocks with mixed magic and non-magic code."
        )

    # Optionally, if the block only contains comments, you might also choose to leave it alone.
    if all(is_comment(line) or not line.strip() for line in lines):
        return code_block

    # Build a unique cassette name.
    cassette_name = f"{cassette_prefix}_{block_id}.msgpack.zlib"

    # Prepend the VCR context manager and indent the original code by 4 spaces.
    # wrapped_lines = [
    #     f"with custom_vcr.use_cassette('{cassette_name}', filter_headers=['x-api-key', 'authorization'], record_mode='once', serializer='advanced_compressed'):"
    # ]
    # for line in lines:
    #     wrapped_lines.append("    " + line)

    # Add context manager at start with explicit __enter__ and __exit__ calls
    wrapped_lines = [
        f"c = custom_vcr.use_cassette('{cassette_name}', filter_headers=['x-api-key, authorization'], record_mode='once', serializer='advanced_compressed') # markdown-exec: hide",
        "c.__enter__() # markdown-exec: hide",
    ]
    wrapped_lines.extend(lines)
    wrapped_lines.append("c.__exit__() # markdown-exec: hide")
    return "\n".join(wrapped_lines)


def add_nock_setup_to_markdown(content: str, session_id: str) -> str:
    """Prepend a Typescript code block to the Markdown content that sets up Nock.

    The code block defines Nock configuration and helper functions. You can
    call this function once on your Markdown file so that later Typescript code blocks
    will replay their HTTP requests instead of making real ones.
    """
