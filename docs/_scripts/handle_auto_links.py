"""Logic to identify and transform cross-reference links in markdown files.

This module allows supporting custom markdown syntax for "autolinks". These are links
that will be transformed based on the current scope context, such as "global", "python",
or "js" into an appropriate markdown link format.

For example,

```markdown
@[StateGraph]
```

May be transformed into:

```markdown
[StateGraph](some_path/api-reference/state-graph.md)
```

The transformation value depends on the scope in which the link is used.
"""

import logging
import re
from typing import Optional

from _scripts.link_map import SCOPE_LINK_MAPS

logger = logging.getLogger(__name__)


def _transform_link(
    link_name: str,
    scope: str,
    file_path: str,
    line_number: int,
    custom_title: Optional[str] = None,
) -> Optional[str]:
    """Transform a cross-reference link based on the current scope.

    Args:
        link_name: The name of the link to transform (e.g., "StateGraph").
        scope: The current scope context ("global", "python", "js", etc.).
        file_path: The file path for error reporting.
        line_number: The line number for error reporting.
        custom_title: Optional custom title for the link. If `None`, uses link_name.

    Returns:
        A formatted markdown link if the link is found in the scope mapping,
        None otherwise.

    Example:
        >>> _transform_link("StateGraph", "python", "file.md", 5)
        "[StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph)"

        >>> _transform_link("StateGraph", "python", "file.md", 5, "Custom Title")
        "[Custom Title](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph)"

        >>> _transform_link("unknown-link", "python", "file.md", 5)
        None
    """
    if scope == "global":
        # Special scope that is composed of both Python and JS links
        # For now, we will substitute in the python scope!
        # But we need to add support for handling both scopes.
        scope = "python"
        logger.error(
            "Encountered unhandled 'global' scope. Defaulting to 'python'."
            "In file: %s, line %d, link_name: %s",
            file_path,
            line_number,
            link_name,
        )
    link_map = SCOPE_LINK_MAPS.get(scope, {})
    url = link_map.get(link_name)

    if url:
        title = custom_title if custom_title is not None else link_name
        return f"[{title}]({url})"
    else:
        # Log error with file location information
        logger.info(
            # Using %s
            "Link '%s' not found in scope '%s'. "
            "In file: %s, line %d. Available links in scope: %s",
            link_name,
            scope,
            file_path,
            line_number,
            list(link_map.keys() if link_map else []),
        )
        return None


CONDITIONAL_FENCE_PATTERN = re.compile(
    r"""
    ^                       # Start of line
    (?P<indent>[ \t]*)      # Optional indentation (spaces or tabs)
    :::                     # Literal fence marker
    (?P<language>\w+)?      # Optional language identifier (named group: language)
    \s*                     # Optional trailing whitespace
    $                       # End of line
    """,
    re.VERBOSE,
)
CROSS_REFERENCE_PATTERN = re.compile(
    r"""
    (?:                     # Non-capturing group for two possible formats:
        @\[                 # @ symbol followed by opening bracket for title
        (?P<title>[^\]]+)   # Custom title - one or more non-bracket characters
        \]                  # Closing bracket for title
        \[                  # Opening bracket for link name
        (?P<link_name_with_title>[^\]]+)  # Link name - one or more non-bracket characters
        \]                  # Closing bracket for link name
        |                   # OR
        @\[                 # @ symbol followed by opening bracket
        (?P<link_name>[^\]]+)   # Link name - one or more non-bracket characters
        \]                  # Closing bracket
    )
    """,
    re.VERBOSE,
)


def _replace_autolinks(
    markdown: str, file_path: str, *, default_scope: str = "python"
) -> str:
    """Preprocess markdown lines to handle @[links] with conditional fence scopes.

    This function processes markdown content to transform @[link_name] references
    based on the current conditional fence scope. Conditional fences use the
    syntax :::language to define scope boundaries.

    Args:
        markdown: The markdown content to process.
        file_path: The file path for error reporting.
        default_scope: The default scope to use if no scope is matched.

    Returns:
        Processed markdown content with @[references] transformed to proper
        markdown links or left unchanged if not found.

    Example:
        Input:
            "@[StateGraph]\\n:::python\\n@[Command]\\n:::\\n"
        Output:
            "[StateGraph](url)\\n:::python\\n[Command](url)\\n:::\\n"
    """
    # Track the current scope context
    current_scope = default_scope
    lines = markdown.splitlines(keepends=True)
    processed_lines = []

    for line_number, line in enumerate(lines, 1):
        line_stripped = line.strip()

        # Check if this line defines a new conditional fence scope
        fence_match = CONDITIONAL_FENCE_PATTERN.match(line_stripped)
        if fence_match:
            language = fence_match.group("language")
            # Set scope to the specified language, or reset to global if no language
            current_scope = language.lower() if language else default_scope
            processed_lines.append(line)
            continue

        # Transform all @[link_name] references in this line based on current scope
        def replace_cross_reference(match: re.Match[str]) -> str:
            """Replace a single @[link_name] with the scoped equivalent."""
            # Check if this is the @[title][ref] format or @[ref] format
            title = match.group("title")
            if title is not None:
                # This is @[title][ref] format
                link_name = match.group("link_name_with_title")
                custom_title = title
            else:
                # This is @[ref] format
                link_name = match.group("link_name")
                custom_title = None

            transformed = _transform_link(
                link_name, current_scope, file_path, line_number, custom_title
            )
            return transformed if transformed is not None else match.group(0)

        transformed_line = CROSS_REFERENCE_PATTERN.sub(replace_cross_reference, line)
        processed_lines.append(transformed_line)

    return "".join(processed_lines)
