"""Test generation of links into the API reference."""

import pytest

from _scripts.generate_api_reference_links import (
    update_markdown_with_imports,
    get_imports,
)

MARKDOWN_IMPORTS = """\
```python
from langgraph.types import interrupt
```
"""

EXPECTED_MARKDOWN = """\
<sup><i>API Reference: <a href="https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt">interrupt</a></i></sup>

```python
from langgraph.types import interrupt
```
"""


def test_update_markdown_with_imports() -> None:
    """Light weight end-to-end test."""
    assert (
        update_markdown_with_imports(MARKDOWN_IMPORTS, "some_path") == EXPECTED_MARKDOWN
    )


@pytest.mark.parametrize(
    "code_block, expected_imports",
    [
        (
            "from langgraph.types import interrupt",
            [
                {
                    "docs": "https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt",
                    "imported": "interrupt",
                    "path": "some_path",
                    "source": "langgraph.types",
                }
            ],
        ),
        (
            "from langgraph.types import ( interrupt )",
            [
                {
                    "docs": "https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt",
                    "imported": "interrupt",
                    "path": "some_path",
                    "source": "langgraph.types",
                }
            ],
        ),
        (
            "from langgraph.types import interrupt as foo",
            [
                {
                    "docs": "https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt",
                    "imported": "interrupt",
                    "path": "some_path",
                    "source": "langgraph.types",
                }
            ],
        ),
    ],
)
def test_get_imports(code_block: str, expected_imports: list) -> None:
    """Get imports from a code block."""
    assert (
        get_imports(code_block, "some_path") == expected_imports
    ), f"Failed for code_block=`{code_block}`"


@pytest.mark.parametrize(
    "code, expected_imports",
    [
        # Single import without parenthesis
        (
            "from langgraph.types import interrupt",
            [
                {
                    "source": "langgraph.types",
                    "imported": "interrupt",
                }
            ],
        ),
        # Multiple imports
        (
            (
                "from langgraph.types import interrupt\n"
                "from langgraph.func import task"
            ),
            [
                {
                    "source": "langgraph.types",
                    "imported": "interrupt",
                },
                {
                    "source": "langgraph.func",
                    "imported": "task",
                },
            ],
        ),
        # Single import with parenthesis and extra whitespace
        (
            "from langgraph.types import ( interrupt )",
            [
                {
                    "source": "langgraph.types",
                    "imported": "interrupt",
                }
            ],
        ),
        # Single import with an alias
        (
            "from langgraph.types import interrupt as foo",
            [
                {
                    "source": "langgraph.types",
                    "imported": "interrupt",
                }
            ],
        ),
        # Multiple imports on one line with an alias
        (
            "from langgraph.types import interrupt, StreamWriter as bar",
            [
                {
                    "source": "langgraph.types",
                    "imported": "interrupt",
                },
                {
                    "source": "langgraph.types",
                    "imported": "StreamWriter",
                },
            ],
        ),
        # Multiple imports without aliases
        (
            "from langgraph.types import interrupt, StreamWriter",
            [
                {
                    "source": "langgraph.types",
                    "imported": "interrupt",
                },
                {
                    "source": "langgraph.types",
                    "imported": "StreamWriter",
                },
            ],
        ),
        # Multiline import with parenthesis and trailing comma
        (
            """from langgraph.types import (
            interrupt,
            StreamWriter as foo,
            Command,
        )""",
            [
                {
                    "source": "langgraph.types",
                    "imported": "interrupt",
                },
                {
                    "source": "langgraph.types",
                    "imported": "StreamWriter",
                },
                {
                    "source": "langgraph.types",
                    "imported": "Command",
                },
            ],
        ),
        # Multiline import with parenthesis and trailing comma
        (
            (
                "from langgraph.types import (\n"
                "        interrupt,\n"
                "        StreamWriter as foo\n,"
                "        Command,\n"
                ")\n"
                "def foo():\n"
                "    pass\n"
                ""
            ),
            [
                {
                    "source": "langgraph.types",
                    "imported": "interrupt",
                },
                {
                    "source": "langgraph.types",
                    "imported": "StreamWriter",
                },
                {
                    "source": "langgraph.types",
                    "imported": "Command",
                },
            ],
        ),
    ],
)
def test_regexp_matching(code: str, expected_imports: list) -> None:
    results = get_imports(code, "some_path")
    for result in results:
        del result["docs"]
        del result["path"]

    assert results == expected_imports
