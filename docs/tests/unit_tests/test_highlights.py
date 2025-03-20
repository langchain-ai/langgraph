from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File
from mkdocs.structure.pages import Page

from _scripts.notebook_hooks import _highlight_code_blocks, on_page_markdown

NO_OP_INPUT_1 = """\
This is a plain text without any code blocks.

```python
print("Hello, World!")
```
"""

NO_OP_INPUT_2 = """\

=== "Python"

    ```python
        def foo():
            pass
    print("Hello, World!")
    ```
"""


def test_highlight_code_blocks_no_op() -> None:
    assert _highlight_code_blocks(NO_OP_INPUT_1) == NO_OP_INPUT_1
    assert _highlight_code_blocks(NO_OP_INPUT_2) == NO_OP_INPUT_2


# Examples are written in multiline style to make sure that whitespace
# is easy to interpret.
INPUT_HIGHLIGHT_1 = """\
This is a plain text without any code blocks.

```python
# highlight-next-line
print("Hello, World!")
```
"""

EXPECTED_HIGHLIGHT_1 = """\
This is a plain text without any code blocks.

```python hl_lines="1"
print("Hello, World!")
```
"""

INPUT_HIGHLIGHT_2 = """\
This is a plain text without any code blocks.

```python
# highlight-next-line
print("Hello, World!")

x = 5

# highlight-next-line
print("Hello, World!")

```
"""

EXPECTED_HIGHLIGHT_2 = """\
This is a plain text without any code blocks.

```python hl_lines="1 5"
print("Hello, World!")

x = 5

print("Hello, World!")

```
"""


# Test end-to-end behavior of on_page_markdown
INPUT_HIGHLIGHT_3 = """\
```python exec="on" source="below"
print("Hello, World!")
# highlight-next-line
print("Hello, World!")
```
"""

EXPECTED_HIGHLIGHT_3 = """\
```python exec="on" source="below" hl_lines="2"
print("Hello, World!")
print("Hello, World!")
```
"""


def test_highlight_code_blocks() -> None:
    """Test that code blocks are highlighted correctly."""
    assert _highlight_code_blocks(INPUT_HIGHLIGHT_1) == EXPECTED_HIGHLIGHT_1
    assert _highlight_code_blocks(INPUT_HIGHLIGHT_2) == EXPECTED_HIGHLIGHT_2
    assert _highlight_code_blocks(INPUT_HIGHLIGHT_3) == EXPECTED_HIGHLIGHT_3


END_TO_END_INPUT_HIGHLIGHT_1 = """\
```python exec="on" source="below"
print("Hello, World!")
# highlight-next-line
print("Hello, World!")
```
"""


END_TO_END_INPUT_HIGHLIGHT_1_EXPECT = """\
```python exec="on" source="below" hl_lines="2" path="dummy.md"
print("Hello, World!")
print("Hello, World!")
```
"""


def test_on_page_markdown_highlights() -> None:
    """Test that on page markdown behaves correctly."""
    # Create a dummy MkDocs File and Page object.
    dummy_file = File("dummy.md", "dummy.md", "placeholder", use_directory_urls=False)
    dummy_page = Page("Test Page", dummy_file, config=MkDocsConfig())

    assert (
        on_page_markdown(END_TO_END_INPUT_HIGHLIGHT_1, dummy_page)
        == END_TO_END_INPUT_HIGHLIGHT_1_EXPECT
    )
