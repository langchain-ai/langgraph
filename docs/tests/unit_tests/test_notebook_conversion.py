import nbformat
import pytest

from _scripts.notebook_convert import (
    _convert_links_in_markdown,
    md_executable,
    _has_output,
)

EXPECTED_OUTPUT = """\
```python exec="on" source="above" session="1" result="ansi"
print("Hello, world!")
```
"""


def test_convert_normal_code_block() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.cells.append(nbformat.v4.new_code_cell('print("Hello, world!")'))
    markdown, _ = md_executable.from_notebook_node(notebook)
    assert markdown == EXPECTED_OUTPUT


# We treat cell magic as a non-executable code block.
CELL_MAGIC_INPUT = """\
%%capture
%pip install numpy
"""

CELL_MAGIC_OUTPUT = """\
```shell
pip install numpy
```
"""


def test_convert_cell_magic() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.cells.append(nbformat.v4.new_code_cell(CELL_MAGIC_INPUT))
    markdown, _ = md_executable.from_notebook_node(notebook)
    assert markdown == CELL_MAGIC_OUTPUT


STDIN_INPUT = """\
input("Enter your name: ")\
"""

STDIN_OUTPUT = """\
```python
input("Enter your name: ")
```
"""


def test_convert_input_cell() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.cells.append(nbformat.v4.new_code_cell(STDIN_INPUT))
    markdown, _ = md_executable.from_notebook_node(notebook)
    assert markdown == STDIN_OUTPUT


NO_STDOUT_EXPECTED = """\
```python exec="on" source="above" session="1"
display(x)
```
"""


def test_convert_block_without_output() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.cells.append(nbformat.v4.new_code_cell("display(x)"))
    markdown, _ = md_executable.from_notebook_node(notebook)
    assert markdown == NO_STDOUT_EXPECTED


def test_has_output() -> None:
    """Test if a given code block is expected to have output."""
    assert _has_output("print('Hello, world!')") is True
    assert _has_output("print_stream(some_iterable)") is True
    assert _has_output("foo.y") is True
    assert _has_output("display(x)") is False
    assert _has_output("assert 1 == 1") is False
    assert _has_output("def foo(): pass") is False
    assert _has_output("import foobar") is False


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "This is a [link](https://example.com).",
            "This is a [link](https://example.com).",
        ),
        ("This is a [link](../foo).", "This is a [link](foo.md)."),
        ("This is a [link](../foo#hello).", "This is a [link](foo.md#hello)."),
        ("This is a [link](../foo/#hello).", "This is a [link](foo.md#hello)."),
    ],
)
def test_link_conversion(source: str, expected: str) -> None:
    """Test logic to convert links in markdown cells."""
    assert _convert_links_in_markdown(source) == expected
