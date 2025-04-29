import os
import tempfile

import nbformat
import pytest

from _scripts.notebook_convert import (
    _convert_links_in_markdown,
    _has_output,
    convert_notebook,
)


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


EXPECTED_OUTPUT = """\
```shell
pip install -U langgraph
```


```python
print('Hello')
```\

"""


def test_converting_cell_magic() -> None:
    """Test converting cell magic to code blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nb_path = os.path.join(tmpdir, "test_notebook.ipynb")

        # Create a minimal notebook object
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(
                "%%capture --no-stderr\n"
                "%pip install -U langgraph"
            ),
            nbformat.v4.new_code_cell("print('Hello')"),
        ]
        nb.metadata["language_info"] = {"name": "python"}

        # Write to file
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        # Run the conversion
        converted = convert_notebook(nb_path)
        assert converted == EXPECTED_OUTPUT
