import nbformat

from _scripts.notebook_convert import md_executable


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
print("Hello, world!")\
"""

CELL_MAGIC_OUTPUT = """\
```python
%%capture
print("Hello, world!")
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
