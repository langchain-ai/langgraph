import nbformat

from _scripts.notebook_convert import exporter


def _remove_consecutive_new_lines(s) -> str:
    """Remove consecutive new lines from a string."""
    return "\n".join([line for line in s.split("\n") if line.strip()])


def test_convert_notebook():
    # Test the convert_notebook function
    # Create a new, minimal notebook programmatically
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {
        "name": "python3",
        "language": "python",
        "display_name": "Python 3",
    }
    nb.metadata.language_info = {
        "name": "python",
        "mimetype": "text/x-python",
        "codemirror_mode": {
            "name": "ipython",
            "version": 3,
        },
    }

    # Add a markdown cell with a link to an .ipynb file
    md_cell_source = "This is a [link](example_notebook.ipynb) in markdown."
    nb.cells.append(nbformat.v4.new_markdown_cell(md_cell_source))

    # Add a code cell with a noqa comment
    code_cell_source = "print('hello')  # noqa: F401"
    nb.cells.append(nbformat.v4.new_code_cell(code_cell_source))
    nb.metadata.mode = "exec"

    body, _ = exporter.from_notebook_node(nb)
    assert body == """"""
