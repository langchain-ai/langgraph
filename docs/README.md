# LangGraph Documentation

For more information on contributing to our documentation, see the [Contributing Guide](../CONTRIBUTING.md).

## Structure

The primary documentation is located in the `docs/` directory. This directory contains both the source files for the main documentation as well as the API reference doc build process.

### Main Documentation

Main documentation files are located in `docs/docs/` and are written in Markdown format. The site uses [**MkDocs**](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) and includes:

- **Concepts**: Core LangGraph concepts and explanations
- **Tutorials**: Step-by-step learning guides
- **How-tos**: Task-focused guides for specific use cases
- **Examples**: Real-world applications and use cases
- **Jupyter Notebooks**: Interactive tutorials that are automatically converted to markdown

### API Reference

API reference documentation is defined in `docs/docs/reference/`. Each `.md` file outlines the "template" that each page is built from. Reference content is automatically generated from docstrings in the codebase using the **mkdocstrings** plugin. Once generated, the content is plugged into the corresponding markdown file where it is referenced by using manual directives to specify which classes and/or functions are documented:

```markdown
::: langgraph.graph.state.StateGraph
    options:
      show_if_no_docstring: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - add_node
        - add_edge
        - add_conditional_edges
        - add_sequence
        - compile
```

## Build Process

Docs are built following these steps:

1. **Content Processing:**
   - `_scripts/notebook_hooks.py` - Main processing pipeline that:
     - Converts how-tos/tutorial Jupyter notebooks to markdown using `notebook_convert.py`
     - Adds automatic API reference links to code blocks using `generate_api_reference_links.py`
     - Handles conditional rendering for Python/JS versions
     - Processes highlight comments and custom syntax

2. **API Reference Generation:**
   - **mkdocstrings** plugin extracts docstrings from Python source code
   - Manual `::: module.Class` directives in reference pages (`/docs/docs/*`) specify what to document
   - Cross-references are automatically generated between docs and API

3. **Site Generation:**
   - **MkDocs** processes all markdown files and generates static HTML
   - Custom hooks handle redirects and inject additional functionality

4. **Deployment:**
   - Site is deployed with Vercel
   - `make build-docs` generates production build (also usable for local testing)
   - Automatic redirects handle URL changes between versions

### Local Development

For local development, use the Makefile targets:

```bash
# Serve docs locally with hot reloading
make serve-docs

# Clean build for production testing
make build-docs

# Serve with clean build
make serve-clean-docs
```

The `serve-docs` command:

- Watches source files for changes
- Includes dirty builds for faster iteration
- Serves on [http://127.0.0.1:8000/langgraph/](http://127.0.0.1:8000/langgraph/)

## Standards

**Docstring Format:**
The API reference uses **Google-style docstrings** with Markdown markup. The `mkdocstrings` plugin processes these to generate documentation.

**Required format:**

```python
def example_function(param1: str, param2: int = 5) -> bool:
    """Brief description of the function.

    Longer description can go here. Use Markdown syntax for
    rich formatting like **bold** and *italic*.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not an integer.

    !!! warning
        This function is experimental and may change.

    !!! version-added "Added in version 0.2.0"
    """
```

**Special Markers:**

- **MkDocs admonitions**: `!!! warning`, `!!! note`, `!!! version-added`
- **Code blocks**: Standard markdown ``` syntax
- **Cross-references**: Automatic linking via `generate_api_reference_links.py`

## Execute notebooks

If you would like to automatically execute all of the notebooks, to mimic the "Run notebooks" GitHub action, you can run:

```bash
python _scripts/prepare_notebooks_for_ci.py
./_scripts/execute_notebooks.sh
```

**Note**: if you want to run the notebooks without `%pip install` cells, you can run:

```bash
python _scripts/prepare_notebooks_for_ci.py --comment-install-cells
./_scripts/execute_notebooks.sh
```

`prepare_notebooks_for_ci.py` script will add VCR cassette context manager for each cell in the notebook, so that:

- when the notebook is run for the first time, cells with network requests will be recorded to a VCR cassette file
- when the notebook is run subsequently, the cells with network requests will be replayed from the cassettes

## Adding new notebooks

If you are adding a notebook with API requests, it's **recommended** to record network requests so that they can be subsequently replayed. If this is not done, the notebook runner will make API requests every time the notebook is run, which can be costly and slow.

To record network requests, please make sure to first run `prepare_notebooks_for_ci.py` script.

Then, run

```bash
jupyter execute <path_to_notebook>
```

Once the notebook is executed, you should see the new VCR cassettes recorded in `cassettes` directory and discard the updated notebook.

## Updating existing notebooks

If you are updating an existing notebook, please make sure to remove any existing cassettes for the notebook in `cassettes` directory (each cassette is prefixed with the notebook name), and then run the steps from the "Adding new notebooks" section above.

To delete cassettes for a notebook, you can run:

```bash
rm cassettes/<notebook_name>*
```
