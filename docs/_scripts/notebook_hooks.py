import logging
from typing import Any, Dict

from mkdocs.structure.pages import Page
from mkdocs.structure.files import Files, File
from notebook_convert import convert_notebook

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class NotebookFile(File):
    def is_documentation_page(self):
        return True


def on_files(files: Files, **kwargs: Dict[str, Any]):
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


def on_page_markdown(markdown: str, page: Page, **kwargs: Dict[str, Any]):
    if page.file.src_path.endswith(".ipynb"):
        logger.info("Processing Jupyter notebook: %s", page.file.src_path)
        body = convert_notebook(page.file.abs_src_path)
        return body

    return markdown
