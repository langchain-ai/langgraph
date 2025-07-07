"""Experimental script to generate consolidated llms text from the docs."""

import asyncio
import glob
import os
import re
from typing import TypedDict, List, Optional

import yaml
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from mkdocs.structure.files import File
from mkdocs.structure.pages import Page
from pydantic import BaseModel, Field
from yaml import SafeLoader

from _scripts.notebook_hooks import _on_page_markdown_with_config

HERE = os.path.dirname(os.path.abspath(__file__))
# Get source directory (parent of HERE / docs)
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(HERE), "docs"))


async def convert_ipynb_to_md(file_path: str) -> Optional[str]:
    """Process a file (markdown or notebook) to markdown format.

    Args:
        file_path: Path to the file to process

    Returns:
        Processed markdown content if successful, None otherwise
    """
    rel_path = os.path.relpath(file_path, SOURCE_DIR)

    # Create File and Page objects to match mkdocs structure
    file_obj = File(
        path=rel_path, src_dir=SOURCE_DIR, dest_dir="", use_directory_urls=True
    )
    page = Page(
        title="",
        file=file_obj,
        config={},
    )

    try:
        # Read raw content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Convert to markdown without logic to resolve API references
        processed_content = _on_page_markdown_with_config(
            content, page, add_api_references=False, remove_base64_images=True
        )
        # Remove self-closing img tags <img ... />
        processed_content = re.sub(r"<img[^>]*/>", "", processed_content)
        # Remove img tags with content <img ...>...</img>
        processed_content = re.sub(
            r"<img[^>]*>.*?</img>", "", processed_content, flags=re.DOTALL
        )
        return processed_content
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


async def generate_full_llms_text(output_file: str) -> None:
    """Generate a consolidated text file from markdown/notebook files for LLM training.

    Args:
        output_file: Path to output the consolidated text file
    """
    # Collect all markdown and notebook files
    all_files = glob.glob(os.path.join(SOURCE_DIR, "how-tos/*.md"), recursive=True)
    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "how-tos/*.ipynb"), recursive=True)
    )
    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "concepts/*.md"), recursive=True)
    )
    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "concepts/*.ipynb"), recursive=True)
    )

    all_content = []

    # Process files concurrently
    tasks = [convert_ipynb_to_md(file_path) for file_path in all_files]
    results = await asyncio.gather(*tasks)

    # Combine results with file paths
    for file_path, processed_content in zip(all_files, results):
        if processed_content:
            rel_path = os.path.relpath(file_path, SOURCE_DIR)
            # Add file name
            all_content.append(f"---\n{rel_path}\n---")
            # Add content
            all_content.append(processed_content)

    # Write consolidated output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_content))


def no_op_constructor(*args):
    """No-op"""


SafeLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name",
    no_op_constructor,
)


class NavItem(TypedDict):
    title: str
    url: str
    hierarchy: tuple[str, ...]
    description: str


def _flatten_nav(
    nav: list[dict[str, str | list] | str], path: tuple[str, ...] = ()
) -> list[NavItem]:
    flat: List[NavItem] = []
    for item in nav:
        if isinstance(item, dict):
            for title, node in item.items():
                new_path = path + (title,)
                if isinstance(node, str):
                    # Leaf page
                    flat.append(
                        {
                            "title": title,
                            "url": node,
                            "hierarchy": new_path,
                            "description": "",
                        }
                    )
                elif isinstance(node, list):
                    # Dive in, carrying along the updated path
                    flat.extend(_flatten_nav(node, new_path))
                else:
                    raise TypeError(
                        f"Unexpected node type {type(node)} under {title!r}"
                    )
        elif isinstance(item, str):
            # Bare string entry → use itself as title, and as URL
            new_path = path + (item,)
            flat.append(
                {"title": item, "url": item, "hierarchy": new_path, "description": ""}
            )
        else:
            raise TypeError(f"Unexpected item type {type(item)} in nav")
    return flat


class PageInfo(BaseModel):
    title: str = Field(description="The title of the page")
    description: str = Field(
        description="A short description of the page no longer than 3 sentences "
        "explaining the kind of content that can be found in the page."
    )


async def process_nav_items(nav_items: list[NavItem]) -> list[NavItem]:
    """Open the contents of each nav item and come up with a better title and description."""
    rate_limiter = InMemoryRateLimiter(requests_per_second=10)
    model = init_chat_model("gpt-4o-mini", temperature=0.0, rate_limiter=rate_limiter)
    model = model.with_structured_output(PageInfo)

    async def process_single_item(item: NavItem) -> NavItem:
        path = item["url"]
        file_path = os.path.join(SOURCE_DIR, path)

        # Process the file content (handles both markdown and notebooks)
        if path.endswith(".ipynb"):
            content = await convert_ipynb_to_md(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        if not content:
            return item

        # Generate a better title and description
        response = await model.ainvoke(
            [
                {
                    "role": "system",
                    "content": "You are a technical documentation writer. "
                    "You are given a markdown page of documentation. "
                    "Please come up with an appropriate title and "
                    "description for the page. The description should "
                    "be a short summary of the page content that is "
                    "no longer than 3 sentences.",
                },
                {
                    "role": "user",
                    "content": "The markdown page is as follows:\n\n" + content,
                },
            ]
        )
        return {
            "title": response.title,
            "url": item["url"],
            "hierarchy": item["hierarchy"],
            "description": response.description,
        }

    # Remove any items that start with http:// or https:// looking only for
    # local file at this stages.
    nav_items = [
        item
        for item in nav_items
        if not item["url"].startswith(("http://", "https://"))
    ]
    # Process items in parallel
    tasks = [process_single_item(item) for item in nav_items]
    new_nav_items = await asyncio.gather(*tasks)
    return new_nav_items


async def generate_nav_links_text(
    output_file: str, *, replace_links: bool = False
) -> None:
    """Generate llms.txt from mkdocs.yaml."""
    # Get path to mkdocs.yaml relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mkdocs_path = os.path.join(os.path.dirname(script_dir), "mkdocs.yml")

    # Load and parse yaml
    with open(mkdocs_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract nav section
    nav = config.get("nav", [])
    flattened = _flatten_nav(nav)

    processed_nav = await process_nav_items(flattened)

    with open(output_file, "w") as f:
        current_section = None
        for item in processed_nav:
            # Get the top-level section (first item in hierarchy)
            section = item["hierarchy"][0]

            if section not in {"Guides", "Examples", "Resources"}:
                continue

            # If we're starting a new section, add a heading
            if section != current_section:
                f.write(f"\n# {section}\n\n")
                current_section = section

            title = item["title"]
            # Process URL based on replace_links flag
            url = item["url"]
            if replace_links:
                # Remove .md extension and ensure single trailing slash
                url = url.removesuffix(".md")
                url = url.removesuffix(".ipynb")
                url = url.rstrip("/") + "/"
                url = f"https://langchain-ai.github.io/langgraph/{url}"

            f.write(f"- [{title}]({url}): {item['description']}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate consolidated text file from markdown/notebook files for LLMs."
        )
    )
    parser.add_argument("output_file", help="Path to output the consolidated text file")
    parser.add_argument(
        "--link-only",
        action="store_true",
        help="Only include link references in the output",
    )
    parser.add_argument(
        "--replace-links",
        action="store_true",
        help="Replace markdown links with full URLs in the output",
    )

    args = parser.parse_args()
    if args.link_only:
        coro = generate_nav_links_text(
            args.output_file, replace_links=args.replace_links
        )
    else:
        coro = generate_full_llms_text(args.output_file)

    asyncio.run(coro)
