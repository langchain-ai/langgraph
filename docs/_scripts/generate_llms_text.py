"""Experimental script to generate consolidated llms text from the docs."""

import glob
import os
from typing import TypedDict, List

import yaml
from mkdocs.structure.files import File
from mkdocs.structure.pages import Page
from yaml import SafeLoader

from _scripts.notebook_hooks import _on_page_markdown_with_config

HERE = os.path.dirname(os.path.abspath(__file__))
# Get source directory (parent of HERE / docs)
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(HERE), "docs"))


def generate_full_llms_text(output_file: str) -> str:
    """Generate a consolidated text file from markdown/notebook files for LLM training.

    Args:
        output_file: Path to output the consolidated text file
    """
    # Collect all markdown and notebook files
    all_files = glob.glob(os.path.join(SOURCE_DIR, "how-tos/*.md"), recursive=True)

    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "how-tos/*.ipynb"), recursive=True)
    )
    # Add all concepts
    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "concepts/*.md"), recursive=True)
    )
    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "concepts/*.ipynb"), recursive=True)
    )

    all_content = []

    # Process each file
    for file_path in all_files:
        print(f"Processing {file_path}")
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

        # Read raw content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Convert to markdown without logic to resolve API references
        processed_content = _on_page_markdown_with_config(
            content, page, add_api_references=False, remove_base64_images=True
        )
        if processed_content:
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

def _flatten_nav(nav: list[dict[str, str | list] | str], path: tuple[str, ...] = ()) -> list[NavItem]:
    flat: List[NavItem] = []
    for item in nav:
        if isinstance(item, dict):
            for title, node in item.items():
                new_path = path + (title,)
                if isinstance(node, str):
                    # Leaf page
                    flat.append({"title": title, "url": node, "hierarchy": new_path})
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
            flat.append({"title": item, "url": item, "hierarchy": new_path})
        else:
            raise TypeError(f"Unexpected item type {type(item)} in nav")
    return flat

def generate_nav_links_text(output_file: str, *, replace_links: bool = False) -> None:
    """Generate a text file containing navigation structure and links from mkdocs.yaml."""
    # Get path to mkdocs.yaml relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mkdocs_path = os.path.join(os.path.dirname(script_dir), "mkdocs.yml")

    # Load and parse yaml
    with open(mkdocs_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract nav section
    nav = config.get("nav", [])
    flattened = _flatten_nav(nav)
    
    with open(output_file, "w") as f:
        current_section = None
        for item in flattened:
            # Get the top-level section (first item in hierarchy)
            section = item["hierarchy"][0]
            
            # If we're starting a new section, add a heading
            if section != current_section:
                f.write(f"\n# {section}\n\n")
                current_section = section
            
            # Add the item as a bullet point with title and link
            # Include full hierarchy path in title, separated by " > "
            hierarchy_path = " > ".join(item["hierarchy"][1:])
            title = f"{item['title']} ({hierarchy_path})" if hierarchy_path else item['title']
            
            # Process URL based on replace_links flag
            url = item['url']
            if replace_links:
                # Remove .md extension and ensure single trailing slash
                url = url.replace('.md', '')
                url = url.rstrip('/') + '/'
                url = f"https://langchain-ai.github.io/langgraph/{url}"
            
            f.write(f"- [{title}]({url})\n")


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
        generate_nav_links_text(args.output_file, replace_links=args.replace_links)
    else:
        generate_full_llms_text(args.output_file)
