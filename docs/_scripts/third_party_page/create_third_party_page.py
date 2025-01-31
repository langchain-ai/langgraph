"""Create the third party page for the documentation."""

import yml
from typing import TypedDict
import argparse
from typing import List

MARKDOWN = """
# Third Party Libraries 

Here is a list of third-party libraries that can be used with LangGraph.

{library_list}

If you'd like to add your library to this list, please open a pull request on the [LangGraph GitHub repository](https://github.com/langchain-ai/langgraph/).
"""


class ResolvedPackage(TypedDict):
    name: str
    """The name of the package."""
    repo: str
    """Repository ID within github. Format is: [orgname]/[repo_name]."""
    weekly_downloads: int | None


def generate_markdown(resolved_packages: List[ResolvedPackage]) -> str:
    """Generate the markdown content for the third party page.

    Args:
        resolved_packages: A list of resolved package information.

    Returns:
        The markdown content as a string.
    """
    sorted_packages = sorted(
        resolved_packages, key=lambda p: p["weekly_downloads"] or 0, reverse=True
    )
    rows = [
        "| Name | GitHub URL | Downloads |",
        "| --- | --- | --- |",
    ]
    for package in sorted_packages:
        name = f"**{package['name']}**"
        repo_url = f"[{package['repo']}](https://github.com/{package['repo']})"
        downloads = package["weekly_downloads"] or 0
        row = f"| {name} | {repo_url} | {downloads} |"
        rows.append(row)
    markdown_content = MARKDOWN.format(library_list="\n".join(rows))
    return markdown_content


def main(input_file: str, output_file: str) -> None:
    """Main function to create the third party page.

    Args:
        input_file: Path to the input YAML file containing resolved package information.
        output_file: Path to the output file for the third party page.
    """
    # Parse the input YAML file
    with open(input_file, "r") as f:
        resolved_packages: List[ResolvedPackage] = yml.safe_load(f)

    markdown_content = generate_markdown(resolved_packages)

    # Write the markdown content to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the third party page.")
    parser.add_argument(
        "input_file",
        help="Path to the input YAML file containing resolved package information.",
    )
    parser.add_argument(
        "output_file", help="Path to the output file for the third party page."
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file)
