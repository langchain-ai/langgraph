#!/usr/bin/env python
"""Retrieve download count for a list of Python packages from PyPI."""

import argparse
from datetime import datetime
from typing import TypedDict
import pathlib

import requests
import yaml


class Package(TypedDict):
    """A TypedDict representing a package"""

    name: str
    """The name of the package."""
    repo: str
    """Repository ID within github. Format is: [orgname]/[repo_name]."""
    description: str
    """A brief description of what the package does."""


class ResolvedPackage(Package):
    weekly_downloads: int | None


HERE = pathlib.Path(__file__).parent
PACKAGES_FILE = HERE / "packages.yml"
PACKAGES = yaml.safe_load(PACKAGES_FILE.read_text())['packages']


def _get_weekly_downloads(packages: list[Package]) -> list[ResolvedPackage]:
    """Retrieve the monthly download count for a list of packages from PyPIStats."""
    resolved_packages: list[ResolvedPackage] = []

    for package in packages:
        url = f"https://pypistats.org/api/packages/{package['name']}/overall"

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        sorted_data = sorted(
            data["data"],
            key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"),
            reverse=True,
        )

        # Sum the last 7 days of downloads
        num_downloads = sum(entry["downloads"] for entry in sorted_data[:7])

        resolved_packages.append(
            {
                "name": package["name"],
                "repo": package["repo"],
                "weekly_downloads": num_downloads,
                "description": package["description"],
            }
        )

    return resolved_packages



def main(output_file: str) -> None:
    """Main function to generate package download information.

    Args:
        output_file: Path to the output YAML file.
    """
    resolved_packages: list[ResolvedPackage] = _get_weekly_downloads(PACKAGES)

    if not output_file.endswith(".yml"):
        raise ValueError("Output file must have a .yml extension")

    with open(output_file, "w") as f:
        f.write("# This file is auto-generated. Do not edit.\n")
        yaml.dump(resolved_packages, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate package download information."
    )
    parser.add_argument(
        "output_file",
        help=(
            "Path to the output YAML file. Example: python generate_downloads.py "
            "downloads.yml"
        ),
    )
    args = parser.parse_args()

    main(args.output_file)
