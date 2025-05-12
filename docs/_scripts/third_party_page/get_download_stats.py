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


def _get_weekly_downloads(packages: list[Package], fake: bool) -> list[ResolvedPackage]:
    """Retrieve the monthly download count for a list of packages from PyPIStats."""
    resolved_packages: list[ResolvedPackage] = []

    if fake:
        # To avoid making network requests during testing, return fake download counts
        for package in packages:
            resolved_packages.append(
                {
                    "name": package["name"],
                    "repo": package["repo"],
                    "weekly_downloads": -12345,
                    "description": package["description"],
                }
            )
        return resolved_packages

    for package in packages:
        # First check if package exists on PyPI
        pypi_url = f"https://pypi.org/pypi/{package['name']}/json"
        try:
            pypi_response = requests.get(pypi_url)
            pypi_response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise AssertionError(f"Package {package['name']} does not exist on PyPI")

        # Get first release date
        pypi_data = pypi_response.json()
        releases = pypi_data["releases"]
        first_release_date = None
        for version_releases in releases.values():
            if version_releases:  # Some versions may be empty lists
                upload_time = datetime.fromisoformat(version_releases[0]["upload_time"])
                if first_release_date is None or upload_time < first_release_date:
                    first_release_date = upload_time

        if first_release_date is None:
            raise AssertionError(f"Package {package['name']} has no releases yet")

        # If package was published in last 48 hours, skip download stats
        if (datetime.now() - first_release_date).total_seconds() >= 48 * 3600:
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
        else:
            num_downloads = None

        resolved_packages.append(
            {
                "name": package["name"],
                "repo": package["repo"],
                "weekly_downloads": num_downloads,
                "description": package["description"],
            }
        )

    return resolved_packages



def main(output_file: str, fake: bool) -> None:
    """Main function to generate package download information.

    Args:
        output_file: Path to the output YAML file.
    """
    resolved_packages: list[ResolvedPackage] = _get_weekly_downloads(PACKAGES, fake)

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
    parser.add_argument(
        "--fake",
        default=False,
        action="store_true",
        help=(
            "Generate fake download counts for testing purposes. "
            "This option will not make any network requests."
        ),
    )
    args = parser.parse_args()

    main(args.output_file, args.fake)
