#!/usr/bin/env python3
"""
Generate HTML redirect files from redirects.json.

Usage:
    python generate_redirects.py

This script reads redirects.json and generates individual HTML files
for each redirect path. Each HTML file uses meta refresh (0 delay)
which is SEO-friendly and treated similarly to 301 redirects by Google.

To add new redirects, simply edit redirects.json and re-run this script.
"""

import json
import os
from pathlib import Path

# Default fallback URL for any path not in the redirect map
DEFAULT_REDIRECT = "https://docs.langchain.com/oss/python/langgraph/overview"

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Redirecting...</title>
    <link rel="canonical" href="{url}">
    <meta name="robots" content="noindex">
    <script>var anchor=window.location.hash.substr(1);location.href="{url}"+(anchor?"#"+anchor:"")</script>
    <meta http-equiv="refresh" content="0; url={url}">
</head>
<body>
Redirecting...
</body>
</html>
"""

ROOT_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Redirecting to LangGraph Documentation</title>
    <link rel="canonical" href="{url}">
    <meta name="robots" content="noindex">
    <script>var anchor=window.location.hash.substr(1);location.href="{url}"+(anchor?"#"+anchor:"")</script>
    <meta http-equiv="refresh" content="0; url={url}">
</head>
<body>
<h1>Documentation has moved</h1>
<p>The LangGraph documentation has moved to <a href="{url}">docs.langchain.com</a>.</p>
<p>Redirecting you now...</p>
</body>
</html>
"""

CATCHALL_404_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Redirecting to LangGraph Documentation</title>
    <link rel="canonical" href="{default_url}">
    <meta name="robots" content="noindex">
    <script>
        // Catchall redirect for any unmapped paths
        window.location.replace("{default_url}");
    </script>
    <meta http-equiv="refresh" content="0; url={default_url}">
</head>
<body>
<h1>Documentation has moved</h1>
<p>The LangGraph documentation has moved to <a href="{default_url}">docs.langchain.com</a>.</p>
<p>Redirecting you now...</p>
</body>
</html>
"""


def generate_redirects():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "_site"

    # Load redirects
    with open(script_dir / "redirects.json") as f:
        redirects = json.load(f)

    # Clean output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Generate individual HTML files for each redirect
    for old_path, new_url in redirects.items():
        # Remove leading slash and create directory structure
        path = old_path.lstrip("/")

        # Check if path has a file extension (e.g., .txt, .xml)
        # If so, create the file directly instead of a directory with index.html
        path_obj = Path(path)
        has_extension = path_obj.suffix and len(path_obj.suffix) <= 5

        if not path:
            html_path = output_dir / "index.html"
        elif has_extension:
            # For files with extensions, create the file directly
            html_path = output_dir / path
        else:
            # For directory-style URLs, create index.html inside
            html_path = output_dir / path / "index.html"

        # Create parent directories
        html_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the redirect HTML
        html_path.write_text(HTML_TEMPLATE.format(url=new_url))
        print(f"Created: {html_path}")

    # Create root index.html
    root_index = output_dir / "index.html"
    if not root_index.exists():
        root_index.write_text(ROOT_HTML_TEMPLATE.format(url=DEFAULT_REDIRECT))
        print(f"Created: {root_index}")

    # Create 404.html for catchall
    catchall_404 = output_dir / "404.html"
    catchall_404.write_text(CATCHALL_404_TEMPLATE.format(default_url=DEFAULT_REDIRECT))
    print(f"Created: {catchall_404}")

    # Copy static files (like llms.txt) that can't be redirected via HTML
    static_files = ["llms.txt"]
    for static_file in static_files:
        src = script_dir / static_file
        if src.exists():
            dst = output_dir / static_file
            dst.write_text(src.read_text())
            print(f"Copied: {dst}")

    print(f"\nGenerated {len(redirects)} redirect files in {output_dir}")


if __name__ == "__main__":
    generate_redirects()
