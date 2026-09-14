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

import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Default fallback URL for any path not in the redirect map
DEFAULT_REDIRECT = "https://docs.langchain.com/oss/python/langgraph/overview"

# The docs site regenerates this index on every deploy, so fetching it here
# keeps the published llms.txt from drifting. The URL is a hardcoded constant,
# never built from input, and both it and the post-redirect URL are checked
# against ALLOWED_LLMS_HOST before anything is read.
CANONICAL_LLMS_URL = "https://docs.langchain.com/oss/python/langgraph/llms.txt"
ALLOWED_LLMS_HOST = "docs.langchain.com"
LLMS_FETCH_TIMEOUT = 30
LLMS_MAX_BYTES = 1_000_000

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


def is_allowed_llms_url(url):
    """Return True if url is HTTPS on the one host we accept content from."""
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme == "https" and parsed.hostname == ALLOWED_LLMS_HOST


def fetch_canonical_llms_txt():
    """Return the published LangGraph index, or None if it cannot be used.

    Returning None leaves the caller on the committed docs/llms.txt, so a
    docs.langchain.com outage degrades to a stale file rather than a broken
    deploy or a published error page.
    """
    if not is_allowed_llms_url(CANONICAL_LLMS_URL):
        print(f"Refusing to fetch {CANONICAL_LLMS_URL}: host not allowed")
        return None

    try:
        with urllib.request.urlopen(  # noqa: S310 - constant, allowlisted URL
            CANONICAL_LLMS_URL, timeout=LLMS_FETCH_TIMEOUT
        ) as response:
            # urlopen follows redirects, so re-check where it actually landed.
            if not is_allowed_llms_url(response.url):
                print(f"Refusing {CANONICAL_LLMS_URL}: redirected to {response.url}")
                return None
            body = response.read(LLMS_MAX_BYTES + 1)
    # A connection dropped mid-body raises http.client.IncompleteRead, which
    # descends from HTTPException rather than OSError, so catching only the
    # urllib and OS errors would let it escape and fail the whole deploy.
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
    ) as exc:
        print(f"Could not fetch {CANONICAL_LLMS_URL}: {type(exc).__name__}: {exc}")
        return None

    if len(body) > LLMS_MAX_BYTES:
        print(f"Refusing {CANONICAL_LLMS_URL}: larger than {LLMS_MAX_BYTES} bytes")
        return None

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"Refusing {CANONICAL_LLMS_URL}: not valid UTF-8: {exc}")
        return None

    # An index opens with a markdown heading and links to the docs site. A
    # body that does not is an error page or a truncated response, not content
    # worth publishing.
    if not text.startswith("# ") or f"https://{ALLOWED_LLMS_HOST}/" not in text:
        print(f"Refusing {CANONICAL_LLMS_URL}: does not look like an llms.txt index")
        return None

    return text


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

    # llms.txt can't be redirected via HTML, so publish the docs site's own
    # generated index. The committed copy is only a fallback.
    llms_txt = fetch_canonical_llms_txt()
    if llms_txt is not None:
        (output_dir / "llms.txt").write_text(llms_txt)
        print(f"Fetched: {output_dir / 'llms.txt'} (from {CANONICAL_LLMS_URL})")
    else:
        src = script_dir / "llms.txt"
        if src.exists():
            (output_dir / "llms.txt").write_text(src.read_text())
            print(f"Copied: {output_dir / 'llms.txt'} (fallback, may be stale)")
        else:
            print("No llms.txt fetched and no committed fallback; skipping")

    print(f"\nGenerated {len(redirects)} redirect files in {output_dir}")


if __name__ == "__main__":
    generate_redirects()
