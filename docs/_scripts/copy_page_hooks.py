"""
Copy page functionality hooks for MkDocs.

This module provides hooks to inject original markdown content into HTML pages
for the copy page functionality, allowing users to copy clean markdown content
optimized for LLMs.
"""

import json
import re
from pathlib import Path
from typing import Optional

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.pages import Page


def _process_includes(content: str, docs_dir: Path) -> str:
    """Process MkDocs includes like {!../README.md!}."""
    include_pattern = r'\{!([^!]+)!\}'
    
    def replace_include(match):
        include_path = match.group(1)
        # Resolve relative path
        if include_path.startswith('../'):
            # Go up from docs dir
            include_file = docs_dir.parent / include_path[3:]
        else:
            include_file = docs_dir / include_path
        
        try:
            with open(include_file, 'r', encoding='utf-8') as f:
                included_content = f.read()
                # Remove frontmatter from included content to avoid duplication
                included_content = re.sub(r'^---\n.*?\n---\n', '', included_content, flags=re.DOTALL)
                return included_content
        except:
            return f"[Content from {include_path}]"
    
    return re.sub(include_pattern, replace_include, content)


def _clean_markdown(content: str) -> str:
    """Minimal cleanup of markdown content - preserve original as much as possible."""
    # Remove frontmatter
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
    
    # Remove script tags (security)
    content = re.sub(r'<script[^>]*>.*?</script\s*>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove style tags (security)
    content = re.sub(r'<style[^>]*>.*?</style\s*>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    
    # Just strip and return - preserve original structure
    return content.strip()


def inject_markdown_content(html: str, page: Page, config: MkDocsConfig) -> str:
    """
    Inject the original markdown content into the HTML for copy page functionality.
    
    Args:
        html: The HTML content to inject into
        page: The MkDocs page object
        config: The MkDocs configuration
        
    Returns:
        Modified HTML with markdown content injected as JSON
    """
    if not hasattr(page, 'file') or not page.file:
        return html
    
    # Get the original markdown file path
    docs_dir = Path(config.get('docs_dir', 'docs'))
    src_path = page.file.src_path
    
    # Handle different file types
    if src_path.endswith('.ipynb'):
        # For notebook files, we might want to use the converted markdown
        # For now, just return the HTML as-is
        return html
    
    markdown_file = docs_dir / src_path
    
    if not markdown_file.exists():
        return html
    
    try:
        # Read the original markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Special handling for index page - use relative path to the actual README.md
        if src_path == 'index.md':
            # Relative path to the repository README.md file (go up two levels from docs/docs)
            readme_path = docs_dir.parent.parent / 'README.md'
            
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                # Remove frontmatter if present
                processed_markdown = re.sub(r'^---\n.*?\n---\n', '', readme_content, flags=re.DOTALL)
                processed_markdown = processed_markdown.strip()
            except Exception as e:
                # If we can't read the README, fallback to original behavior
                processed_markdown = _process_includes(markdown_content, docs_dir)
                processed_markdown = re.sub(r'^---\n.*?\n---\n', '', processed_markdown, flags=re.DOTALL)
                processed_markdown = processed_markdown.strip()
        else:
            # Process any includes in the markdown to get the full content
            processed_markdown = _process_includes(markdown_content, docs_dir)
            # Clean up the processed markdown normally for other pages
            processed_markdown = _clean_markdown(processed_markdown)
        
        # Create the JSON data
        markdown_data = {
            'markdown': processed_markdown,
            'title': page.title or 'Page Content',
            'url': page.url or ''
        }
        
        # Properly escape the JSON for HTML
        json_content = json.dumps(markdown_data, ensure_ascii=False)
        json_content = json_content.replace('</', '\\u003c/')
        json_content = json_content.replace('<script', '\\u003cscript')
        json_content = json_content.replace('</script', '\\u003c/script')
        
        script_content = f'<script id="page-markdown-content" type="application/json">{json_content}</script>'
        
        # Insert before </head> if it exists, otherwise before </body>
        if '</head>' in html:
            html = html.replace('</head>', f'{script_content}</head>')
        elif '</body>' in html:
            html = html.replace('</body>', f'{script_content}</body>')
    
    except Exception as e:
        # If anything goes wrong, just return the original HTML
        # Could log the error here if needed
        pass
    
    return html


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    """
    MkDocs hook to inject markdown content into HTML pages.
    
    This hook is called after each page is rendered and injects the original
    markdown content as JSON for the copy page functionality.
    
    Args:
        output: The HTML output of the page
        page: The MkDocs page object
        config: The MkDocs configuration
        
    Returns:
        Modified HTML with markdown content injected
    """
    return inject_markdown_content(output, page, config)