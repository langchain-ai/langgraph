"""Add typescript translation to a given markdown file."""

import argparse
import re

import requests
from langchain_anthropic import ChatAnthropic

URL = "https://gist.githubusercontent.com/eyurtsev/e7486731415463a9bc5b4682358859c8/raw/b5a5fda9c7e3387cfcb781f25082814d43675d50/gistfile1.txt"
response = requests.get(URL)
response.raise_for_status()
reference_snippets = response.text

model = ChatAnthropic(model="claude-3-5-sonnet-latest")


def _get_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:
        # If not available return a simple identity function
        def tqdm(iterable, *args, **kwargs):
            return iterable

    return tqdm


_tqdm = _get_tqdm()

opening_pattern = re.compile(r"^\s*```python(?:\s+.*)?\s*$")
closing_pattern = re.compile(r"^\s*```\s*$")


def extract_python_snippets(markdown: str) -> list[str]:
    """
    Extract all python code blocks (including their fence lines) from the markdown content.
    A python block is defined as any block that starts with a line containing an opening fence
    with '```python' (optionally with extra parameters) and ends with a closing fence '```'.
    """
    snippets = []
    inside_block = False
    current_snippet = []

    for line in markdown.splitlines(keepends=True):
        if not inside_block:
            if opening_pattern.match(line):
                inside_block = True
                current_snippet = [line]
        else:
            current_snippet.append(line)
            if closing_pattern.match(line):
                inside_block = False
                snippets.append("".join(current_snippet))
                current_snippet = []
    return snippets


def translate_snippet(python_snippet: str) -> str:
    """Translate a python code block into a TypeScript code block using Langchain.
    The response is expected to be a properly fenced TypeScript code block (i.e.
    starting with ```typescript and ending with ```).
    """
    ai_message = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    f"You have access to the following up-to-date example TypeScript code "
                    f"snippets that show examples of building with langgraph "
                    f"and langchain:\n\n{reference_snippets}\n\n"
                    "Use this context to translate the following Python code to equivalent "
                    "TypeScript. Ensure that your output is a valid fenced TypeScript "
                    "code block (i.e. starts with ```typescript and ends with ```)."
                ),
            },
            {
                "role": "user",
                "content": f"Translate this Python snippet to TypeScript:\n\n{python_snippet}",
            },
        ]
    )

    # Use a regular expression to search for a TypeScript code block in the response.
    pattern = r"```typescript\s*(.*?)\s*```"
    match = re.search(pattern, ai_message.content, re.DOTALL)
    if match:
        # Reconstruct the code block with proper fences.
        typescript_code = match.group(1).strip()
        return f"```typescript\n{typescript_code}\n```"
    else:
        raise ValueError("No TypeScript code block found in the model's response.")


def insert_translations_into_markdown(
    markdown: str, typescript_snippets: list[str]
) -> str:
    """Walks through the original markdown content and, after each
    Python snippet block, inserts the corresponding translated TypeScript snippet.
    It assumes that the ordering of the Python snippets
    (from extract_python_snippets) matches the order they appear in the markdown.
    """
    output_lines = []
    lines = markdown.splitlines(keepends=True)
    inside_block = False
    snippet_index = 0

    for line in lines:
        output_lines.append(line)
        if not inside_block and opening_pattern.match(line):
            # We've encountered the start of a python code block.
            inside_block = True
        elif inside_block:
            if closing_pattern.match(line):
                # End of a python snippet block.
                inside_block = False
                if snippet_index < len(typescript_snippets):
                    # Insert an extra newline for clarity, then the translated TypeScript snippet.
                    output_lines.append("\n")
                    output_lines.append(typescript_snippets[snippet_index])
                    output_lines.append("\n")
                    snippet_index += 1
    return "".join(output_lines)


def main(file_path: str) -> None:
    # Read the markdown file.
    with open(file_path, "r") as f:
        markdown_content = f.read()

    # 1. Extract all Python snippets.
    python_snippets = extract_python_snippets(markdown_content)[:1]

    # 2. Translate each Python snippet to TypeScript.
    typescript_snippets = []
    # Replace with .batch() for faster translation
    for python_snippet in _tqdm(python_snippets):
        ts_snippet = translate_snippet(python_snippet)
        typescript_snippets.append(ts_snippet)

    # 3. Insert the TypeScript translations after their respective Python snippets.
    updated_markdown = insert_translations_into_markdown(
        markdown_content, typescript_snippets
    )

    # Overwrite the original markdown file with the updated content.
    with open(file_path, "w") as f:
        f.write(updated_markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate Python snippets in a markdown file to TypeScript and insert them after each Python snippet."
    )
    parser.add_argument("file_path", type=str, help="Path to the markdown file.")
    args = parser.parse_args()

    main(args.file_path)
