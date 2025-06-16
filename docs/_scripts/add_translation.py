"""Add typescript translation to a given markdown file."""

import argparse

import requests
from langchain_anthropic import ChatAnthropic

URL = "https://gist.githubusercontent.com/eyurtsev/e7486731415463a9bc5b4682358859c8/raw/b5a5fda9c7e3387cfcb781f25082814d43675d50/gistfile1.txt"
response = requests.get(URL)
response.raise_for_status()
reference_snippets = response.text

model = ChatAnthropic(model="claude-sonnet-4-0", max_tokens=64_000)


def main(file_path: str) -> None:
    # Read the markdown file.
    with open(file_path, "r") as f:
        markdown_content = f.read()

    ai_message = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that translates Python-based technical "
                    "documentation written in Markdown to equivalent TypeScript-based "
                    "documentation. "
                    "The input is a Markdown file written in mkdocs format. It contains "
                    "Python code snippets embedded in prose. "
                    "Your task is to rewrite the content by translating the Python code to "
                    "idiomatic TypeScript, using the provided TypeScript reference snippets "
                    "to ensure accurate and consistent usage (e.g., correct imports, function "
                    "names, and patterns). "
                    "Remove the original Python code and replace it with the corresponding "
                    "TypeScript version. "
                    "Do not alter the surrounding prose unless a change is necessary to "
                    "reflect differences between Python and TypeScript. "
                    "Preserve the structure and formatting of the original Markdown document. "
                    "Do not make stylistic or structural changes unless they directly support "
                    "the translation. "
                    "Use the reference TypeScript snippets as guidance whenever possible to "
                    "maintain alignment with existing conventions."
                    "Here are the reference TypeScript snippets:\n\n"
                    f"{reference_snippets}\n\n"
                ),
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": markdown_content},
        ]
    )

    with open(file_path, "w") as f:
        f.write(ai_message.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate Python snippets in a markdown file to TypeScript and insert them after each Python snippet."
    )
    parser.add_argument("file_path", type=str, help="Path to the markdown file.")
    args = parser.parse_args()

    main(args.file_path)
