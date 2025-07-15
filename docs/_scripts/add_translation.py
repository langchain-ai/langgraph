"""Translate Python markdown to TypeScript and/or consolidate Python-JS markdown into a single document."""

import argparse

import requests
from langchain_anthropic import ChatAnthropic
from textwrap import dedent


# Load reference TypeScript snippets
URL = "https://gist.githubusercontent.com/dqbd/b35d49e2ceec80e654fe1c5ab61ec477/raw/f4768aeedb67628190a4e06d063a938afc8e7672/snippets.md"
response = requests.get(URL)
response.raise_for_status()
reference_snippets = response.text

# Initialize model
model = ChatAnthropic(model="claude-sonnet-4-0", max_tokens=64_000)


FLUENT_INTERFACE_PROMPT = (
    "CRITICAL: Always use method chaining (fluent interface) for StateGraph operations in TypeScript. "
    "Never create separate variables for the graph builder or call methods individually. "
    "The fluent interface provides better type safety and is the preferred pattern.\n\n"
    "CORRECT examples with fluent interface:\n"
    + dedent(
        """
        ```typescript
        const graph = new StateGraph(MyState)
          .addNode('node1', node1)
          .addNode('node2', node2)
          .addEdge(START, 'node1')
          .addEdge('node1', 'node2')
          .addEdge('node2', END)
          .compile()
        ```
        
        ```typescript
        const graph = new StateGraph(MyState)
          .addNode('chatbot', chatbot)
          .addEdge(START, 'chatbot')
          .addEdge('chatbot', END)
          .compile()
        ```
        
        ```typescript
        const graph = new StateGraph(MyState)
          .addNode('chatbot', chatbot)
          .addEdge(START, 'chatbot')
          .addEdge('chatbot', END)
          .compile()
        ```
        """
    )
    + "\n"
    + "INCORRECT examples to avoid:\n"
    + dedent(
        """
        ```typescript
        // WRONG: Creating separate builder variable
        const graphBuilder = new StateGraph(MyState)
        graphBuilder.addNode('node1', node1)
        graphBuilder.addEdge(START, 'node1')
        const graph = graphBuilder.compile()
        ```
        
        ```typescript
        // WRONG: Using Python-style method names
        const workflow = new StateGraph(MyState)
        workflow.add_node('node1', node1)
        workflow.add_edge(START, 'node1')
        const graph = workflow.compile()
        ```
        
        ```typescript
        // WRONG: Calling methods individually
        const graphBuilder = new StateGraph(MyState)
        graphBuilder.addNode('chatbot', chatbot)
        graphBuilder.addEdge(START, 'chatbot')
        graphBuilder.addEdge('chatbot', END)
        const graph = graphBuilder.compile()
        ```
        """
    )
    + "\n"
    + "Key rules:\n"
    + "- Always chain methods directly on the StateGraph constructor\n"
    + "- Use camelCase method names (addNode, addEdge, not add_node, add_edge)\n"
    + "- Always end with .compile()\n"
    + "- Never store the builder in a separate variable\n"
)


TRANSLATION_PROMPT = (
    "You are a helpful assistant that translates Python-based technical "
    "documentation written in Markdown to equivalent TypeScript-based documentation. "
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
    "maintain alignment with existing conventions.\n\n"
    "IMPORTANT REQUIREMENTS:\n"
    "- Use Zod for state definition for StateGraph. Avoid using Annotation since it will be deprecated in the future.\n"
    "- ALWAYS use fluent interface (method chaining) for StateGraph operations - this is CRITICAL\n"
    "- Never create separate variables for graph builders\n"
    "- Always chain methods directly on the StateGraph constructor and end with .compile()\n\n"
    f"{FLUENT_INTERFACE_PROMPT}\n\n"
    f"Here are the reference TypeScript snippets:\n\n{reference_snippets}\n\n"
)

CONSOLIDATION_PROMPT = (
    "You are a helpful assistant that consolidates parallel Python and JavaScript (TypeScript) technical documentation "
    "written in Markdown into a single unified Markdown document. "
    "The input consists of two documents: the first is for Python users, and the second is for JavaScript/TypeScript users. "
    "Your task is to merge these into one Markdown file using language-specific fenced blocks to separate the content where needed. "
    "Use the following syntax to distinguish content for each language:\n\n"
    ":::python\n"
    "# Python-specific content\n"
    ":::\n\n"
    ":::js\n"
    "# JavaScript/TypeScript-specific content\n"
    ":::\n\n"
    "Follow these consolidation rules:\n"
    "- When content (prose or code) is the same or nearly identical in both versions, include it only once—outside of any fenced block.\n"
    "- When content differs between the Python and JS versions, wrap each version in its corresponding fenced block.\n"
    "- Prefer **paragraph-level separation** of language-specific content. Do not combine Python and JS snippets or terminology in the same sentence or paragraph using conditional phrases.\n"
    "  For example, avoid inline constructs like:\n"
    "  `The :::python add_messages ::: :::js reducer ::: function...`\n"
    "  Instead, write two distinct paragraphs:\n\n"
    "  :::python\n"
    "  The `add_messages` function in our `State` will append the LLM's response messages to whatever messages are already in the state.\n"
    "  ::: \n\n"
    "  :::js\n"
    "  The `reducer` function in our `StateAnnotation` will append the LLM's response messages to whatever messages are already in the state.\n"
    "  :::\n\n"
    "- Preserve the overall structure, ordering, and formatting of the original Markdown documents.\n"
    "- Do not rephrase or unify content unless it is logically and semantically identical.\n"
    "- Use the fenced blocks for both prose and code as needed, and ensure output is clean, readable Markdown suitable for tools that parse these directives.\n"
    "Your goal is to produce a cleanly merged documentation file that serves both Python and JavaScript users without redundancy, while maximizing clarity and separation of language-specific details."
)


def translate_python_to_ts(markdown_content: str) -> str:
    response = model.invoke(
        [
            {
                "role": "system",
                "content": TRANSLATION_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": markdown_content},
        ]
    )
    return response.content


def consolidate_python_and_ts(combined_content: str) -> str:
    response = model.invoke(
        [
            {
                "role": "system",
                "content": CONSOLIDATION_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": combined_content},
        ]
    )
    return response.content


def main(file_path: str, translate_only: bool, consolidate_only: bool) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    if translate_only:
        translated = translate_python_to_ts(markdown_content)
        output_path = file_path.replace(".md", ".translated.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translated)
        print(f"Translated JS/TS version written to: {output_path}")

    elif consolidate_only:
        consolidated = consolidate_python_and_ts(markdown_content)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(consolidated)
        print(f"Consolidated content written to: {file_path}")

    else:
        # Default behavior: translate first, then consolidate both
        translated = translate_python_to_ts(markdown_content)
        combined = f"{markdown_content.strip()}\n\n\n{translated.strip()}"
        consolidated = consolidate_python_and_ts(combined)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(consolidated)
        print(f"Translated and consolidated content written to: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Translate Python markdown to TypeScript and/or consolidate "
            "Python-JS markdown into one file."
        )
    )
    parser.add_argument("file_path", type=str, help="Path to the markdown file.")
    parser.add_argument(
        "--translate-only",
        action="store_true",
        help="Only generate the JS translation.",
    )
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="Only consolidate pre-paired Python and JS content.",
    )
    args = parser.parse_args()

    if args.translate_only and args.consolidate_only:
        raise ValueError(
            "Cannot use both --translate-only and --consolidate-only at the same time."
        )

    main(
        args.file_path,
        translate_only=args.translate_only,
        consolidate_only=args.consolidate_only,
    )
