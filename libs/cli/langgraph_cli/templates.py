import os
import shutil
import sys
from io import BytesIO
from urllib import error, request
from zipfile import ZipFile

import click

TEMPLATES: dict[str, dict[str, str]] = {
    "Deep Agent": {
        "description": "An opinionated deployment template for a Deep Agent.",
        "python": "https://github.com/langchain-ai/deep-agent-template/archive/refs/heads/main.zip",
        "js": "https://github.com/langchain-ai/deep-agent-template-js/archive/refs/heads/main.zip",
    },
    "Agent": {
        "description": "A simple agent that can be flexibly extended to many tools.",
        "python": "https://github.com/langchain-ai/simple-agent-template/archive/refs/heads/main.zip",
    },
    "New LangGraph Project": {
        "description": "A simple, minimal chatbot with memory.",
        "python": "https://github.com/langchain-ai/new-langgraph-project/archive/refs/heads/main.zip",
        "js": "https://github.com/langchain-ai/new-langgraphjs-project/archive/refs/heads/main.zip",
    },
}

# Generate TEMPLATE_IDS programmatically
TEMPLATE_ID_TO_CONFIG = {
    f"{name.lower().replace(' ', '-')}-{lang}": (name, lang, url)
    for name, versions in TEMPLATES.items()
    for lang, url in versions.items()
    if lang in {"python", "js"}
}

TEMPLATE_IDS = list(TEMPLATE_ID_TO_CONFIG.keys())

TEMPLATE_HELP_STRING = (
    "The name of the template to use. Available options:\n"
    + "\n".join(f"{id_}" for id_ in TEMPLATE_ID_TO_CONFIG)
)


def _choose_template() -> str:
    """Presents a list of templates to the user and prompts them to select one.

    Returns:
        str: The URL of the selected template.
    """
    click.secho("🌟 Please select a template:", bold=True, fg="yellow")
    for idx, (template_name, template_info) in enumerate(TEMPLATES.items(), 1):
        click.secho(f"{idx}. ", nl=False, fg="cyan")
        click.secho(template_name, fg="cyan", nl=False)
        click.secho(f" - {template_info['description']}", fg="white")

    # Get the template choice from the user, defaulting to the first template if blank
    template_choice: int | None = click.prompt(
        "Enter the number of your template choice (default is 1)",
        type=int,
        default=1,
        show_default=False,
    )

    template_keys = list(TEMPLATES.keys())
    if 1 <= template_choice <= len(template_keys):
        selected_template: str = template_keys[template_choice - 1]
    else:
        click.secho("❌ Invalid choice. Please try again.", fg="red")
        return _choose_template()

    template_info = TEMPLATES[selected_template]
    available_langs = [lang for lang in ("python", "js") if lang in template_info]

    click.secho(
        f"\nYou selected: {selected_template} - {template_info['description']}",
        fg="green",
    )

    if len(available_langs) == 1:
        return template_info[available_langs[0]]

    version_choice: int = click.prompt(
        "Choose language (1 for Python 🐍, 2 for JS/TS 🌐)", type=int
    )

    if version_choice == 1:
        return template_info["python"]
    elif version_choice == 2:
        return template_info["js"]
    else:
        click.secho("❌ Invalid choice. Please try again.", fg="red")
        return _choose_template()


def _download_repo_with_requests(repo_url: str, path: str) -> None:
    """Download a ZIP archive from the given URL and extracts it to the specified path.

    Args:
        repo_url: The URL of the repository to download.
        path: The path where the repository should be extracted.
    """
    click.secho("📥 Attempting to download repository as a ZIP archive...", fg="yellow")
    click.secho(f"URL: {repo_url}", fg="yellow")
    try:
        with request.urlopen(repo_url) as response:
            if response.status == 200:
                with ZipFile(BytesIO(response.read())) as zip_file:
                    zip_file.extractall(path)
                    # Move extracted contents to path
                    for item in os.listdir(path):
                        if item.endswith("-main"):
                            extracted_dir = os.path.join(path, item)
                            for filename in os.listdir(extracted_dir):
                                shutil.move(os.path.join(extracted_dir, filename), path)
                            shutil.rmtree(extracted_dir)
                click.secho(
                    f"✅ Downloaded and extracted repository to {path}", fg="green"
                )
    except error.HTTPError as e:
        click.secho(
            f"❌ Error: Failed to download repository.\nDetails: {e}\n",
            fg="red",
            bold=True,
            err=True,
        )
        sys.exit(1)


def create_new(path: str | None, template: str | None) -> None:
    """Create a new LangGraph project at the specified PATH using the chosen TEMPLATE.

    Args:
        path: The path where the new project will be created.
        template: The name of the template to use.
    """
    # Prompt for path if not provided
    if not path:
        path = click.prompt(
            "📂 Please specify the path to create the application", default="."
        )

    path = os.path.abspath(path)  # Ensure path is absolute

    # Check if path exists and is not empty
    if os.path.exists(path) and os.listdir(path):
        click.secho(
            "❌ The specified directory already exists and is not empty. "
            "Aborting to prevent overwriting files.",
            fg="red",
            bold=True,
        )
        sys.exit(1)

    # Get template URL either from command-line argument or
    # through interactive selection
    if template:
        if template not in TEMPLATE_ID_TO_CONFIG:
            # Format available options in a readable way with descriptions
            template_options = ""
            for id_ in TEMPLATE_IDS:
                name, lang, _ = TEMPLATE_ID_TO_CONFIG[id_]
                description = TEMPLATES[name]["description"]

                # Add each template option with color formatting
                template_options += (
                    click.style("- ", fg="yellow", bold=True)
                    + click.style(f"{id_}", fg="cyan")
                    + click.style(f": {description}", fg="white")
                    + "\n"
                )

            # Display error message with colors and formatting
            click.secho("❌ Error:", fg="red", bold=True, nl=False)
            click.secho(f" Template '{template}' not found.", fg="red")
            click.secho(
                "Please select from the available options:\n", fg="yellow", bold=True
            )
            click.secho(template_options, fg="cyan")
            sys.exit(1)
        _, _, template_url = TEMPLATE_ID_TO_CONFIG[template]
    else:
        template_url = _choose_template()

    # Download and extract the template
    _download_repo_with_requests(template_url, path)

    click.secho(f"🎉 New project created at {path}", fg="green", bold=True)
