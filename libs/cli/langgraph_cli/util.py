"""General-purpose utilities shared across the LangGraph CLI."""

from collections.abc import Callable

import click


def clean_empty_lines(input_str: str):
    return "\n".join(filter(None, input_str.splitlines()))


def warn_non_wolfi_distro(
    config_json: dict,
    *,
    emit: Callable[[str], None] | None = None,
) -> None:
    """Show warning if image_distro is not set to 'wolfi'.

    When ``emit`` is provided, each warning line is sent through it (used by
    callers that need JSON-aware output). Otherwise falls back to colored
    ``click.secho`` output.
    """
    image_distro = config_json.get("image_distro", "debian")  # Default is debian
    if image_distro == "wolfi":
        return
    if emit is not None:
        emit(
            "⚠️  Security Recommendation: Consider switching to Wolfi Linux for enhanced security."
        )
        emit(
            "   Wolfi is a security-oriented, minimal Linux distribution designed for containers."
        )
        emit(
            '   To switch, add \'"image_distro": "wolfi"\' to your langgraph.json config file.'
        )
        return
    click.secho(
        "⚠️  Security Recommendation: Consider switching to Wolfi Linux for enhanced security.",
        fg="yellow",
        bold=True,
    )
    click.secho(
        "   Wolfi is a security-oriented, minimal Linux distribution designed for containers.",
        fg="yellow",
    )
    click.secho(
        '   To switch, add \'"image_distro": "wolfi"\' to your langgraph.json config file.',
        fg="yellow",
    )
    click.secho("")  # Empty line for better readability
