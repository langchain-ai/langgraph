import click


def clean_empty_lines(input_str: str):
    return "\n".join(filter(None, input_str.splitlines()))


def warn_non_wolfi_distro(config_json: dict) -> None:
    """Show warning if image_distro is not set to 'wolfi'."""
    image_distro = config_json.get("image_distro", "debian")  # Default is debian
    if image_distro != "wolfi":
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
