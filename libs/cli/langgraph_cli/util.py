from collections.abc import Sequence

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


def _extract_deployment_url(deployment: dict[str, object]) -> str:
    source_config = deployment.get("source_config")
    if isinstance(source_config, dict):
        custom_url = source_config.get("custom_url")
        if isinstance(custom_url, str) and custom_url:
            return custom_url
    return "-"


def format_deployments_table(deployments: Sequence[dict[str, object]]) -> str:
    headers = ("Deployment ID", "Deployment Name", "Deployment URL")
    rows = [
        (
            str(deployment.get("id", "-") or "-"),
            str(deployment.get("name", "-") or "-"),
            _extract_deployment_url(deployment),
        )
        for deployment in deployments
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    lines = [format_row(headers), format_row(tuple("-" * width for width in widths))]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def format_revisions_table(revisions: Sequence[dict[str, object]]) -> str:
    headers = ("Revision ID", "Status", "Created At")
    latest_deployed_seen = False
    rows = []
    for revision in revisions:
        status = str(revision.get("status", "-") or "-")
        if status == "DEPLOYED":
            if latest_deployed_seen:
                status = "REPLACED"
            else:
                latest_deployed_seen = True
        rows.append(
            (
                str(revision.get("id", "-") or "-"),
                status,
                str(revision.get("created_at", "-") or "-"),
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    lines = [format_row(headers), format_row(tuple("-" * width for width in widths))]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)
