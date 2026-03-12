"""Helpers for the ``langgraph logs`` CLI command."""

from __future__ import annotations

from datetime import datetime, timezone

import click

from langgraph_cli.host_backend import HostBackendClient


def resolve_deployment_id(
    client: HostBackendClient,
    deployment_id: str | None,
    name: str | None,
) -> str:
    """Resolve a deployment ID from --deployment-id or --name."""
    if deployment_id:
        return deployment_id
    if not name:
        raise click.UsageError("Either --deployment-id or --name is required.")
    existing = client.list_deployments(name_contains=name)
    if isinstance(existing, dict):
        for dep in existing.get("resources", []):
            if isinstance(dep, dict) and dep.get("name") == name:
                found_id = dep.get("id")
                if found_id:
                    return str(found_id)
    raise click.ClickException(f"Deployment '{name}' not found.")


def format_timestamp(ts) -> str:
    """Convert a timestamp (epoch ms or string) to a readable string."""
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts) if ts else ""


def format_log_entry(entry: dict) -> str:
    """Format a single log entry for display."""
    ts = format_timestamp(entry.get("timestamp", ""))
    level = entry.get("level", "")
    message = entry.get("message", "")
    if ts and level:
        return f"[{ts}] [{level}] {message}"
    elif ts:
        return f"[{ts}] {message}"
    return message


def level_fg(level: str) -> str | None:
    """Return click color for a log level."""
    level_upper = level.upper() if level else ""
    if level_upper in {"ERROR", "CRITICAL"}:
        return "red"
    if level_upper == "WARNING":
        return "yellow"
    return None
