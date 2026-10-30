"""LangSmith OAuth login, shared with the ``langsmith`` CLI via ``~/.langsmith/config.json``."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import time
import webbrowser
from typing import Any
from urllib.parse import urlparse

import click
import httpx

CLIENT_ID = "langsmith-cli"
DEFAULT_API_URL = "https://api.smith.langchain.com"
_WELL_KNOWN = "/.well-known/oauth-authorization-server"
_REFRESH_LEEWAY = datetime.timedelta(seconds=60)
_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


def _config_path() -> pathlib.Path:
    override = os.environ.get("LANGSMITH_CONFIG_FILE")
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".langsmith" / "config.json"


def _read_config() -> dict[str, Any]:
    try:
        config = json.loads(_config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _write_config(config: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _profile_name(config: dict[str, Any]) -> str:
    return (
        os.environ.get("LANGSMITH_PROFILE")
        or config.get("current_profile")
        or "default"
    )


def load_profile() -> dict[str, Any]:
    """Return the active profile from the LangSmith config file, or ``{}``."""
    config = _read_config()
    profile = (config.get("profiles") or {}).get(_profile_name(config))
    return profile if isinstance(profile, dict) else {}


def resolve_api_url(endpoint: str | None = None) -> str:
    """Pick the LangSmith API URL: explicit endpoint, then profile, then cloud."""
    return endpoint or load_profile().get("api_url") or DEFAULT_API_URL


def _save_tokens(token: dict[str, Any], api_url: str | None = None) -> None:
    config = _read_config()
    name = _profile_name(config)
    profile = config.setdefault("profiles", {}).setdefault(name, {})
    if api_url:
        profile["api_url"] = api_url
    oauth = profile.setdefault("oauth", {})
    oauth["access_token"] = token["access_token"]
    if token.get("refresh_token"):
        oauth["refresh_token"] = token["refresh_token"]
    if token.get("expires_in"):
        expires_at = datetime.datetime.now(datetime.timezone.utc) + (
            datetime.timedelta(seconds=int(token["expires_in"]))
        )
        oauth["expires_at"] = expires_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    config.setdefault("current_profile", name)
    _write_config(config)


def _oauth_endpoints(api_url: str) -> dict[str, str]:
    """Discover the OAuth endpoints, falling back to ``<root>/oauth/*``."""
    root = api_url.rstrip("/")
    for suffix in ("/api/v1", "/api"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    for base in (root, f"{root}/api"):
        try:
            resp = httpx.get(base + _WELL_KNOWN, timeout=10)
            doc = resp.json() if resp.is_success else None
        except (httpx.HTTPError, ValueError):
            continue
        if (
            isinstance(doc, dict)
            and str(doc.get("issuer", "")).rstrip("/") == base
            and all(
                urlparse(str(doc.get(key, ""))).netloc == urlparse(base).netloc
                for key in ("token_endpoint", "device_authorization_endpoint")
            )
        ):
            return {
                "issuer": base,
                "token": doc["token_endpoint"],
                "device": doc["device_authorization_endpoint"],
            }
    return {
        "issuer": root,
        "token": f"{root}/oauth/token",
        "device": f"{root}/oauth/device/code",
    }


def _expiring(oauth: dict[str, Any]) -> bool:
    if not oauth.get("access_token"):
        return True
    expires_at = oauth.get("expires_at")
    if not expires_at:
        return False
    try:
        expiry = datetime.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=datetime.timezone.utc)
    return expiry <= datetime.datetime.now(datetime.timezone.utc) + _REFRESH_LEEWAY


def get_access_token(api_url: str) -> str:
    """Return a valid access token from the profile, refreshing it if needed."""
    oauth = load_profile().get("oauth") or {}
    if not _expiring(oauth):
        return oauth["access_token"]
    refresh_token = oauth.get("refresh_token")
    if not refresh_token:
        raise click.ClickException(
            "LangSmith session expired. Run `langgraph login` to sign in again."
        )
    endpoints = _oauth_endpoints(api_url)
    try:
        resp = httpx.post(
            endpoints["token"],
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            },
            timeout=10,
        )
        resp.raise_for_status()
        token = resp.json()
    except (httpx.HTTPError, ValueError):
        raise click.ClickException(
            "Could not refresh your LangSmith session. "
            "Run `langgraph login` to sign in again."
        ) from None
    if not isinstance(token, dict) or not token.get("access_token"):
        raise click.ClickException(
            "Could not refresh your LangSmith session. "
            "Run `langgraph login` to sign in again."
        )
    _save_tokens(token)
    return token["access_token"]


def login(api_url: str, *, open_browser: bool = True) -> None:
    """Sign in with the OAuth device flow and store tokens in the profile."""
    endpoints = _oauth_endpoints(api_url)
    try:
        resp = httpx.post(
            endpoints["device"],
            data={"client_id": CLIENT_ID, "resource": endpoints["issuer"]},
            timeout=10,
        )
        resp.raise_for_status()
        device = resp.json()
        device_code, user_code = device["device_code"], device["user_code"]
        verification_url = (
            device.get("verification_uri_complete") or device["verification_uri"]
        )
    except (httpx.HTTPError, ValueError, KeyError):
        raise click.ClickException(
            f"Could not start LangSmith login against {api_url}."
        ) from None

    click.echo(f"Open this URL to authorize the CLI:\n{verification_url}\n")
    click.echo(f"Code: {user_code}\n")
    if open_browser:
        webbrowser.open(verification_url)
    click.echo("Waiting for authorization...")

    interval = max(int(device.get("interval") or 5), 5)
    deadline = time.monotonic() + int(device.get("expires_in") or 600)
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            resp = httpx.post(
                endpoints["token"],
                data={
                    "grant_type": _DEVICE_GRANT,
                    "client_id": CLIENT_ID,
                    "device_code": device_code,
                    "resource": endpoints["issuer"],
                },
                timeout=10,
            )
            body = resp.json()
        except (httpx.HTTPError, ValueError):
            continue
        if resp.is_success and body.get("access_token"):
            _save_tokens(body, api_url)
            click.secho("Logged in to LangSmith.", fg="green")
            return
        error = body.get("error") if isinstance(body, dict) else None
        if error == "slow_down":
            interval += 5
        elif error != "authorization_pending":
            raise click.ClickException(f"LangSmith login failed: {error or 'error'}.")
    raise click.ClickException("Timed out waiting for LangSmith login.")
