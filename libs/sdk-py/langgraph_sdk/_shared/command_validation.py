"""Shared validation helpers for command payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_RESUME_AUTH_REQUIRED_KEYS = ("actor_id", "token", "signature")
_RESUME_AUTH_ALLOWED_KEYS = frozenset(
    (
        "actor_id",
        "token",
        "signature",
        "key_id",
        "issuer",
        "issued_at",
        "expires_at",
        "scheme",
    )
)
_RESUME_AUTH_OPTIONAL_KEYS = (
    "key_id",
    "issuer",
    "issued_at",
    "expires_at",
    "scheme",
)


def normalize_and_validate_command(
    command: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a command payload with `None` values dropped and validated."""
    if command is None:
        return None

    payload = {k: v for k, v in command.items() if v is not None}
    if "resume_authorization" not in payload:
        return payload

    if "resume" not in payload:
        raise ValueError("`command.resume_authorization` requires `command.resume`.")

    _validate_resume_authorization(payload["resume_authorization"])
    return payload


def _validate_resume_authorization(resume_authorization: Any) -> None:
    if not isinstance(resume_authorization, Mapping):
        raise ValueError("`command.resume_authorization` must be a mapping.")

    missing = [
        key for key in _RESUME_AUTH_REQUIRED_KEYS if key not in resume_authorization
    ]
    if missing:
        raise ValueError(
            "`command.resume_authorization` is missing required key(s): "
            + ", ".join(missing)
        )

    unknown = sorted(
        str(key) for key in resume_authorization if key not in _RESUME_AUTH_ALLOWED_KEYS
    )
    if unknown:
        raise ValueError(
            "`command.resume_authorization` contains unknown key(s): "
            + ", ".join(unknown)
        )

    for key in _RESUME_AUTH_REQUIRED_KEYS:
        value = resume_authorization[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"`command.resume_authorization.{key}` must be a non-empty string."
            )

    for key in _RESUME_AUTH_OPTIONAL_KEYS:
        if key in resume_authorization and not isinstance(
            resume_authorization[key], str
        ):
            raise ValueError(f"`command.resume_authorization.{key}` must be a string.")
