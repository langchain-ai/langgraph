"""Errors for graph capability contracts and composition."""

from __future__ import annotations


class CapabilityError(Exception):
    """Base error for capability contract violations and runtime failures."""


class CapabilityContractError(CapabilityError, ValueError):
    """Raised when a capability contract is invalid or incomplete."""


class CapabilitySchemaError(CapabilityError, TypeError):
    """Raised when input/output does not satisfy the capability schemas."""


class CapabilityVersionError(CapabilityError, ValueError):
    """Raised when a requested capability version is unsupported."""


class CapabilityInvocationError(CapabilityError, RuntimeError):
    """Raised when a capability fails at the boundary (local or remote)."""

    def __init__(
        self,
        message: str,
        *,
        capability_id: str | None = None,
        version: str | None = None,
        run_id: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.capability_id = capability_id
        self.version = version
        self.run_id = run_id
        self.__cause__ = cause
