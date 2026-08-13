"""LangGraph tools for discovering and safely requesting TaskMarket work."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

DEFAULT_API_URL = "https://api.taskmarket.dev"
DEFAULT_CLI = "taskmarket"
BASE_CHAIN_ID = 8453
USDC_CONTRACT = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
PLATFORM_FEE_BPS = Decimal("750")
RELAY_FEE_USDC = Decimal("0.001")
USDC_QUANTUM = Decimal("0.000001")
PREVIEW_TTL = timedelta(minutes=15)
SUPPORTED_MODES = frozenset({"bounty", "claim", "pitch", "benchmark"})

JsonTransport = Callable[[str, str, dict[str, str], Any], Any]


class _ListTasksInput(BaseModel):
    status: str = Field(default="open", description="Task status to list.")
    mode: str | None = Field(default=None, description="Optional task mode.")
    tags: list[str] | None = Field(
        default=None, description="Optional tags that must match."
    )
    limit: int = Field(default=20, ge=1, le=100, description="Maximum tasks to return.")


class _TaskIdInput(BaseModel):
    task_id: str = Field(description="0x-prefixed TaskMarket task ID.")


class _PreviewTaskInput(BaseModel):
    description: str = Field(description="The complete task deliverable description.")
    reward_usdc: str = Field(description="Positive USDC reward, up to 6 decimals.")
    duration_hours: float = Field(description="How long the task should remain open.")
    mode: str = Field(default="bounty", description="TaskMarket task mode.")
    tags: list[str] | None = Field(default=None, description="Optional task tags.")


class _CreateTaskInput(_PreviewTaskInput):
    confirmation_token: str = Field(
        description="Unchanged token returned by taskmarket_preview_task."
    )
    confirm: bool = Field(
        default=False,
        description="Must be true in addition to the configured approval callback.",
    )


@dataclass(frozen=True)
class _PendingPreview:
    request: dict[str, Any]
    deadline: datetime
    expires_at: datetime
    maximum_spend: Decimal


@dataclass(frozen=True)
class _CliResult:
    succeeded: bool
    data: Any = None
    error: str | None = None
    ambiguous: bool = False


class TaskMarketClient:
    """Client for the public TaskMarket API used by the bundled tools."""

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_API_URL,
        cli_path: str = DEFAULT_CLI,
        timeout: float = 15.0,
        transport: JsonTransport | None = None,
        clock: Callable[[], datetime] | None = None,
        approval: Callable[[dict[str, Any]], bool] | None = None,
        cli_runner: Callable[[list[str], bool], _CliResult] | None = None,
    ) -> None:
        if not api_url.strip():
            raise ValueError("api_url must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self.api_url = api_url.rstrip("/")
        self.cli_path = cli_path
        self.timeout = timeout
        self._transport = transport
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._pending_previews: dict[str, _PendingPreview] = {}
        self._approval = approval
        self._cli_runner = cli_runner

    def list_tasks(
        self,
        status: str = "open",
        mode: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> Any:
        """List live TaskMarket tasks without spending funds."""
        if not 1 <= limit <= 100:
            return {"error": "limit must be between 1 and 100"}
        params = {"status": status, "limit": str(limit)}
        if mode:
            params["mode"] = mode
        if tags:
            params["tags"] = ",".join(tag.strip() for tag in tags if tag.strip())
        return self._request_json("GET", "/api/tasks", params=params)

    def get_task(self, task_id: str) -> Any:
        """Retrieve the live status of a TaskMarket task."""
        error = _validate_task_id(task_id)
        if error:
            return {"error": error, "retry": False}
        return self._request_json("GET", f"/api/tasks/{task_id}")

    def list_submissions(self, task_id: str) -> Any:
        """List submissions for human review without accepting any work."""
        error = _validate_task_id(task_id)
        if error:
            return {"error": error, "retry": False}
        return self._request_json("GET", f"/api/tasks/{task_id}/submissions")

    def preview_task(
        self,
        description: str,
        reward_usdc: str,
        duration_hours: float,
        mode: str = "bounty",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Prepare an exact, reviewable task request without spending funds."""
        try:
            request = self._normalise_request(
                description=description,
                reward_usdc=reward_usdc,
                duration_hours=duration_hours,
                mode=mode,
                tags=tags,
            )
        except ValueError as exc:
            return {"error": str(exc), "retry": False}

        now = self._utc_now()
        try:
            deadline = now + timedelta(hours=float(request["durationHours"]))
        except (OverflowError, ValueError) as exc:
            return {"error": f"duration_hours is too large: {exc}", "retry": False}
        if deadline <= now:
            return {
                "error": "duration_hours must produce a future deadline",
                "retry": False,
            }

        request["deadline"] = _format_datetime(deadline)
        maximum_spend = _maximum_spend(Decimal(request["rewardUsdc"]))
        request["maximumSpendUsdc"] = _format_usdc(maximum_spend)
        token = _confirmation_digest(request)
        expires_at = min(deadline, now + PREVIEW_TTL)
        self._pending_previews[token] = _PendingPreview(
            request=request,
            deadline=deadline,
            expires_at=expires_at,
            maximum_spend=maximum_spend,
        )
        return {
            **request,
            "network": "Base",
            "chainId": BASE_CHAIN_ID,
            "currency": "USDC",
            "usdcContract": USDC_CONTRACT,
            "confirmationToken": token,
            "expiresAt": _format_datetime(expires_at),
        }

    def _normalise_request(
        self,
        *,
        description: str,
        reward_usdc: str,
        duration_hours: float,
        mode: str,
        tags: list[str] | None,
    ) -> dict[str, Any]:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must not be empty")
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"mode must be one of: {', '.join(sorted(SUPPORTED_MODES))}"
            )
        try:
            reward = Decimal(str(reward_usdc).strip())
        except (InvalidOperation, ValueError):
            raise ValueError("reward_usdc must be a positive USDC amount") from None
        if not reward.is_finite() or reward <= 0:
            raise ValueError("reward_usdc must be a positive USDC amount")
        exponent = reward.as_tuple().exponent
        if isinstance(exponent, str) or exponent < -6:
            raise ValueError("reward_usdc supports at most 6 decimal places")
        reward = reward.quantize(USDC_QUANTUM)

        try:
            duration = Decimal(str(duration_hours))
        except (InvalidOperation, ValueError):
            raise ValueError(
                "duration_hours must be a positive finite number"
            ) from None
        if not duration.is_finite() or duration <= 0:
            raise ValueError("duration_hours must be a positive finite number")
        clean_tags = [tag.strip() for tag in tags or [] if tag.strip()]
        return {
            "description": description.strip(),
            "rewardUsdc": _format_usdc(reward),
            "durationHours": _format_decimal(duration),
            "mode": mode,
            "tags": clean_tags,
        }

    def _utc_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    def create_task(
        self,
        description: str,
        reward_usdc: str,
        duration_hours: float,
        confirmation_token: str,
        confirm: bool = False,
        mode: str = "bounty",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a task only after preview, matching arguments, and approval."""
        if not confirm:
            return {
                "error": (
                    "Creation is confirmation-gated. Review taskmarket_preview_task first, "
                    "then call taskmarket_create_task with confirm=true."
                ),
                "retry": False,
            }
        if not confirmation_token:
            return {
                "error": "A confirmation_token from taskmarket_preview_task is required.",
                "retry": False,
            }
        pending = self._pending_previews.get(confirmation_token)
        if pending is None:
            return {
                "error": "The preview is missing or has already been used.",
                "retry": False,
            }
        now = self._utc_now()
        if now >= pending.expires_at or now >= pending.deadline:
            self._pending_previews.pop(confirmation_token, None)
            return {
                "error": "The preview expired. Run taskmarket_preview_task again.",
                "retry": False,
            }
        try:
            request = self._normalise_request(
                description=description,
                reward_usdc=reward_usdc,
                duration_hours=duration_hours,
                mode=mode,
                tags=tags,
            )
        except ValueError as exc:
            return {"error": str(exc), "retry": False}
        if request != {
            key: pending.request[key] for key in request if key in pending.request
        }:
            return {
                "error": (
                    "The creation arguments differ from the reviewed preview. "
                    "Run taskmarket_preview_task again and review the new record."
                ),
                "retry": False,
            }
        if self._approval is None:
            return {
                "error": (
                    "No approval callback is configured. Connect a human approval "
                    "step before creating a TaskMarket task."
                ),
                "retry": False,
                "status": "blocked",
            }
        preview = {
            **pending.request,
            "network": "Base",
            "chainId": BASE_CHAIN_ID,
            "currency": "USDC",
            "usdcContract": USDC_CONTRACT,
            "confirmationToken": confirmation_token,
            "expiresAt": _format_datetime(pending.expires_at),
        }
        if not self._approval(preview):
            return {
                "error": "The configured approval callback did not authorize task creation.",
                "retry": False,
                "status": "denied",
            }
        preflight = self._preflight_cli(pending.maximum_spend)
        if not preflight.succeeded:
            return {
                "error": preflight.error or "TaskMarket wallet preflight failed.",
                "retry": False,
                "status": "blocked",
            }
        remaining_hours = (pending.deadline - now).total_seconds() / 3600
        if remaining_hours <= 0:
            self._pending_previews.pop(confirmation_token, None)
            return {
                "error": "The reviewed deadline has passed. Run taskmarket_preview_task again.",
                "retry": False,
            }
        cli_args = [
            "task",
            "create",
            "--description",
            pending.request["description"],
            "--reward",
            pending.request["rewardUsdc"],
            "--duration",
            f"{remaining_hours:.9f}",
            "--mode",
            pending.request["mode"],
        ]
        if pending.request["tags"]:
            cli_args.extend(["--tags", ",".join(pending.request["tags"])])
        result = self._run_cli(cli_args, True)
        if not result.succeeded:
            return {
                "error": (
                    result.error
                    or "Task creation failed; inspect live TaskMarket status before retrying."
                ),
                "retry": False,
                "status": "unknown" if result.ambiguous else "failed",
            }
        task_id = _task_id_from_cli_result(result.data)
        if not task_id:
            return {
                "error": (
                    "The CLI returned no task ID. Inspect TaskMarket live status "
                    "before retrying."
                ),
                "retry": False,
                "status": "unknown",
            }
        self._pending_previews.pop(confirmation_token, None)
        return {
            "taskId": task_id,
            "taskUrl": f"{self.api_url}/api/tasks/{task_id}",
            "status": "created",
            "retry": False,
        }

    def _preflight_cli(self, maximum_spend: Decimal) -> _CliResult:
        """Verify the first-party wallet is Base/USDC-funded before a write."""
        if self._cli_runner is None and shutil.which(self.cli_path) is None:
            return _CliResult(
                succeeded=False,
                error=(
                    "The first-party taskmarket CLI was not found. Install "
                    "@lucid-agents/taskmarket and retry."
                ),
            )
        deposit = self._run_cli(["deposit"], False)
        if not deposit.succeeded or not isinstance(deposit.data, dict):
            return _CliResult(
                succeeded=False,
                error=deposit.error or "Could not verify TaskMarket wallet network.",
            )
        expected = {
            "network": "Base",
            "chainId": BASE_CHAIN_ID,
            "currency": "USDC",
            "usdcContract": USDC_CONTRACT,
        }
        if any(deposit.data.get(key) != value for key, value in expected.items()):
            return _CliResult(
                succeeded=False,
                error="TaskMarket wallet is not configured for Base USDC.",
            )
        stats = self._run_cli(["stats"], False)
        if not stats.succeeded or not isinstance(stats.data, dict):
            return _CliResult(
                succeeded=False,
                error=stats.error or "Could not verify the available USDC balance.",
            )
        try:
            balance = Decimal(str(stats.data["balanceUsdc"]))
        except (KeyError, InvalidOperation, TypeError, ValueError):
            return _CliResult(
                succeeded=False,
                error="TaskMarket CLI returned an unreadable USDC balance.",
            )
        if not balance.is_finite() or balance < maximum_spend:
            return _CliResult(
                succeeded=False,
                error=(
                    "Insufficient USDC balance for the reviewed maximum spend "
                    f"({_format_usdc(maximum_spend)} USDC)."
                ),
            )
        return _CliResult(succeeded=True, data=stats.data)

    def _run_cli(self, args: list[str], is_write: bool) -> _CliResult:
        """Run one first-party CLI command without shell interpolation or retries."""
        if self._cli_runner is not None:
            return self._cli_runner(args, is_write)
        try:
            completed = subprocess.run(
                [self.cli_path, *args],
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            return _CliResult(
                succeeded=False, error="The first-party taskmarket CLI was not found."
            )
        except subprocess.TimeoutExpired:
            return _CliResult(
                succeeded=False,
                error=(
                    "TaskMarket CLI timed out. Inspect live status before retrying; "
                    "the command was not retried."
                ),
                ambiguous=is_write,
            )
        parsed: Any = None
        stdout = completed.stdout.strip()
        if stdout:
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                parsed = None
        if completed.returncode == 0 and isinstance(parsed, dict):
            if parsed.get("ok") is False:
                return _CliResult(
                    succeeded=False,
                    error="TaskMarket CLI rejected the command.",
                    ambiguous=is_write,
                )
            return _CliResult(succeeded=True, data=parsed.get("data", parsed))
        return _CliResult(
            succeeded=False,
            error="TaskMarket CLI rejected the command.",
            ambiguous=is_write,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        body: Any = None,
    ) -> Any:
        if self._transport is not None:
            return self._transport(method, path, params or {}, body)

        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{self.api_url}{path}{query}",
            data=None if body is None else _json_bytes(body),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return _json_loads(response.read())
        except Exception as exc:
            return {"error": f"TaskMarket request failed: {exc}", "retry": True}


def create_taskmarket_tools(
    *,
    api_url: str = DEFAULT_API_URL,
    cli_path: str = DEFAULT_CLI,
    timeout: float = 15.0,
    transport: JsonTransport | None = None,
    clock: Callable[[], datetime] | None = None,
    approval: Callable[[dict[str, Any]], bool] | None = None,
    cli_runner: Callable[[list[str], bool], _CliResult] | None = None,
) -> list[BaseTool]:
    """Create LangChain tools for an explicit TaskMarket requester workflow."""
    client = TaskMarketClient(
        api_url=api_url,
        cli_path=cli_path,
        timeout=timeout,
        transport=transport,
        clock=clock,
        approval=approval,
        cli_runner=cli_runner,
    )
    return [
        StructuredTool.from_function(
            func=client.list_tasks,
            name="taskmarket_list_tasks",
            description=(
                "List live TaskMarket tasks. This read-only operation never spends funds."
            ),
            args_schema=_ListTasksInput,
        ),
        StructuredTool.from_function(
            func=client.get_task,
            name="taskmarket_get_task",
            description="Retrieve live status for a TaskMarket task.",
            args_schema=_TaskIdInput,
        ),
        StructuredTool.from_function(
            func=client.list_submissions,
            name="taskmarket_list_submissions",
            description=(
                "List TaskMarket submissions for human review; never accept or reject work."
            ),
            args_schema=_TaskIdInput,
        ),
        StructuredTool.from_function(
            func=client.preview_task,
            name="taskmarket_preview_task",
            description=(
                "Prepare an exact TaskMarket request for user review; never spends funds."
            ),
            args_schema=_PreviewTaskInput,
        ),
        StructuredTool.from_function(
            func=client.create_task,
            name="taskmarket_create_task",
            description=(
                "Create a TaskMarket task only after a matching preview and configured "
                "human approval; never accept or reject submissions automatically."
            ),
            args_schema=_CreateTaskInput,
        ),
    ]


def _json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _json_loads(value: bytes) -> Any:
    import json

    return json.loads(value.decode("utf-8"))


def _validate_task_id(task_id: str) -> str | None:
    if not isinstance(task_id, str) or not task_id.startswith("0x"):
        return "task_id must be a 0x-prefixed TaskMarket task ID"
    return None


def _format_usdc(value: Decimal) -> str:
    return f"{value.quantize(USDC_QUANTUM):.6f}"


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_datetime(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _maximum_spend(reward: Decimal) -> Decimal:
    fee = reward * PLATFORM_FEE_BPS / Decimal("10000")
    return (reward + fee + RELAY_FEE_USDC).quantize(
        USDC_QUANTUM, rounding=ROUND_CEILING
    )


def _confirmation_digest(request: dict[str, Any]) -> str:
    encoded = json.dumps(
        request, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_id_from_cli_result(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    task_id = data.get("taskId")
    if isinstance(task_id, str) and task_id.startswith("0x"):
        return task_id
    return None
