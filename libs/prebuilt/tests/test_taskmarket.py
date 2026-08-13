from datetime import datetime, timezone
from typing import Any

from langgraph.prebuilt import create_taskmarket_tools
from langgraph.prebuilt._taskmarket import BASE_CHAIN_ID, USDC_CONTRACT, _CliResult


def test_taskmarket_tools_list_live_tasks() -> None:
    calls: list[tuple[str, str, dict[str, str], Any]] = []

    def transport(
        method: str,
        path: str,
        params: dict[str, str],
        body: Any,
    ) -> dict[str, Any]:
        calls.append((method, path, params, body))
        return {"tasks": [{"id": "0xabc", "status": "open"}]}

    tools = create_taskmarket_tools(transport=transport)
    list_tasks = next(tool for tool in tools if tool.name == "taskmarket_list_tasks")

    result = list_tasks.invoke({"status": "open", "limit": 1})

    assert result == {"tasks": [{"id": "0xabc", "status": "open"}]}
    assert calls == [("GET", "/api/tasks", {"status": "open", "limit": "1"}, None)]


def test_taskmarket_tools_read_task_and_submissions() -> None:
    def transport(
        method: str,
        path: str,
        params: dict[str, str],
        body: Any,
    ) -> dict[str, Any]:
        assert method == "GET"
        assert params == {}
        assert body is None
        if path.endswith("/submissions"):
            return {"submissions": [{"id": "submission-1"}]}
        return {"id": "0xabc", "status": "open"}

    tools = create_taskmarket_tools(transport=transport)
    get_task = next(tool for tool in tools if tool.name == "taskmarket_get_task")
    list_submissions = next(
        tool for tool in tools if tool.name == "taskmarket_list_submissions"
    )

    assert get_task.invoke({"task_id": "0xabc"}) == {
        "id": "0xabc",
        "status": "open",
    }
    assert list_submissions.invoke({"task_id": "0xabc"}) == {
        "submissions": [{"id": "submission-1"}]
    }


def test_taskmarket_tools_preview_is_read_only_and_reviewable() -> None:
    now = datetime(2026, 8, 13, 20, 0, tzinfo=timezone.utc)
    tools = create_taskmarket_tools(clock=lambda: now)
    preview = next(tool for tool in tools if tool.name == "taskmarket_preview_task")

    result = preview.invoke(
        {
            "description": "Collect and summarize three public sources.",
            "reward_usdc": "0.500000",
            "duration_hours": 2,
            "mode": "bounty",
            "tags": ["research", "public"],
        }
    )

    assert result["deadline"] == "2026-08-13T22:00:00Z"
    assert result["maximumSpendUsdc"] == "0.538500"
    assert result["network"] == "Base"
    assert result["chainId"] == 8453
    assert result["currency"] == "USDC"
    assert result["tags"] == ["research", "public"]
    assert len(result["confirmationToken"]) == 64


def test_taskmarket_create_requires_explicit_confirmation() -> None:
    tools = create_taskmarket_tools()
    create_task = next(tool for tool in tools if tool.name == "taskmarket_create_task")

    result = create_task.invoke(
        {
            "description": "Do the reviewed work.",
            "reward_usdc": "0.500000",
            "duration_hours": 2,
            "confirmation_token": "not-a-preview",
            "confirm": False,
        }
    )

    assert result == {
        "error": (
            "Creation is confirmation-gated. Review taskmarket_preview_task first, "
            "then call taskmarket_create_task with confirm=true."
        ),
        "retry": False,
    }


def test_taskmarket_create_checks_wallet_then_runs_cli_once() -> None:
    now = datetime(2026, 8, 13, 20, 0, tzinfo=timezone.utc)
    calls: list[tuple[list[str], bool]] = []

    def cli_runner(args: list[str], is_write: bool) -> _CliResult:
        calls.append((args, is_write))
        if args == ["deposit"]:
            return _CliResult(
                succeeded=True,
                data={
                    "network": "Base",
                    "chainId": BASE_CHAIN_ID,
                    "currency": "USDC",
                    "usdcContract": USDC_CONTRACT,
                },
            )
        if args == ["stats"]:
            return _CliResult(succeeded=True, data={"balanceUsdc": "1.000000"})
        return _CliResult(succeeded=True, data={"taskId": "0xcreated"})

    tools = create_taskmarket_tools(
        clock=lambda: now,
        approval=lambda preview: preview["rewardUsdc"] == "0.500000",
        cli_runner=cli_runner,
    )
    preview_tool = next(
        tool for tool in tools if tool.name == "taskmarket_preview_task"
    )
    create_tool = next(tool for tool in tools if tool.name == "taskmarket_create_task")
    preview = preview_tool.invoke(
        {
            "description": "Do the reviewed work.",
            "reward_usdc": "0.500000",
            "duration_hours": 2,
        }
    )

    result = create_tool.invoke(
        {
            "description": "Do the reviewed work.",
            "reward_usdc": "0.500000",
            "duration_hours": 2,
            "confirmation_token": preview["confirmationToken"],
            "confirm": True,
        }
    )

    assert result == {
        "taskId": "0xcreated",
        "taskUrl": "https://api.taskmarket.dev/api/tasks/0xcreated",
        "status": "created",
        "retry": False,
    }
    assert calls[0] == (["deposit"], False)
    assert calls[1] == (["stats"], False)
    assert calls[2][0][:2] == ["task", "create"]
    assert calls[2][1] is True


def test_taskmarket_create_does_not_call_wallet_when_approval_denied() -> None:
    now = datetime(2026, 8, 13, 20, 0, tzinfo=timezone.utc)
    calls: list[tuple[list[str], bool]] = []

    def cli_runner(args: list[str], is_write: bool) -> _CliResult:
        calls.append((args, is_write))
        return _CliResult(succeeded=True, data={"balanceUsdc": "1.000000"})

    tools = create_taskmarket_tools(
        clock=lambda: now,
        approval=lambda preview: False,
        cli_runner=cli_runner,
    )
    preview_tool = next(
        tool for tool in tools if tool.name == "taskmarket_preview_task"
    )
    create_tool = next(tool for tool in tools if tool.name == "taskmarket_create_task")
    preview = preview_tool.invoke(
        {
            "description": "Do the reviewed work.",
            "reward_usdc": "0.500000",
            "duration_hours": 2,
        }
    )

    result = create_tool.invoke(
        {
            "description": "Do the reviewed work.",
            "reward_usdc": "0.500000",
            "duration_hours": 2,
            "confirmation_token": preview["confirmationToken"],
            "confirm": True,
        }
    )

    assert result["status"] == "denied"
    assert calls == []


def test_taskmarket_create_reports_unknown_after_cli_timeout_without_retry() -> None:
    now = datetime(2026, 8, 13, 20, 0, tzinfo=timezone.utc)
    calls: list[tuple[list[str], bool]] = []

    def cli_runner(args: list[str], is_write: bool) -> _CliResult:
        calls.append((args, is_write))
        if args == ["deposit"]:
            return _CliResult(
                succeeded=True,
                data={
                    "network": "Base",
                    "chainId": BASE_CHAIN_ID,
                    "currency": "USDC",
                    "usdcContract": USDC_CONTRACT,
                },
            )
        if args == ["stats"]:
            return _CliResult(succeeded=True, data={"balanceUsdc": "1.000000"})
        return _CliResult(
            succeeded=False,
            error="timeout",
            ambiguous=True,
        )

    tools = create_taskmarket_tools(
        clock=lambda: now,
        approval=lambda preview: True,
        cli_runner=cli_runner,
    )
    preview_tool = next(
        tool for tool in tools if tool.name == "taskmarket_preview_task"
    )
    create_tool = next(tool for tool in tools if tool.name == "taskmarket_create_task")
    preview = preview_tool.invoke(
        {
            "description": "Do the reviewed work.",
            "reward_usdc": "0.500000",
            "duration_hours": 2,
        }
    )

    result = create_tool.invoke(
        {
            "description": "Do the reviewed work.",
            "reward_usdc": "0.500000",
            "duration_hours": 2,
            "confirmation_token": preview["confirmationToken"],
            "confirm": True,
        }
    )

    assert result["status"] == "unknown"
    assert result["retry"] is False
    assert calls[-1][0][:2] == ["task", "create"]
    assert len(calls) == 3


def test_taskmarket_tools_reject_invalid_task_ids_before_network_call() -> None:
    calls: list[tuple[str, str, dict[str, str], Any]] = []

    def transport(
        method: str,
        path: str,
        params: dict[str, str],
        body: Any,
    ) -> dict[str, Any]:
        calls.append((method, path, params, body))
        return {}

    tools = create_taskmarket_tools(transport=transport)
    get_task = next(tool for tool in tools if tool.name == "taskmarket_get_task")

    result = get_task.invoke({"task_id": "not-a-task-id"})

    assert result == {
        "error": "task_id must be a 0x-prefixed TaskMarket task ID",
        "retry": False,
    }
    assert calls == []
