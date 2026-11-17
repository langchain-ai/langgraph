from copy import deepcopy

import pytest

from langgraph_cli.deployment_metrics import compact_deployment_metrics


def reshape(response, deployment_id, **kwargs):
    return compact_deployment_metrics(
        response,
        deployment_id,
        start_time=kwargs.pop("start_time", "2026-10-28T00:00:00Z"),
        end_time=kwargs.pop("end_time", "2026-10-28T01:00:00Z"),
        **kwargs,
    )


def test_compact_metrics(deployment_metrics_response):
    original = deepcopy(deployment_metrics_response)
    result = reshape(deployment_metrics_response, "deployment-id")
    assert deployment_metrics_response == original
    assert result["deployment_id"] == "deployment-id"
    assert result["start_time"] == "2026-10-28T00:00:00Z"
    assert result["end_time"] == "2026-10-28T01:00:00Z"
    assert result["step_seconds"] == 60
    assert result["sections"]["Postgres"] == {"_empty": True}
    queue = result["sections"]["Queue"]
    memory = queue["Memory Usage %"]
    assert memory["Avg Memory Usage %"] == {
        "metric": "memory_usage",
        "unit": "percent",
        "latest": 20.235,
        "min": 10.123,
        "max": 20.235,
        "avg": 15.179,
        "points": [
            ["2026-10-28T00:00:00Z", 10.123],
            ["2026-10-28T00:01:00Z", 20.235],
            ["2026-10-28T00:02:00Z", None],
        ],
    }
    assert memory["Max Memory Usage %"]["max"] == 50.333
    assert queue["Restarts"]["Restarts"]["unit"] == "count"
    assert queue["Queue wait time"]["Queue wait time"] == {
        "metric": "run_queue_wait_time",
        "unit": None,
        "latest": None,
        "min": None,
        "max": None,
        "avg": None,
        "points": [],
    }
    assert result["sections"]["API"]["Latency"]["P95 Latency"]["unit"] is None


def test_summary_and_requested_bounds(deployment_metrics_response):
    result = reshape(
        deployment_metrics_response,
        "id",
        start_time="2026-10-27T23:00:00Z",
        end_time="2026-10-28T01:00:00Z",
        summary=True,
    )
    assert result["start_time"] == "2026-10-27T23:00:00Z"
    assert result["end_time"] == "2026-10-28T01:00:00Z"
    for section in result["sections"].values():
        if section == {"_empty": True}:
            continue
        for chart in section.values():
            for series in chart.values():
                assert set(series) == {"metric", "unit", "latest", "min", "max", "avg"}


@pytest.mark.parametrize("values", [[], [None], [float("nan"), float("inf")]])
def test_empty_and_nonfinite_values(deployment_metrics_response, values):
    chart = deployment_metrics_response["sub_sections"][0]["charts"][0]
    chart["series"] = chart["series"][:1]
    chart["data"] = [
        {
            "series_id": chart["series"][0]["id"],
            "timestamp": f"2026-10-28T00:0{i}:00Z",
            "value": value,
        }
        for i, value in enumerate(values)
    ]
    result = reshape(deployment_metrics_response, "id")
    series = result["sections"]["Queue"]["Memory Usage %"]["Avg Memory Usage %"]
    assert all(series[key] is None for key in ["latest", "min", "max", "avg"])
    assert all(point[1] is None for point in series["points"])


def test_tied_spacing_and_top_level_charts(deployment_metrics_response):
    response = deployment_metrics_response
    chart = response["sub_sections"][1]["charts"][0]
    chart["data"][0]["timestamp"] = "2026-10-28T00:02:30Z"
    response["charts"] = [chart]
    response["sub_sections"] = []
    result = reshape(response, "id")
    assert result["step_seconds"] == 60
    assert "Latency" in result["sections"][response["title"]]


def test_missing_point_preserves_common_spacing(deployment_metrics_response):
    chart = deployment_metrics_response["sub_sections"][0]["charts"][0]
    series_id = chart["series"][0]["id"]
    chart["data"] = [
        point
        for point in chart["data"]
        if not (point["series_id"] == series_id and "00:01:00" in point["timestamp"])
    ]
    result = reshape(deployment_metrics_response, "id")
    assert result["step_seconds"] == 60
    assert type(result["step_seconds"]) is int


def test_only_single_point_and_empty_series(deployment_metrics_response):
    for section in deployment_metrics_response["sub_sections"]:
        for chart in section["charts"]:
            chart["data"] = chart["data"][:1]
    assert reshape(deployment_metrics_response, "id")["step_seconds"] is None


def test_fractional_spacing(deployment_metrics_response):
    chart = deployment_metrics_response["sub_sections"][1]["charts"][0]
    chart["data"][0]["timestamp"] = "2026-10-28T00:00:01Z"
    chart["data"][1]["timestamp"] = "2026-10-28T00:00:00.500Z"
    deployment_metrics_response["sub_sections"] = []
    deployment_metrics_response["charts"] = [chart]
    assert reshape(deployment_metrics_response, "id")["step_seconds"] == 0.5


def test_no_data():
    assert reshape({"charts": [], "sub_sections": []}, "id") == {
        "deployment_id": "id",
        "start_time": "2026-10-28T00:00:00Z",
        "end_time": "2026-10-28T01:00:00Z",
        "step_seconds": None,
        "sections": {"_empty": True},
    }
