"""Compact deployment chart responses for command-line consumers."""

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import fmean
from typing import Any


def _timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return timestamp.replace(tzinfo=timestamp.tzinfo or timezone.utc).astimezone(
        timezone.utc
    )


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _unit(title: str, metric: str | None) -> str | None:
    if "%" in title:
        return "percent"
    if metric in {
        "restart_count",
        "replica_count",
        "worker_count",
        "lg_run_count",
        "run_count",
    }:
        return "count"
    return None


def compact_deployment_metrics(
    response: dict[str, Any],
    deployment_id: str,
    *,
    start_time: str,
    end_time: str,
    summary: bool = False,
) -> dict[str, Any]:
    """Reshape UI charts without mutating the response; step is in seconds."""
    sections = {}
    spacings: Counter[float] = Counter()
    source_sections = list(response.get("sub_sections") or [])
    if response.get("charts"):
        source_sections.insert(0, response)
    for section in source_sections:
        charts = {}
        for chart in section.get("charts") or []:
            by_series = defaultdict(list)
            for point in chart.get("data") or []:
                timestamp = _timestamp(point["timestamp"])
                value = point.get("value")
                if value is not None and not math.isfinite(value):
                    value = None
                by_series[point["series_id"]].append((timestamp, value))
            series_output = {}
            for series in chart.get("series") or []:
                points = sorted(by_series[series["id"]], key=lambda point: point[0])
                times = sorted({timestamp for timestamp, _ in points})
                spacings.update(
                    (b - a).total_seconds()
                    for a, b in zip(times, times[1:], strict=False)
                )
                values = [value for _, value in points if value is not None]
                metric = series.get("project_metric")
                stats = {
                    "metric": metric,
                    "unit": _unit(chart["title"], metric),
                    "latest": round(values[-1], 3) if values else None,
                    "min": round(min(values), 3) if values else None,
                    "max": round(max(values), 3) if values else None,
                    "avg": round(fmean(values), 3) if values else None,
                }
                if not summary:
                    stats["points"] = [
                        [
                            _iso(timestamp),
                            round(value, 3) if value is not None else None,
                        ]
                        for timestamp, value in points
                    ]
                series_output[series["name"]] = stats
            charts[chart["title"]] = series_output or {"_empty": True}
        sections[section["title"]] = charts or {"_empty": True}
    step = min(spacings, key=lambda gap: (-spacings[gap], gap)) if spacings else None
    return {
        "deployment_id": deployment_id,
        "start_time": start_time,
        "end_time": end_time,
        "step_seconds": int(step) if step is not None and step.is_integer() else step,
        "sections": sections or {"_empty": True},
    }
