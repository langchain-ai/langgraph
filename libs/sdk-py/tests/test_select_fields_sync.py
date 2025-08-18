import functools
import json
import os
from pathlib import Path
from typing import get_args

from langgraph_sdk.schema import (
    AssistantSelectField,
    CronSelectField,
    RunSelectField,
    ThreadSelectField,
)

current_dir = os.path.dirname(os.path.abspath(__file__))


@functools.cache
def _load_spec() -> dict:
    with (
        Path(current_dir).parents[2]
        / "docs"
        / "docs"
        / "cloud"
        / "reference"
        / "api"
        / "openapi.json"
    ).open() as f:
        return json.load(f)


def _enum_from_request_select(spec: dict, path: str, method: str) -> set[str]:
    schema = spec["paths"][path][method]["requestBody"]["content"]["application/json"][
        "schema"
    ]
    if "properties" in schema:
        props = schema["properties"]
    elif "$ref" in schema:
        component = spec
        index = schema["$ref"].split("/")[1:]
        for part in index:
            component = component[part]
        props = component["properties"]
    else:
        raise ValueError(f"Unknown schema: {schema}")
    sel = props["select"]
    return set(sel["items"]["enum"])


def _enum_from_query_select(spec: dict, path: str, method: str) -> set[str]:
    params = spec["paths"][path][method]["parameters"]
    sel = next(p for p in params if p["name"] == "select")
    return set(sel["schema"]["items"]["enum"])


def test_assistants_select_enum_matches_sdk():
    spec = _load_spec()
    expected = set(get_args(AssistantSelectField))
    assert _enum_from_request_select(spec, "/assistants/search", "post") == expected


def test_threads_select_enum_matches_sdk():
    spec = _load_spec()
    expected = set(get_args(ThreadSelectField))
    assert _enum_from_request_select(spec, "/threads/search", "post") == expected


def test_runs_select_enum_matches_sdk():
    spec = _load_spec()
    expected = set(get_args(RunSelectField))
    assert _enum_from_query_select(spec, "/threads/{thread_id}/runs", "get") == expected


def test_crons_select_enum_matches_sdk():
    spec = _load_spec()
    expected = set(get_args(CronSelectField))
    assert _enum_from_request_select(spec, "/runs/crons/search", "post") == expected
