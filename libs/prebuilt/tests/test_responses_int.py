from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, create_model

from langgraph.prebuilt import create_agent
from langgraph.prebuilt.responses import ToolOutput

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    skip_openai_integration_tests = True
else:
    skip_openai_integration_tests = False


def _load_spec() -> list[dict[str, Any]]:
    with (Path(__file__).parent / "specifications" / "responses.json").open(
        "r", encoding="utf-8"
    ) as f:
        return json.load(f)


TEST_CASES = _load_spec()

AGENT_PROMPT = "You are an HR assistant."

EMPLOYEES = [
    {"name": "Sabine", "role": "Developer", "department": "IT"},
    {"name": "Henrik", "role": "Product Manager", "department": "IT"},
    {"name": "Jessica", "role": "HR", "department": "People"},
]


def _make_tool(fn, *, name: str, description: str):
    mock = MagicMock(side_effect=lambda *, name: fn(name=name))
    InputModel = create_model(f"{name}_input", name=(str, ...))

    @tool(name, description=description, args_schema=InputModel)
    def _wrapped(name: str):
        return mock(name=name)

    return {"tool": _wrapped, "mock": mock}


def _build_tool_output_response_format(
    response_format_spec: Sequence[dict[str, Any]],
) -> ToolOutput:
    models: list[type[BaseModel]] = []
    keyset_to_tool_name: dict[frozenset[str], str] = {}
    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    for idx, schema in enumerate(response_format_spec):
        properties = schema["properties"]
        required = set(schema["required"])
        type_name = schema.get("title") or f"structured_output_format_{idx + 1}"
        fields = {}
        for k, prop in properties.items():
            py_type = type_map.get(prop.get("type"), Any)
            fields[k] = (py_type, ...) if k in required else (Optional[py_type], None)  # noqa: UP045
        model = create_model(type_name, **fields)
        models.append(model)
        keyset_to_tool_name[frozenset(required)] = type_name

    union_type = Union[tuple(models)]  # noqa: UP045, UP007
    return ToolOutput(union_type)


@pytest.mark.skipif(
    skip_openai_integration_tests, reason="OpenAI integration tests are disabled."
)
@pytest.mark.xfail(
    reason="currently failing due to undefined behavior for multiple structured responses."
)
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_responses_integration_matrix(case: dict[str, Any]) -> None:
    def get_employee_role(*, name: str) -> str | None:
        for e in EMPLOYEES:
            if e["name"] == name:
                return e["role"]
        return None

    def get_employee_department(*, name: str) -> str | None:
        for e in EMPLOYEES:
            if e["name"] == name:
                return e["department"]
        return None

    role_tool = _make_tool(
        get_employee_role,
        name="getEmployeeRole",
        description="Get the employee role by name",
    )
    dept_tool = _make_tool(
        get_employee_department,
        name="getEmployeeDepartment",
        description="Get the employee department by name",
    )

    response_spec = case["responseFormat"]
    if isinstance(response_spec, dict):
        response_spec = [response_spec]
    tool_output = _build_tool_output_response_format(response_spec)

    for assertion in case["assertionsByInvocation"]:
        prompt: str = assertion["prompt"]
        expected_calls: dict[str, int] = assertion["toolsWithExpectedCalls"]
        expected_structured = assertion.get("expectedStructuredResponse")
        expected_last_message = assertion.get("expectedLastMessage")

        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        agent = create_agent(
            model,
            tools=[role_tool["tool"], dept_tool["tool"]],
            prompt=AGENT_PROMPT,
            response_format=tool_output,
        )
        result = agent.invoke({"messages": [HumanMessage(prompt)]})

        # TODO: Count LLM calls. JS handles with mock fetch. Could pass in mock http_client?

        # Count tool calls
        assert role_tool["mock"].call_count == expected_calls["getEmployeeRole"]
        assert dept_tool["mock"].call_count == expected_calls["getEmployeeDepartment"]

        # Check last message content
        last_message = result["messages"][-1]
        assert last_message.content == expected_last_message

        # Check structured response
        structured_response_json = result["structured_response"].model_dump()
        assert structured_response_json == expected_structured

        print("Passed test for: ", case["name"])
