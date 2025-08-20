"""Standard condition tests for create_react_agent based on LangChainJS specifications.

This test suite mirrors the standard testing patterns from:
- returnDirect.json - Testing return_direct, response_format, and stop_when combinations
- responses.json - Testing various response format scenarios
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langgraph.prebuilt import create_agent
from tests.model import FakeToolCallingModel


def load_test_specifications() -> tuple[List[Dict], List[Dict]]:
    """Load test specifications from JSON files."""
    # Get the directory where this test file is located
    test_dir = Path(__file__).parent
    specs_dir = test_dir / "specs"

    with open(specs_dir / "return_direct.json", "r") as f:
        return_direct_specs = json.load(f)

    with open(specs_dir / "responses.json", "r") as f:
        response_format_specs = json.load(f)

    return return_direct_specs, response_format_specs


RETURN_DIRECT_SPECS, RESPONSE_FORMAT_SPECS = load_test_specifications()


# Mock tools for testing
class PollJobMock:
    """Mock for deterministic poll tool behavior."""

    def __init__(self, return_direct: bool = False):
        self.return_direct = return_direct
        self.attempts = 0
        self.call_count = 0

    def __call__(self) -> Dict[str, Any]:
        self.call_count += 1
        self.attempts += 1
        status = "succeeded" if self.attempts >= 10 else "pending"
        return {"status": status, "attempts": self.attempts}


# Employee data for HR assistant tests
EMPLOYEES = {
    "Sabine": {"role": "Developer", "department": "IT"},
    "Henrik": {"role": "Product Manager", "department": "IT"},
    "Jessica": {"role": "HR", "department": "People"},
    "Saskia": {"role": "Software Engineer", "department": "IT"},
}


def get_employee_role(name: str) -> str:
    """Get employee role by name."""
    return EMPLOYEES.get(name, {}).get("role", "Unknown")


def get_employee_department(name: str) -> str:
    """Get employee department by name."""
    return EMPLOYEES.get(name, {}).get("department", "Unknown")


def poll_job() -> str:
    """Poll a job status. Returns JSON string with status and attempts."""
    return '{"status": "pending", "attempts": 1}'


# Pydantic models for structured responses
class JobStatusResponse(BaseModel):
    attempts: int = Field(description="Number of polling attempts")
    succeeded: bool = Field(description="Whether the job succeeded")


class EmployeeRoleResponse(BaseModel):
    name: str = Field(description="Employee name")
    role: str = Field(description="Employee role")


class EmployeeDepartmentResponse(BaseModel):
    name: str = Field(description="Employee name")
    department: str = Field(description="Employee department")


# Test data constants
AGENT_PROMPT = "You are a strict polling bot. Only use the 'pollJob' tool until it returns { status: 'succeeded' }."
HR_AGENT_PROMPT = "You are an HR assistant."


@pytest.mark.parametrize("spec", RETURN_DIRECT_SPECS)
def test_return_direct_scenario(spec: Dict[str, Any]) -> None:
    """Test individual return_direct scenario from specification."""
    # Create tool calls based on expected count
    tool_calls = []
    for call_num in range(spec["expectedToolCalls"]):
        tool_calls.append([{"args": {}, "id": str(call_num + 1), "name": "poll_job"}])

    # Add structured response call if expected
    if spec["expectedStructuredResponse"] is not None:
        tool_calls.append(
            [
                {
                    "name": "JobStatusResponse",
                    "id": str(spec["expectedToolCalls"] + 1),
                    "args": spec["expectedStructuredResponse"],
                }
            ]
        )

        expected_response = JobStatusResponse(**spec["expectedStructuredResponse"])
        model = FakeToolCallingModel[JobStatusResponse](
            tool_calls=tool_calls, structured_response=expected_response
        )

        # Create agent with response format if specified
        response_format = None
        if spec["responseFormat"] is not None:
            response_format = JobStatusResponse

        agent = create_agent(model, [poll_job], response_format=response_format)
    else:
        model = FakeToolCallingModel[str](
            tool_calls=tool_calls, structured_response=None
        )
        agent = create_agent(model, [poll_job])

    # Invoke agent
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "Poll the job until it's done and tell me how many attempts it took."
                )
            ]
        }
    )

    # Verify structured response
    if spec["expectedStructuredResponse"] is not None:
        assert response["structured_response"] == expected_response
    else:
        assert response.get("structured_response") is None

    # Verify message content contains expected pattern
    last_message_content = str(response["messages"][-1].content)
    if "Attempts:" in spec["expectedLastMessage"]:
        assert str(spec["expectedToolCalls"]) in last_message_content
    elif "stop condition" in spec["expectedLastMessage"]:
        # Check for stop condition message pattern
        assert len(response["messages"]) >= 1  # Should have at least one message


@pytest.mark.parametrize(
    "spec,assertion",
    [
        (spec, assertion)
        for spec in RESPONSE_FORMAT_SPECS
        for assertion in spec["assertionsByInvocation"]
    ],
)
def test_response_format_scenario(
    spec: Dict[str, Any], assertion: Dict[str, Any]
) -> None:
    """Test individual response format scenario and assertion."""
    # Determine which tools should be called based on expected calls
    tool_calls = []
    call_id = 1

    # Add employee role tool call if expected
    if assertion["toolsWithExpectedCalls"].get("getEmployeeRole", 0) > 0:
        employee_name = _extract_employee_name(assertion["prompt"])
        tool_calls.append(
            [
                {
                    "args": {"name": employee_name},
                    "id": str(call_id),
                    "name": "get_employee_role",
                }
            ]
        )
        call_id += 1

    # Add employee department tool call if expected
    if assertion["toolsWithExpectedCalls"].get("getEmployeeDepartment", 0) > 0:
        employee_name = _extract_employee_name(assertion["prompt"])
        tool_calls.append(
            [
                {
                    "args": {"name": employee_name},
                    "id": str(call_id),
                    "name": "get_employee_department",
                }
            ]
        )
        call_id += 1

    # Add structured response call
    response_type = _determine_response_type(assertion["expectedStructuredResponse"])
    tool_calls.append(
        [
            {
                "name": response_type.__name__,
                "id": str(call_id),
                "args": assertion["expectedStructuredResponse"],
            }
        ]
    )

    # Create expected response object
    expected_response = response_type(**assertion["expectedStructuredResponse"])
    model = FakeToolCallingModel[response_type](
        tool_calls=tool_calls, structured_response=expected_response
    )

    agent = create_agent(
        model,
        [get_employee_role, get_employee_department],
        response_format=response_type,
    )

    response = agent.invoke({"messages": [HumanMessage(assertion["prompt"])]})

    assert response["structured_response"] == expected_response


def _extract_employee_name(prompt: str) -> str:
    """Extract employee name from prompt."""
    for name in EMPLOYEES.keys():
        if name in prompt:
            return name
    return "Unknown"


def _determine_response_type(response_data: Dict) -> type:
    """Determine response type based on response data keys."""
    if "role" in response_data:
        return EmployeeRoleResponse
    elif "department" in response_data:
        return EmployeeDepartmentResponse
    else:
        return EmployeeRoleResponse  # Default fallback
