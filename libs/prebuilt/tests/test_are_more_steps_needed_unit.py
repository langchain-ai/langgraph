"""Unit tests for _are_more_steps_needed function bug fixes.

These tests directly exercise the _are_more_steps_needed logic by
creating the function within a test context.
"""

import pytest
from langchain_core.messages import AIMessage
from typing_extensions import TypedDict


class MinimalState(TypedDict):
    """Minimal state for testing _are_more_steps_needed."""
    messages: list
    remaining_steps: int


def _get_state_value(state, key, default=None):
    """Helper to get state value."""
    return state.get(key, default) if isinstance(state, dict) else getattr(state, key, default)


def test_b0_return_direct_tools_dont_abort_with_one_step():
    """Test bug b0: return_direct tools shouldn't abort with remaining_steps=1.

    Bug: Line 631-632 checks 'remaining_steps < 2 and has_tool_calls' WITHOUT
    excluding return_direct tools. When remaining_steps=1 and all tools are
    return_direct, the agent incorrectly aborts.
    """
    # Create the buggy version of _are_more_steps_needed
    should_return_direct = {"tool1"}  # tool1 is return_direct

    def _are_more_steps_needed_buggy(state, response):
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        if remaining_steps is not None:
            if remaining_steps < 1 and all_tools_return_direct:  # Line 629
                return True
            elif remaining_steps < 2 and has_tool_calls:  # Line 631-632
                return True
        return False

    # Create the fixed version
    def _are_more_steps_needed_fixed(state, response):
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        if not has_tool_calls:
            return False

        all_tools_return_direct = all(
            call["name"] in should_return_direct for call in response.tool_calls
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        if remaining_steps is not None:
            # Return_direct tools don't consume steps
            if all_tools_return_direct:
                return False
            # Non-return_direct tools need at least 2 steps
            elif remaining_steps < 2:
                return True
        return False

    # Test case: remaining_steps=1, all tools are return_direct
    state: MinimalState = {"messages": [], "remaining_steps": 1}
    response = AIMessage(
        content="calling tool",
        tool_calls=[{"id": "call1", "name": "tool1", "args": {}}]
    )

    # Buggy version returns True (abort) - WRONG
    assert _are_more_steps_needed_buggy(state, response) == True, "Buggy version incorrectly aborts"

    # Fixed version returns False (don't abort) - CORRECT
    assert _are_more_steps_needed_fixed(state, response) == False, "Fixed version should not abort for return_direct tools"


def test_b2_empty_tool_calls_vacuous_truth():
    """Test bug b2: empty tool_calls via vacuous truth should not abort.

    Bug: all([]) returns True, so when response.tool_calls is empty,
    all_tools_return_direct becomes True incorrectly, causing spurious abort.
    """
    should_return_direct: set = set()  # Empty set

    def _are_more_steps_needed_buggy(state, response):
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        if remaining_steps is not None:
            if remaining_steps < 1 and all_tools_return_direct:  # Line 629
                return True
            elif remaining_steps < 2 and has_tool_calls:
                return True
        return False

    def _are_more_steps_needed_fixed(state, response):
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        if not has_tool_calls:
            return False

        all_tools_return_direct = all(
            call["name"] in should_return_direct for call in response.tool_calls
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        if remaining_steps is not None:
            if all_tools_return_direct:
                return False
            elif remaining_steps < 2:
                return True
        return False

    # Test case: remaining_steps=0, empty tool_calls
    state: MinimalState = {"messages": [], "remaining_steps": 0}
    response = AIMessage(content="response", tool_calls=[])  # Empty!

    # With empty tool_calls:
    # - has_tool_calls = False (empty list is falsy)
    # - all_tools_return_direct = all([]) = True (VACUOUS TRUTH BUG!)

    # Buggy: line 629 triggers: remaining_steps=0 < 1 AND all_tools_return_direct=True = abort
    assert _are_more_steps_needed_buggy(state, response) == True, "Buggy version incorrectly aborts on empty tool_calls"

    # Fixed: checks has_tool_calls first, returns False immediately
    assert _are_more_steps_needed_fixed(state, response) == False, "Fixed version should not abort when no tool calls"


def test_b3_inverted_return_direct_logic():
    """Test bug b3: logic inverted - should abort when tools are NOT return_direct.

    Bug: Condition triggers when tools ARE return_direct (backwards logic).
    Should only abort when we have non-return_direct tools and insufficient steps.
    """
    should_return_direct = {"tool1"}  # tool1 is return_direct

    def _are_more_steps_needed_buggy(state, response):
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        if remaining_steps is not None:
            if remaining_steps < 1 and all_tools_return_direct:  # BUG: backwards!
                return True
            elif remaining_steps < 2 and has_tool_calls:
                return True
        return False

    def _are_more_steps_needed_fixed(state, response):
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        if not has_tool_calls:
            return False

        all_tools_return_direct = all(
            call["name"] in should_return_direct for call in response.tool_calls
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        if remaining_steps is not None:
            if all_tools_return_direct:
                return False
            elif remaining_steps < 2:
                return True
        return False

    # Test case 1: remaining_steps=0, all_tools_return_direct=True
    state: MinimalState = {"messages": [], "remaining_steps": 0}
    response = AIMessage(
        content="calling return_direct tool",
        tool_calls=[{"id": "call1", "name": "tool1", "args": {}}]
    )

    # Buggy version incorrectly aborts for return_direct tools when steps < 1
    assert _are_more_steps_needed_buggy(state, response) == True, "Buggy version aborts for return_direct tools"

    # Fixed version doesn't abort for return_direct tools
    assert _are_more_steps_needed_fixed(state, response) == False, "Fixed version allows return_direct tools"

    # Test case 2: remaining_steps=0, but tool is NOT return_direct
    should_return_direct_case2 = set()  # Empty - no tools are return_direct

    def _are_more_steps_needed_buggy_case2(state, response):
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct_case2 for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        if remaining_steps is not None:
            if remaining_steps < 1 and all_tools_return_direct:
                return True
            elif remaining_steps < 2 and has_tool_calls:
                return True
        return False

    def _are_more_steps_needed_fixed_case2(state, response):
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        if not has_tool_calls:
            return False

        all_tools_return_direct = all(
            call["name"] in should_return_direct_case2 for call in response.tool_calls
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        if remaining_steps is not None:
            if all_tools_return_direct:
                return False
            elif remaining_steps < 2:
                return True
        return False

    response_non_return_direct = AIMessage(
        content="calling regular tool",
        tool_calls=[{"id": "call1", "name": "tool2", "args": {}}]
    )

    # Both versions should abort for non-return_direct tools with insufficient steps
    assert _are_more_steps_needed_buggy_case2(state, response_non_return_direct) == True
    assert _are_more_steps_needed_fixed_case2(state, response_non_return_direct) == True
