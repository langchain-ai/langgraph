from langgraph.pregel import StateSnapshot


def assert_state_history_equal(
    actual_state_history: list[StateSnapshot],
    expected_state_history: list[StateSnapshot],
    ignore_parent_config: bool = False,
) -> None:
    assert (
        len(actual_state_history) == len(expected_state_history)
    ), f"Got different lengths for state history: {len(actual_state_history)} for actual, {len(expected_state_history)} for expected"
    for actual, expected in zip(actual_state_history, expected_state_history):
        if ignore_parent_config:
            actual = actual._replace(parent_config=None)
            expected = expected._replace(parent_config=None)

        assert actual == expected
