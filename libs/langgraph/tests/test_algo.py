from langgraph._internal._constants import PULL, PUSH
from langgraph.pregel._algo import (
    _filter_cache_input,
    prepare_next_tasks,
    task_path_str,
)
from langgraph.pregel._checkpoint import channels_from_checkpoint, empty_checkpoint


def test_prepare_next_tasks() -> None:
    config = {}
    processes = {}
    checkpoint = empty_checkpoint()
    channels, managed = channels_from_checkpoint({}, checkpoint)

    assert (
        prepare_next_tasks(
            checkpoint,
            {},
            processes,
            channels,
            managed,
            config,
            0,
            -1,
            for_execution=False,
        )
        == {}
    )
    assert (
        prepare_next_tasks(
            checkpoint,
            {},
            processes,
            channels,
            managed,
            config,
            0,
            -1,
            for_execution=True,
            checkpointer=None,
            store=None,
            manager=None,
        )
        == {}
    )

    # TODO: add more tests


def test_tuple_str() -> None:
    push_path_a = (PUSH, 2)
    pull_path_a = (PULL, "abc")
    push_path_b = (PUSH, push_path_a, 1)
    push_path_c = (PUSH, push_path_b, 3)

    assert task_path_str(push_path_a) == f"~{PUSH}, 0000000002"
    assert task_path_str(push_path_b) == f"~{PUSH}, ~{PUSH}, 0000000002, 0000000001"
    assert (
        task_path_str(push_path_c)
        == f"~{PUSH}, ~{PUSH}, ~{PUSH}, 0000000002, 0000000001, 0000000003"
    )
    assert task_path_str(pull_path_a) == f"~{PULL}, abc"

    path_list = [push_path_b, push_path_a, pull_path_a, push_path_c]
    assert sorted(map(task_path_str, path_list)) == [
        f"~{PULL}, abc",
        f"~{PUSH}, 0000000002",
        f"~{PUSH}, ~{PUSH}, 0000000002, 0000000001",
        f"~{PUSH}, ~{PUSH}, ~{PUSH}, 0000000002, 0000000001, 0000000003",
    ]


def test_filter_cache_input() -> None:
    """Test the _filter_cache_input function for cache key filtering."""
    from langgraph.pregel._checkpoint import empty_checkpoint
    
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"some": "values"}  # Simulate non-empty checkpoint
    
    # Test 1: Filter out result fields for non-result nodes
    input_state = {
        "input": "test_data",
        "result": "processed_data",
        "output": "some_output",
        "data": "keep_this"
    }
    
    filtered = _filter_cache_input(input_state, "processor_node", checkpoint)
    
    # Should keep input and data, but filter out result and output
    assert "input" in filtered
    assert "data" in filtered
    assert "result" not in filtered
    assert "output" not in filtered
    
    # Test 2: Keep result fields for result-related nodes
    result_node_filtered = _filter_cache_input(input_state, "result_generator", checkpoint)
    
    # Should keep result field since node name contains "result"
    assert "input" in result_node_filtered
    assert "data" in result_node_filtered
    assert "result" in result_node_filtered  # Kept because node handles results
    assert "output" not in result_node_filtered
    
    # Test 3: Keep common state fields
    common_state = {
        "input": "test",
        "query": "search",
        "text": "content",
        "message": "hello",
        "state": "active",
        "result": "processed"
    }
    
    filtered_common = _filter_cache_input(common_state, "simple_node", checkpoint)
    
    # Should keep all common fields except result
    assert "input" in filtered_common
    assert "query" in filtered_common
    assert "text" in filtered_common
    assert "message" in filtered_common
    assert "state" in filtered_common
    assert "result" not in filtered_common
    
    # Test 4: Non-dict input should be returned as-is
    non_dict_input = "simple_string"
    assert _filter_cache_input(non_dict_input, "node", checkpoint) == non_dict_input
    
    # Test 5: Empty checkpoint should return original
    empty_cp = empty_checkpoint()
    assert _filter_cache_input(input_state, "node", empty_cp) == input_state
    
    # Test 6: If all fields are filtered out, return original
    all_output_state = {
        "result": "data",
        "output": "data", 
        "response": "data",
        "answer": "data",
        "generated": "data"
    }
    
    filtered_all = _filter_cache_input(all_output_state, "simple_node", checkpoint)
    
    # Should return original since all fields would be filtered
    assert filtered_all == all_output_state
