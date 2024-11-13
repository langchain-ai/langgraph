from langgraph.checkpoint.base import empty_checkpoint
from langgraph.pregel.algo import prepare_next_tasks
from langgraph.pregel.manager import ChannelsManager


def test_prepare_next_tasks() -> None:
    config = {}
    processes = {}
    checkpoint = empty_checkpoint()

    with ChannelsManager({}, checkpoint, config) as (channels, managed):
        assert (
            prepare_next_tasks(
                checkpoint, processes, channels, managed, config, 0, for_execution=False
            )
            == {}
        )
        assert (
            prepare_next_tasks(
                checkpoint,
                processes,
                channels,
                managed,
                config,
                0,
                for_execution=True,
                checkpointer=None,
                store=None,
                manager=None,
            )
            == {}
        )

def test_local_write_validation() -> None:
    from langgraph.constants import TASKS, Send
    from langgraph.errors import InvalidUpdateError
    from langgraph.pregel.algo import local_write
    import pytest
    
    writes = []
    def commit(w):
        writes.extend(w)
        
    process_keys = ["node1", "node2"]
    
    # Test valid Send
    valid_writes = [(TASKS, Send("node1", "test"))]
    local_write(commit, process_keys, valid_writes)
    assert writes == valid_writes
    
    writes.clear()
    
    # Test invalid Send type
    invalid_writes = [(TASKS, "not a Send object")]
    with pytest.raises(InvalidUpdateError, match="Expected Send"):
        local_write(commit, process_keys, invalid_writes)
        
    # Test invalid node name
    invalid_node = [(TASKS, Send("invalid_node", "test"))]
    with pytest.raises(InvalidUpdateError, match="Invalid node name"):
        local_write(commit, process_keys, invalid_node)


        # TODO: add more tests
