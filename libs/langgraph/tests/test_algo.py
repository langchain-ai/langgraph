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
                checkpoint, processes, channels, managed, config, 0, for_execution=True
            )
            == {}
        )

        # TODO: add more tests
