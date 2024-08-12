from langgraph.channels.manager import ChannelsManager
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.managed.base import ManagedValuesManager
from langgraph.pregel.algo import prepare_next_tasks


def test_prepare_next_tasks() -> None:
    config = {}
    processes = {}
    checkpoint = empty_checkpoint()

    with ManagedValuesManager({}, config) as managed, ChannelsManager(
        {}, checkpoint, config
    ) as channels:
        assert (
            prepare_next_tasks(
                checkpoint, processes, channels, managed, config, 0, for_execution=False
            )
            == []
        )
        assert (
            prepare_next_tasks(
                checkpoint, processes, channels, managed, config, 0, for_execution=True
            )
            == []
        )

        # TODO: add more tests
