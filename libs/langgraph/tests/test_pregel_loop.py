import warnings

from langgraph.pregel._loop import SyncPregelLoop
from langgraph.pregel.types import CheckpointTuple, empty_checkpoint


class DummyCheckpointerNoStep:
    """Fake checkpointer that omits 'step' in metadata."""

    def get_tuple(self, checkpoint_config):
        return CheckpointTuple(
            config=checkpoint_config,
            checkpoint=empty_checkpoint(),
            metadata={},  # ⚠️ Missing 'step'
            parent_config=None,
            pending_writes=[],
        )


class DummyCheckpointerWithStep:
    """Fake checkpointer that includes a 'step' in metadata."""

    def get_tuple(self, checkpoint_config):
        return CheckpointTuple(
            config=checkpoint_config,
            checkpoint=empty_checkpoint(),
            metadata={"step": 5},  # ✅ Has step
            parent_config=None,
            pending_writes=[],
        )


def make_loop(checkpointer):
    return SyncPregelLoop(
        input=None,
        stream=None,
        config={"recursion_limit": 1},
        store=None,
        cache=None,
        checkpointer=checkpointer,
        nodes={},
        specs={},
        trigger_to_nodes={},
        durability="ephemeral",
    )


def test_enter_defaults_step_and_warns():
    loop = make_loop(DummyCheckpointerNoStep())

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        with loop as i:
            # ✅ It should not raise KeyError
            assert isinstance(i, SyncPregelLoop)

            # ✅ Step should default to 1 (0 + 1)
            assert i.step == 1

        # ✅ Warning should be raised
        assert any("Checkpoint metadata missing 'step'" in str(wi.message) for wi in w)


def test_enter_resumes_from_existing_step():
    loop = make_loop(DummyCheckpointerWithStep())

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        with loop as j:
            # ✅ Should resume correctly from step 5 → +1
            assert j.step == 6

        # ✅ No warning expected in this case
        assert not any("Checkpoint metadata missing 'step'" in str(wi.message) for wi in w)
