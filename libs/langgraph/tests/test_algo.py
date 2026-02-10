from langgraph._internal._constants import PULL, PUSH, RESUME
from langgraph.pregel._algo import _scratchpad, prepare_next_tasks, task_path_str
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


def test_scratchpad_root_without_global_resume_value() -> None:
    writes = [("task-1", RESUME, "task-resume")]
    scratchpad = _scratchpad(
        parent_scratchpad=None,
        pending_writes=writes,
        task_id="task-1",
        namespace_hash="ns",
        resume_map=None,
        step=0,
        stop=1,
    )

    assert scratchpad.resume == ["task-resume"]
    assert scratchpad.get_null_resume() is None
    assert scratchpad.get_null_resume(consume=True) is None
