import threading
from types import SimpleNamespace

from langgraph.pregel._loop import PregelLoop


def test_parallel_put_writes_keep_all_writes() -> None:
    """Two tasks finishing at once on worker threads must not drop writes (#8115)."""
    barrier = threading.Barrier(2, timeout=0.5)

    class SlowList(list):
        # Holds both threads inside the read of the old list, the moment the
        # race needs. With the lock in place the second thread never gets here,
        # so the first waits out the timeout and goes on alone.
        def __iter__(self):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            return super().__iter__()

    loop = SimpleNamespace(
        _written_task_ids=set(),
        specs={},
        channels={},
        checkpoint_pending_writes=SlowList(),
        _pending_writes_lock=threading.Lock(),
        durability="exit",
        checkpointer_put_writes=None,
    )

    threads = [
        threading.Thread(
            target=PregelLoop.put_writes, args=(loop, f"t{i}", [("ch", i)])
        )
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(loop.checkpoint_pending_writes) == [("t0", "ch", 0), ("t1", "ch", 1)]
