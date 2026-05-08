"""Replay state for subgraph checkpoint loading during time-travel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph._internal._constants import NS_END

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple


class ReplayState:
    """Tracks which subgraphs have already loaded their pre-replay checkpoint.

    During a parent replay, each subgraph's first invocation should restore the
    checkpoint from before the replay point. Subsequent invocations of the same
    subgraph (e.g. in a loop) should use normal checkpoint loading so they pick
    up freshly created checkpoints.

    The single `ReplayState` instance is shared by reference across all derived
    configs within one parent execution.
    """

    __slots__ = ("checkpoint_id", "_visited_ns")

    def __init__(self, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id
        # DO NOT CHANGE THIS VARIABLE – it may need to be rehydrated
        # in other runtimes
        self._visited_ns: set[str] = set()

    def _is_first_visit(self, checkpoint_ns: str) -> bool:
        """Return True the first time a subgraph namespace is seen.

        The task-id suffix is stripped so that the same logical subgraph
        (e.g. ``"sub_node"``) is recognized across loop iterations even
        though each iteration has a different task id.
        """
        # "sub_node:task_id" -> "sub_node"
        stable_ns = (
            checkpoint_ns.rsplit(NS_END, 1)[0]
            if NS_END in checkpoint_ns
            else checkpoint_ns
        )
        if stable_ns in self._visited_ns:
            return False
        self._visited_ns.add(stable_ns)
        return True

    def get_checkpoint(
        self,
        checkpoint_ns: str,
        checkpointer: BaseCheckpointSaver,
        checkpoint_config: RunnableConfig,
    ) -> CheckpointTuple | None:
        """Load the right checkpoint for a subgraph during replay.

        On the first call for a given subgraph namespace, returns the latest
        checkpoint created *before* the replay point. On subsequent calls
        (e.g. the same subgraph in a later loop iteration), falls back to
        normal latest-checkpoint loading.
        """
        if self._is_first_visit(checkpoint_ns):
            for saved in checkpointer.list(
                checkpoint_config,
                before={"configurable": {"checkpoint_id": self.checkpoint_id}},
                limit=1,
            ):
                return saved
            return None
        return checkpointer.get_tuple(checkpoint_config)

    async def aget_checkpoint(
        self,
        checkpoint_ns: str,
        checkpointer: BaseCheckpointSaver,
        checkpoint_config: RunnableConfig,
    ) -> CheckpointTuple | None:
        """Async version of `get_checkpoint`."""
        if self._is_first_visit(checkpoint_ns):
            async for saved in checkpointer.alist(
                checkpoint_config,
                before={"configurable": {"checkpoint_id": self.checkpoint_id}},
                limit=1,
            ):
                return saved
            return None
        return await checkpointer.aget_tuple(checkpoint_config)
