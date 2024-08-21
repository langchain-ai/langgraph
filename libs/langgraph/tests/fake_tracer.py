from typing import Any, Optional
from uuid import UUID

from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.tracers import BaseTracer, Run


class FakeTracer(BaseTracer):
    """Fake tracer that records LangChain execution.
    It replaces run ids with deterministic UUIDs for snapshotting."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: list[Run] = []
        self.uuids_map: dict[UUID, UUID] = {}
        self.uuids_generator = (
            UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
        )

    def _replace_uuid(self, uuid: UUID) -> UUID:
        if uuid not in self.uuids_map:
            self.uuids_map[uuid] = next(self.uuids_generator)
        return self.uuids_map[uuid]

    def _replace_message_id(self, maybe_message: Any) -> Any:
        if isinstance(maybe_message, BaseMessage):
            maybe_message.id = str(next(self.uuids_generator))
        if isinstance(maybe_message, ChatGeneration):
            maybe_message.message.id = str(next(self.uuids_generator))
        if isinstance(maybe_message, LLMResult):
            for i, gen_list in enumerate(maybe_message.generations):
                for j, gen in enumerate(gen_list):
                    maybe_message.generations[i][j] = self._replace_message_id(gen)
        if isinstance(maybe_message, dict):
            for k, v in maybe_message.items():
                maybe_message[k] = self._replace_message_id(v)
        if isinstance(maybe_message, list):
            for i, v in enumerate(maybe_message):
                maybe_message[i] = self._replace_message_id(v)

        return maybe_message

    def _copy_run(self, run: Run) -> Run:
        if run.dotted_order:
            levels = run.dotted_order.split(".")
            processed_levels = []
            for level in levels:
                timestamp, run_id = level.split("Z")
                new_run_id = self._replace_uuid(UUID(run_id))
                processed_level = f"{timestamp}Z{new_run_id}"
                processed_levels.append(processed_level)
            new_dotted_order = ".".join(processed_levels)
        else:
            new_dotted_order = None
        return run.copy(
            update={
                "id": self._replace_uuid(run.id),
                "parent_run_id": (
                    self.uuids_map[run.parent_run_id] if run.parent_run_id else None
                ),
                "child_runs": [self._copy_run(child) for child in run.child_runs],
                "trace_id": self._replace_uuid(run.trace_id) if run.trace_id else None,
                "dotted_order": new_dotted_order,
                "inputs": self._replace_message_id(run.inputs),
                "outputs": self._replace_message_id(run.outputs),
            }
        )

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""

        self.runs.append(self._copy_run(run))

    def flattened_runs(self) -> list[Run]:
        q = [] + self.runs
        result = []
        while q:
            parent = q.pop()
            result.append(parent)
            if parent.child_runs:
                q.extend(parent.child_runs)
        return result

    @property
    def run_ids(self) -> list[Optional[UUID]]:
        runs = self.flattened_runs()
        uuids_map = {v: k for k, v in self.uuids_map.items()}
        return [uuids_map.get(r.id) for r in runs]
