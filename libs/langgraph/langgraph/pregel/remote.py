from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Self,
    Sequence,
    Union,
)

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import Graph as DrawableGraph

from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.types import All, StateSnapshot, StreamMode

if TYPE_CHECKING:
    from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
    from langgraph_sdk.schema import Checkpoint, ThreadState


class RemotePregel(PregelProtocol):
    def __init__(
        self, client: LangGraphClient, sync_client: SyncLangGraphClient, graph_id: str
    ):
        self.client = client
        self.sync_client = sync_client
        self.graph_id = graph_id

    def copy(self, update: dict[str, Any]) -> Self:
        attrs = {**self.__dict__, **update}
        return self.__class__(**attrs)

    def with_config(
        self, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Self:
        return self.copy({})

    def get_graph(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        xray: Union[int, bool] = False,
    ) -> DrawableGraph:
        graph = self.sync_client.assistants.get_graph(
            assistant_id=self.graph_id,
            xray=xray,
        )

        return DrawableGraph(
            nodes=graph["nodes"],
            edges=graph["edges"],
        )

    async def aget_graph(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        xray: Union[int, bool] = False,
    ) -> DrawableGraph:
        graph = await self.client.assistants.get_graph(
            assistant_id=self.graph_id,
            xray=xray,
        )
        return DrawableGraph(
            nodes=graph["nodes"],
            edges=graph["edges"],
        )

    def get_subgraphs(
        self, namespace: Optional[str] = None, recurse: bool = False
    ) -> Iterator[tuple[str, "PregelProtocol"]]:
        subgraphs = self.sync_client.assistants.get_subgraphs(
            assistant_id=self.graph_id,
            namespace=namespace,
            recurse=recurse,
        )
        for namespace, graph_schema in subgraphs.items():
            remote_subgraph = self.copy({"graph_id": graph_schema.graph_id})
            yield (namespace, remote_subgraph)

    async def aget_subgraphs(
        self, namespace: Optional[str] = None, recurse: bool = False
    ) -> AsyncIterator[tuple[str, "PregelProtocol"]]:
        subgraphs = await self.client.assistants.get_subgraphs(
            assistant_id=self.graph_id,
            namespace=namespace,
            recurse=recurse,
        )
        for namespace, graph_schema in subgraphs.items():
            remote_subgraph = self.copy({"graph_id": graph_schema.graph_id})
            yield (namespace, remote_subgraph)

    def _create_state_snapshot(self, state: ThreadState) -> StateSnapshot:
        return StateSnapshot(
            values=state["values"],
            next=state["next"],
            config={
                "configurable": {
                    "thread_id": state["checkpoint"]["thread_id"],
                    "checkpoint_ns": state["checkpoint"]["checkpoint_ns"],
                    "checkpoint_id": state["checkpoint"]["checkpoint_id"],
                    "checkpoint_map": state["checkpoint"]["checkpoint_map"],
                }
            },
            metadata=state["metadata"],
            created_at=state["created_at"],
            parent_config={
                "configurable": {
                    "thread_id": state["parent_checkpoint"]["thread_id"],
                    "checkpoint_ns": state["parent_checkpoint"]["checkpoint_ns"],
                    "checkpoint_id": state["parent_checkpoint"]["checkpoint_id"],
                    "checkpoint_map": state["parent_checkpoint"]["checkpoint_map"],
                }
            }
            if state["parent_checkpoint"]
            else None,
            tasks=state["tasks"],
        )

    def _get_checkpoint(self, config: Optional[RunnableConfig]) -> Optional[Checkpoint]:
        if config is None:
            return None

        checkpoint = {}

        if "thread_id" in config["configurable"]:
            checkpoint["thread_id"] = config["configurable"]["thread_id"]
        if "checkpoint_ns" in config["configurable"]:
            checkpoint["checkpoint_ns"] = config["configurable"]["checkpoint_ns"]
        if "checkpoint_id" in config["configurable"]:
            checkpoint["checkpoint_id"] = config["configurable"]["checkpoint_id"]
        if "checkpoint_map" in config["configurable"]:
            checkpoint["checkpoint_map"] = config["configurable"]["checkpoint_map"]

        return checkpoint if checkpoint else None

    def _get_config(self, checkpoint: Checkpoint) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": checkpoint["thread_id"],
                "checkpoint_ns": checkpoint["checkpoint_ns"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint_map": checkpoint["checkpoint_map"],
            }
        }

    def get_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        state = self.sync_client.threads.get_state(
            thread_id=config["configurable"]["thread_id"],
            checkpoint=self._get_checkpoint(config),
            subgraphs=subgraphs,
        )
        return self._create_state_snapshot(state)

    async def aget_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        state = await self.client.threads.get_state(
            thread_id=config["configurable"]["thread_id"],
            checkpoint=self._get_checkpoint(config),
            subgraphs=subgraphs,
        )
        return self._create_state_snapshot(state)

    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[StateSnapshot]:
        states = self.sync_client.threads.get_history(
            thread_id=config["configurable"]["thread_id"],
            limit=limit if limit else 10,
            before=self._get_checkpoint(before),
            metadata=filter,
            checkpoint=self._get_checkpoint(config),
        )
        for state in states:
            yield self._create_state_snapshot(state)

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[StateSnapshot]:
        states = await self.client.threads.get_history(
            thread_id=config["configurable"]["thread_id"],
            limit=limit if limit else 10,
            before=self._get_checkpoint(before),
            metadata=filter,
            checkpoint=self._get_checkpoint(config),
        )
        for state in states:
            yield self._create_state_snapshot(state)

    def update_state(
        self,
        config: RunnableConfig,
        values: Optional[Union[dict[str, Any], Any]],
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        response = self.sync_client.threads.update_state(
            thread_id=config["configurable"]["thread_id"],
            values=values,
            as_node=as_node,
            checkpoint=self._get_checkpoint(config),
        )
        return self._get_config(response["checkpoint"])

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: Optional[Union[dict[str, Any], Any]],
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        response = await self.client.threads.update_state(
            thread_id=config["configurable"]["thread_id"],
            values=values,
            as_node=as_node,
            checkpoint=self._get_checkpoint(config),
        )
        return self._get_config(response["checkpoint"])

    def stream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        subgraphs: bool = False,
    ) -> Iterator[Union[dict[str, Any], Any]]:
        for chunk in self.sync_client.runs.stream(
            thread_id=config["configurable"]["thread_id"],
            assistant_id=self.graph_id,
            input=input,
            stream_mode=stream_mode,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            stream_subgraphs=subgraphs,
        ):
            yield chunk

    async def astream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        subgraphs: bool = False,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
        async for chunk in self.client.runs.stream(
            thread_id=config["configurable"]["thread_id"],
            assistant_id=self.graph_id,
            input=input,
            stream_mode=stream_mode,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            stream_subgraphs=subgraphs,
        ):
            yield chunk

    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
    ) -> Union[dict[str, Any], Any]:
        return self.sync_client.runs.wait(
            thread_id=config["configurable"]["thread_id"],
            assistant_id=self.graph_id,
            input=input,
            config=config,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )

    async def ainvoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
    ) -> Union[dict[str, Any], Any]:
        return await self.client.runs.wait(
            thread_id=config["configurable"]["thread_id"],
            assistant_id=self.graph_id,
            input=input,
            config=config,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )
