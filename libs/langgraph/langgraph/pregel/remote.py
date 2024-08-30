from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Sequence, Union

from langchain_core.runnables import RunnableConfig

from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.types import All, StateSnapshot, StreamMode

if TYPE_CHECKING:
    from langgraph_sdk.client import LangGraphClient


class RemotePregel(PregelProtocol):
    def __init__(self, client: LangGraphClient, graph_id: str):
        self.client = client
        self.graph_id = graph_id

    # TODO sync methods

    # TODO aget_subgraphs

    # TODO with_config

    # TODO get_graph

    async def aget_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        # TODO other parts of config missing
        state = await self.client.threads.get_state(
            config["configurable"]["thread_id"],
            config["configurable"].get("checkpoint_id"),
        )
        return StateSnapshot(
            state["values"],
            state["next"],
            {
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "checkpoint_id": state["checkpoint_id"],
                }
            },
            state["metadata"],
            state["created_at"],
            {
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "checkpoint_id": state["parent_checkpoint_id"],
                }
            },
            state["tasks"],
        )

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[StateSnapshot]:
        # TODO other parts of config missing
        for state in await self.client.threads.get_history(
            config["configurable"]["thread_id"],
            limit=limit,
            before=before,
            metadata=filter,
        ):
            yield StateSnapshot(
                state["values"],
                state["next"],
                {
                    "configurable": {
                        "thread_id": config["configurable"]["thread_id"],
                        "checkpoint_id": state["checkpoint_id"],
                    }
                },
                state["metadata"],
                state["created_at"],
                {
                    "configurable": {
                        "thread_id": config["configurable"]["thread_id"],
                        "checkpoint_id": state["parent_checkpoint_id"],
                    }
                },
                state["tasks"],
            )

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: Optional[Union[dict[str, Any], Any]],
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        # TODO fix return value in sdk
        res = await self.client.threads.update_state(
            config["configurable"]["thread_id"],
            values,
            as_node=as_node,
            checkpoint_id=config["configurable"].get("checkpoint_id"),
        )
        return (
            {
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "checkpoint_id": res["checkpoint_id"],
                }
            },
        )

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
        # TODO subgraphs
        async for chunk in self.client.runs.stream(
            config["configurable"]["thread_id"],
            self.graph_id,
            input=input,
            config=config,
            checkpoint_id=config["configurable"].get("checkpoint_id"),
            stream_mode=stream_mode,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        ):
            yield chunk

    async def ainvoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
    ) -> Union[dict[str, Any], Any]:
        return await self.client.runs.wait(
            config["configurable"]["thread_id"],
            self.graph_id,
            input=input,
            config=config,
            checkpoint_id=config["configurable"].get("checkpoint_id"),
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )
