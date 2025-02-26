from langchain_core.callbacks import CallbackManagerForLLMRun as CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage as BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver as BaseCheckpointSaver
from langgraph.pregel import Pregel as Pregel

def react_agent(n_tools: int, checkpointer: BaseCheckpointSaver | None) -> Pregel: ...
