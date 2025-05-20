import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool as dec_tool, BaseTool
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple, JsonPlusSerializer, ChannelVersions
from langgraph.prebuilt import create_rag_agent, RagState
from tests.messages import _AnyIdHumanMessage, _AnyIdAIMessage


pytestmark = pytest.mark.anyio


class MockInMemoryCheckpointSaver(BaseCheckpointSaver):
    """Minimal in-memory checkpoint saver for testing purposes."""

    def __init__(self, *, serde: Optional[JsonPlusSerializer] = None):
        super().__init__(serde=serde)
        self.checkpoints: Dict[str, CheckpointTuple] = {}

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        return self.checkpoints.get(thread_id)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        self.checkpoints[thread_id] = CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=None # Not strictly needed for these tests
        )
        return config

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)


class FakeRAGModel(FakeChatModel):
    """Fake ChatModel for RAG agent tests."""

    def invoke(
        self, input: Union[str, List[BaseMessage]], config: Optional[dict] = None, **kwargs: Any
    ) -> Union[str, BaseMessage]:
        prompt_str = "".join(msg.content for msg in input) if isinstance(input, list) else input

        if "Assess if the documents are relevant" in prompt_str:
            # Simulate grading
            if "bad_query" in prompt_str.lower() or "no_docs_query" in prompt_str.lower():
                return AIMessage(content="not_relevant")
            return AIMessage(content="relevant")
        elif "rephrasing questions" in prompt_str:
            # Simulate query transformation
            if "original_query_transformed_once" in prompt_str:
                 return AIMessage(content="original_query_transformed_twice")
            return AIMessage(content="original_query_transformed_once")
        elif "User Question:" in prompt_str:
            # Simulate generation
            if "no_docs_for_generation" in prompt_str:
                 return AIMessage(content="Sorry, I found no documents to answer that.")
            if "original_query_transformed_once" in prompt_str:
                 return AIMessage(content="Answer based on transformed_once query.")
            return AIMessage(content="Generated answer based on relevant documents.")
        return AIMessage(content="Default LLM response.")

    async def ainvoke(
        self, input: Union[str, List[BaseMessage]], config: Optional[dict] = None, **kwargs: Any
    ) -> Union[str, BaseMessage]:
        return self.invoke(input, config, **kwargs)


@dec_tool
def mock_retriever_tool(query: str) -> List[Document]:
    """Simulates retrieving documents."""
    if query == "original_query":
        return [
            Document(page_content="Content of doc1 from original_query"),
            Document(page_content="Content of doc2 from original_query"),
        ]
    elif query == "original_query_transformed_once":
        return [
            Document(page_content="Content of docA from transformed_once"),
        ]
    elif query == "bad_query": # for testing not_relevant grading
        return [Document(page_content="Irrelevant content for bad_query")]
    elif query == "no_docs_query": # for testing no documents found pathway
        return []
    return []

@dec_tool
def mock_external_search_tool(query: str) -> List[Document]:
    """Simulates external search."""
    if "external_search_needed_query" in query:
        return [Document(page_content="Document from external search.")]
    return []


# --- Test Cases --- 

def test_rag_agent_basic_retrieval_grading_generation(): 
    llm = FakeRAGModel()
    # For this test, vector_store and embedding_model are not directly used 
    # because retriever_tool is provided.
    mock_checkpointer = MockInMemoryCheckpointSaver()
    agent = create_rag_agent(
        llm=llm,
        embedding_model=None, # Not used if retriever_tool is present
        vector_store=None, # Not used if retriever_tool is present
        retriever_tool=mock_retriever_tool,
        do_document_grading=True,
        do_external_search=False,
        checkpointer=mock_checkpointer, 
    )

    inputs = [HumanMessage(content="original_query")]
    thread = {"configurable": {"thread_id": "rag_test_1"}}

    response = agent.invoke({"messages": inputs}, thread)

    assert len(response["messages"]) == 2
    assert isinstance(response["messages"][0], HumanMessage)
    assert response["messages"][0].content == "original_query"
    assert isinstance(response["messages"][1], AIMessage)
    assert response["messages"][1].content == "Generated answer based on relevant documents."

    # Check RagState in checkpoint
    saved_state_tuple = mock_checkpointer.get_tuple(thread) 
    assert saved_state_tuple is not None
    checkpoint_rag_state: RagState = saved_state_tuple.checkpoint["channel_values"]
    
    assert checkpoint_rag_state["question"] == "original_query"
    assert checkpoint_rag_state["original_question"] == "original_query"
    assert checkpoint_rag_state["document_assessment"] == "relevant"
    assert len(checkpoint_rag_state["documents"]) == 2
    assert checkpoint_rag_state["documents"][0].page_content == "Content of doc1 from original_query"
    assert checkpoint_rag_state["generation"] == "Generated answer based on relevant documents."
    assert checkpoint_rag_state["iterations"] > 0

async def test_rag_agent_basic_retrieval_no_grading_generation(async_checkpointer: BaseCheckpointSaver):
    """
    Tests the RAG agent flow when document grading is disabled.
    1. Input a query ("original_query").
    2. mock_retriever_tool returns relevant documents.
    3. Grading node is skipped.
    4. FakeRAGModel generates an answer based on the retrieved documents.
    5. Verifies the final messages and that document_assessment in RagState is None.
    """
    llm = FakeRAGModel()
    agent = create_rag_agent(
        llm=llm,
        embedding_model=None, 
        vector_store=None, 
        retriever_tool=mock_retriever_tool,
        do_document_grading=False, # Grading disabled
        do_external_search=False,
        checkpointer=async_checkpointer,
    )

    inputs = [HumanMessage(content="original_query")]
    thread = {"configurable": {"thread_id": "rag_test_no_grading_1"}}

    response = await agent.ainvoke({"messages": inputs}, thread)

    assert len(response["messages"]) == 2
    assert isinstance(response["messages"][1], AIMessage)
    assert response["messages"][1].content == "Generated answer based on relevant documents."

    saved_state = await async_checkpointer.aget_tuple(thread)
    assert saved_state is not None
    checkpoint_rag_state: RagState = saved_state.checkpoint["channel_values"]
    
    assert checkpoint_rag_state["question"] == "original_query"
    assert checkpoint_rag_state["document_assessment"] is None # Grading was skipped
    assert len(checkpoint_rag_state["documents"]) == 2
    assert checkpoint_rag_state["generation"] == "Generated answer based on relevant documents."


def test_rag_agent_not_relevant_grading_then_transform(sync_checkpointer: BaseCheckpointSaver):
    """
    Tests the RAG agent flow where:
    1. Initial retrieval provides documents.
    2. These documents are graded as "not_relevant".
    3. External search is DISABLED for this test, forcing query transformation.
    4. The query is transformed by the LLM.
    5. The new (transformed) query is used for a second retrieval attempt.
    6. Documents from the second retrieval are graded as "relevant".
    7. An answer is generated based on the documents from the second retrieval.
    Verifies the final answer and key RagState fields like the transformed question.
    """
    llm = FakeRAGModel()
    agent = create_rag_agent(
        llm=llm,
        embedding_model=None, 
        vector_store=None, 
        retriever_tool=mock_retriever_tool,
        do_document_grading=True,
        do_external_search=False,  # External search disabled to force transformation path
        external_search_tools=None, # Explicitly none
        checkpointer=sync_checkpointer,
    )

    # This query will lead to "not_relevant" grading by FakeRAGModel
    inputs = [HumanMessage(content="bad_query")] 
    thread = {"configurable": {"thread_id": "rag_test_not_relevant_transform_1"}}

    response = agent.invoke({"messages": inputs}, thread)

    # Expected final answer is based on the transformed query
    assert len(response["messages"]) == 2
    assert isinstance(response["messages"][0], HumanMessage)
    assert response["messages"][0].content == "bad_query"
    assert isinstance(response["messages"][1], AIMessage)
    # FakeRAGModel is set to return "Answer based on transformed_once query." after transformation of "bad_query"
    assert response["messages"][1].content == "Answer based on transformed_once query."

    saved_state = sync_checkpointer.get_tuple(thread)
    assert saved_state is not None
    checkpoint_rag_state: RagState = saved_state.checkpoint["channel_values"]

    assert checkpoint_rag_state["original_question"] == "bad_query"
    # FakeRAGModel transforms "bad_query" to "original_query_transformed_once"
    assert checkpoint_rag_state["question"] == "original_query_transformed_once" 
    # The final successful grading after transformation should be "relevant"
    assert checkpoint_rag_state["document_assessment"] == "relevant" 
    # Documents should be from the successful retrieval using "original_query_transformed_once"
    assert len(checkpoint_rag_state["documents"]) == 1 
    assert checkpoint_rag_state["documents"][0].page_content == "Content of docA from transformed_once"
    assert checkpoint_rag_state["generation"] == "Answer based on transformed_once query."
    # Iterations should reflect multiple steps (initial retrieve, grade, transform, retrieve, grade, generate)
    assert checkpoint_rag_state["iterations"] > 1 
    # current_phase_iterations should show at least one transformation attempt
    # (it's reset on successful retrieval, so this checks the state *during* transformation phase if it were to be captured then,
    # or the count on the transform_query node's output. Here we check after the full run)
    # The exact value depends on where current_phase_iterations is incremented and reset.
    # The transform_query_node increments it. retrieve_documents_node resets it.
    # So, if we end after generation, current_phase_iterations might be 0 (reset by the last retrieve_documents_node).
    # Let's verify that the *question was indeed transformed* which implies the path was taken.
    # The 'iterations' count being >1 is also a good indicator of multiple main loop cycles.
