from typing import Any, Callable, Optional, Sequence, Union, Annotated

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.store.base import Checkpointer

# Placeholder for VectorStoreRetriever if it's a custom or specific class
# from langgraph.prebuilt import VectorStoreRetriever # Assuming this path, adjust if different
# For now, assume vector_store.as_retriever() or retriever_tool is used, making this import flexible

class RagState(TypedDict):
    """The state of the RAG agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    original_question: str
    documents: Optional[Sequence[Document]]
    document_assessment: Optional[str]
    generation: Optional[str]
    iterations: int
    current_phase_iterations: int
    attempted_external_search: bool
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps


def create_rag_agent(
    llm: LanguageModelLike,
    embedding_model: Any,
    vector_store: Any,
    retriever_tool: Optional[BaseTool] = None,
    external_search_tools: Optional[Sequence[BaseTool]] = None,
    do_document_grading: bool = True,
    do_external_search: bool = True,
    system_message_prompt: Optional[str] = (
        "You are a helpful RAG assistant. "
        "Use the retrieved documents to answer the question. "
        "If the documents are not relevant or insufficient, you can try to rephrase the question or search externally."
    ),
    checkpointer: Optional[Checkpointer] = None,
    debug: bool = False,
) -> CompiledGraph:
    """Creates a RAG agent graph.

    Args:
        llm: The language model to use.
        embedding_model: The embedding model to use (often implicitly used by vector_store or retriever_tool).
        vector_store: The vector store for primary document retrieval.
        retriever_tool: An optional pre-configured tool for document retrieval. If None, a retriever will be 
                      created from vector_store.
        external_search_tools: Optional list of tools for external search (e.g., web, ArXiv).
        do_document_grading: If True, a step will be added to grade document relevance.
        do_external_search: If True, external_search_tools will be used if initial retrieval is insufficient.
        system_message_prompt: The system prompt to use for the LLM.
        checkpointer: An optional checkpointer for persisting state.
        debug: If True, enables debug logging for the graph.

    Returns:
        A compiled LangGraph runnable for the RAG agent.
    """
    MAX_QUERY_TRANSFORMATIONS = 2  # Max attempts to rephrase a query

    _retriever: Union[BaseTool, Runnable]
    if retriever_tool is not None:
        _retriever = retriever_tool
    elif hasattr(vector_store, 'as_retriever') and callable(vector_store.as_retriever):
        # Standard LangChain vector stores have this method
        _retriever = vector_store.as_retriever()
    else:
        # Fallback for simpler vector stores or direct use, assuming they are runnable/callable
        # This part might need adjustment based on how `vector_store` is expected to be used if not a LangChain VS
        # For now, assuming it's a LangChain VectorStore or compatible retriever
        raise ValueError("vector_store must have an 'as_retriever' method if retriever_tool is not provided.")

    # 1. Define Node Functions
    def retrieve_documents_node(state: RagState) -> dict:
        print("---RETRIEVING DOCUMENTS---")
        question = state["question"]
        if state.get("original_question") is None:
            original_question = question
        else:
            original_question = state["original_question"]

        try:
            retrieved_docs = _retriever.invoke(question) # type: ignore
        except Exception as e:
            print(f"Error invoking retriever: {e}")
            retrieved_docs = []

        return {
            "documents": retrieved_docs,
            "original_question": original_question,
            "iterations": state.get("iterations", 0) + 1,
            "current_phase_iterations": 0,
            "attempted_external_search": False,
            "document_assessment": None,
        }

    def grade_retrieved_documents_node(state: RagState) -> dict:
        print("---GRADING DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]

        if not documents:
            print("No documents to grade.")
            return {"document_assessment": "no_documents"}

        formatted_docs = "\n\n".join([doc.page_content for doc in documents])
        grading_prompt_template = (
            "Given the following question and retrieved documents, assess if the documents are relevant to answer the question. "
            "Respond with only 'relevant' or 'not_relevant'.\n\n"
            "Question: {question}\n\n"
            "Documents:\n{documents_text}\n\n"
            "Assessment:"
        )
        grading_prompt = grading_prompt_template.format(
            question=question, documents_text=formatted_docs
        )

        try:
            response = llm.invoke(grading_prompt) # type: ignore
            if hasattr(response, 'content'):
                assessment = response.content.strip().lower()
            elif isinstance(response, str):
                assessment = response.strip().lower()
            else:
                print(f"Unexpected LLM response type for grading: {type(response)}")
                assessment = "not_relevant"

            if assessment not in ["relevant", "not_relevant"]:
                print(f"LLM grader returned non-standard assessment: '{assessment}'. Defaulting to not_relevant.")
                assessment = "not_relevant"

        except Exception as e:
            print(f"Error invoking LLM for grading: {e}")
            assessment = "not_relevant"

        print(f"Document assessment: {assessment}")
        return {"document_assessment": assessment}

    def generate_answer_node(state: RagState) -> dict:
        print("---GENERATING ANSWER---")
        question = state["question"]
        documents = state["documents"]
        document_assessment = state.get("document_assessment")

        documents_text = ""
        use_documents = False
        if documents:
            if not do_document_grading or document_assessment == "relevant":
                documents_text = "\n\n".join([doc.page_content for doc in documents])
                use_documents = True
            elif document_assessment == "no_documents":
                documents_text = "No documents were found during retrieval."
            else:
                documents_text = "The retrieved documents were not considered relevant to the question."
        else:
            documents_text = "No documents were retrieved."

        prompt_parts = [system_message_prompt, f"User Question: {question}"]
        if use_documents:
            prompt_parts.append(f"Retrieved Documents:\n{documents_text}")
            prompt_parts.append("Based on the question and the retrieved documents, provide a comprehensive answer.")
        else:
            prompt_parts.append(f"Context: {documents_text}")
            prompt_parts.append("Please answer the question based on your general knowledge or indicate if you cannot answer without relevant documents.")
        generation_prompt = "\n\n".join(prompt_parts)

        try:
            response = llm.invoke(generation_prompt) # type: ignore
            if hasattr(response, 'content'):
                generated_answer = response.content
            elif isinstance(response, str):
                generated_answer = response
            else:
                print(f"Unexpected LLM response type for generation: {type(response)}")
                generated_answer = "Sorry, I encountered an error while generating the answer."
        except Exception as e:
            print(f"Error invoking LLM for generation: {e}")
            generated_answer = "Sorry, I encountered an error and cannot provide an answer at this time."
        print(f"Generated answer: {generated_answer[:100]}...")
        new_messages = state.get("messages", []) + [AIMessage(content=generated_answer)]

        return {
            "generation": generated_answer,
            "messages": new_messages
        }

    def transform_query_node(state: RagState) -> dict:
        print("---TRANSFORMING QUERY---")
        original_question = state["original_question"]
        current_question = state["question"]
        documents = state["documents"]
        document_assessment = state.get("document_assessment")

        transformation_context = f"The original question was: '{original_question}'."
        transformation_context += f" The last attempted query was: '{current_question}'."

        if not documents or document_assessment == "no_documents":
            transformation_context += " No documents were found with the last query."
        elif document_assessment == "not_relevant":
            transformation_context += " The retrieved documents were not relevant to the question."

        transformation_prompt_template = (
            "You are an expert at rephrasing questions to improve information retrieval.\n"
            "{context}\n\n"
            "Based on the original question and the issues encountered, provide a new, rephrased question that might yield better search results. "
            "Output ONLY the new question.\n\n"
            "New Question:"
        )
        transformation_prompt = transformation_prompt_template.format(context=transformation_context)

        try:
            response = llm.invoke(transformation_prompt) # type: ignore
            if hasattr(response, 'content'):
                new_question = response.content.strip()
            elif isinstance(response, str):
                new_question = response.strip()
            else:
                print(f"Unexpected LLM response type for query transformation: {type(response)}")
                new_question = original_question
            
            if not new_question or new_question.lower() == "new question:":
                 print("Query transformation resulted in empty or boilerplate response. Using original question.")
                 new_question = original_question

        except Exception as e:
            print(f"Error invoking LLM for query transformation: {e}")
            new_question = original_question

        print(f"Transformed query: {new_question}")
        return {
            "question": new_question,
            "documents": None,
            "document_assessment": None,
            "generation": None,
            "current_phase_iterations": state.get("current_phase_iterations", 0) + 1,
        }

    def perform_external_search_node(state: RagState) -> dict:
        print("---PERFORMING EXTERNAL SEARCH---")
        question = state["question"]
        all_external_docs: list[Document] = []

        if not external_search_tools:
            return {"documents": [], "attempted_external_search": True}

        for tool in external_search_tools:
            try:
                tool_name = tool.name if hasattr(tool, 'name') else 'UnnamedTool'
                print(f"Invoking external search tool: {tool_name}")
                tool_output = tool.invoke(question)
                
                if isinstance(tool_output, str):
                    all_external_docs.append(Document(page_content=tool_output, metadata={"source": tool_name}))
                elif isinstance(tool_output, Document):
                    all_external_docs.append(tool_output)
                elif isinstance(tool_output, list) and all(isinstance(doc, Document) for doc in tool_output):
                    all_external_docs.extend(tool_output)
                elif isinstance(tool_output, list) and all(isinstance(item, dict) and "page_content" in item for item in tool_output):
                    for item_dict in tool_output:
                        all_external_docs.append(Document(**item_dict))
                else:
                    print(f"Tool {tool_name} output not directly convertible to Document: {type(tool_output)}")

            except Exception as e:
                print(f"Error invoking external search tool {tool.name if hasattr(tool, 'name') else 'UnnamedTool'}: {e}")
        
        print(f"Found {len(all_external_docs)} documents from external search.")
        return {
            "documents": all_external_docs,
            "attempted_external_search": True,
            "document_assessment": None,
            "current_phase_iterations": state.get("current_phase_iterations", 0),
        }

    workflow = StateGraph(RagState)
    workflow.add_node("retrieve", retrieve_documents_node)
    if do_document_grading:
        workflow.add_node("grade_documents", grade_retrieved_documents_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("transform_query", transform_query_node)
    if do_external_search and external_search_tools:
        workflow.add_node("external_search", perform_external_search_node)

    def route_after_retrieval(state: RagState) -> str:
        print(f"---ROUTING AFTER RETRIEVAL (iteration {state.get('iterations')}, phase iter {state.get('current_phase_iterations')})---")
        documents = state.get("documents")
        if documents:
            print("Documents found by primary retriever.")
            return "grade_documents" if do_document_grading else "generate"
        else:
            print("No documents found by primary retriever.")
            if do_external_search and external_search_tools and not state.get("attempted_external_search"):
                print("Attempting external search.")
                return "external_search"
            elif state.get("current_phase_iterations", 0) < MAX_QUERY_TRANSFORMATIONS:
                print("Attempting to transform query.")
                return "transform_query"
            else:
                print("Max transformations or no external search option. Ending.")
                return END

    def route_after_grading(state: RagState) -> str:
        print(f"---ROUTING AFTER GRADING (assessment: {state.get('document_assessment')})---")
        assessment = state.get("document_assessment")
        if assessment == "relevant":
            print("Documents relevant. Proceeding to generate.")
            return "generate"
        else:
            print(f"Documents assessment: {assessment}. Fallback options.")
            if do_external_search and external_search_tools and not state.get("attempted_external_search"):
                print("Attempting external search due to poor grading.")
                return "external_search"
            elif state.get("current_phase_iterations", 0) < MAX_QUERY_TRANSFORMATIONS:
                print("Attempting to transform query due to poor grading.")
                return "transform_query"
            else:
                print("Max transformations or no external search after poor grading. Ending.")
                return END

    def route_after_generation(state: RagState) -> str:
        print(f"---ROUTING AFTER GENERATION---")
        generated_answer = state.get("generation", "").lower()
        is_insufficient = (
            len(generated_answer) < 20 or
            any(phrase in generated_answer for phrase in ["don't know", "cannot answer", "not sure", "unable to find"])
        )
        if not is_insufficient:
            print("Answer seems sufficient. Ending.")
            return END
        else:
            print("Answer deemed insufficient.")
            if do_external_search and external_search_tools and not state.get("attempted_external_search"):
                print("Attempting external search due to insufficient answer.")
                return "external_search"
            elif state.get("current_phase_iterations", 0) < MAX_QUERY_TRANSFORMATIONS:
                print("Attempting to transform query due to insufficient answer.")
                return "transform_query"
            else:
                print("Max transformations or no external search after insufficient answer. Ending.")
                return END

    def route_after_transform_query(state: RagState) -> str:
        print(f"---ROUTING AFTER TRANSFORM QUERY (new query: '{state.get('question')}')---")
        return "retrieve"

    def route_after_external_search(state: RagState) -> str:
        print(f"---ROUTING AFTER EXTERNAL SEARCH---")
        documents_from_external_search = state.get("documents")
        if not documents_from_external_search:
            print("External search yielded no documents.")
            if state.get("current_phase_iterations", 0) < MAX_QUERY_TRANSFORMATIONS:
                print("Attempting to transform query as external search failed.")
                return "transform_query"
            else:
                print("Max transformations and external search also failed. Ending.")
                return END
        else:
            print("Documents found from external search.")
            return "grade_documents" if do_document_grading else "generate"

    workflow.set_entry_point("retrieve")
    workflow.add_conditional_edges("retrieve", route_after_retrieval,
        {
            "grade_documents": "grade_documents" if do_document_grading else "generate",
            "generate": "generate",
            "external_search": "external_search" if do_external_search and external_search_tools else "transform_query",
            "transform_query": "transform_query",
            END: END,
        }
    )

    if do_document_grading:
        workflow.add_conditional_edges("grade_documents", route_after_grading,
            {
                "generate": "generate",
                "external_search": "external_search" if do_external_search and external_search_tools else "transform_query",
                "transform_query": "transform_query",
                END: END,
            }
        )
    
    workflow.add_conditional_edges("generate", route_after_generation,
        {
            "external_search": "external_search" if do_external_search and external_search_tools else "transform_query",
            "transform_query": "transform_query",
            END: END,
        }
    )

    workflow.add_conditional_edges("transform_query", route_after_transform_query, {"retrieve": "retrieve"})

    if do_external_search and external_search_tools:
        workflow.add_conditional_edges("external_search", route_after_external_search,
            {
                "grade_documents": "grade_documents" if do_document_grading else "generate",
                "generate": "generate",
                "transform_query": "transform_query",
                END: END,
            }
        )

    return workflow.compile(checkpointer=checkpointer, debug=debug)
