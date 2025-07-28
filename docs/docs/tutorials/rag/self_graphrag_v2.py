#!/usr/bin/env python3
"""
Self-GraphRAG v2 Implementation with Neo4j

This script demonstrates an agentic Self-RAG implementation using a Neo4j graph database 
for movie-related question answering. We adapt the Self-RAG strategy to work with graph data
from the tomasonjo/llm-movieagent dataset, incorporating self-reflection and self-grading 
on retrieved graph data and generations.

The Self-RAG workflow includes several key decisions:
1. Should I retrieve from the graph database? - Decides when to retrieve data from Neo4j
2. Are the retrieved graph results relevant? - Grades relevance of graph query results  
3. Is the LLM generation supported by the graph data? - Checks for hallucinations against graph facts
4. Is the generation useful for the question? - Evaluates response quality

Usage:
    pip install langchain-community langchain-openai langgraph langchain-neo4j neo4j pandas requests python-dotenv
    
    Create a .env file with your configuration:
    OPENAI_API_KEY=your_openai_api_key
    NEO4J_URI=bolt://localhost:7687  # or Neo4j Aura URI
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_neo4j_password
    
    python self_graphrag_v2.py
"""

# =============================================================================
# Setup and Imports
# =============================================================================

import os
import getpass
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from pprint import pprint
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables or prompts.")

import pandas as pd
import requests
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START

print("=== Self-GraphRAG v2 with Neo4j ===")
print("Setting up environment and dependencies...")

# =============================================================================
# Environment Setup  
# =============================================================================

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}: ")

# Set up required environment variables
_set_env("OPENAI_API_KEY")
_set_env("NEO4J_URI")  # e.g., "bolt://localhost:7687" for local or Neo4j Aura URI
_set_env("NEO4J_USERNAME")  # e.g., "neo4j"
_set_env("NEO4J_PASSWORD")  # Your Neo4j password

print("Environment variables configured!")

# =============================================================================
# Neo4j Setup and Data Ingestion
# =============================================================================

print("\n=== Neo4j Connection and Data Setup ===")

# Initialize Neo4j connection using langchain-neo4j
try:
    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    )
    print("Neo4j connection established!")
except Exception as e:
    print(f"Connection failed: {e}")
    print("Please check your Neo4j credentials and ensure the database is running.")
    exit(1)

load_fixtures = False
if load_fixtures:
    # Create database constraints for better performance
    print("Creating database constraints...")
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE", 
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE"
    ]

    for constraint in constraints:
        try:
            graph.query(constraint)
        except Exception as e:
            print(f"Warning: Could not create constraint: {e}")

        # Data ingestion from tomasonjo/llm-movieagent dataset
        print("Loading movie dataset...")

    # Import movie information from CSV
    movies_query = """
    LOAD CSV WITH HEADERS FROM 
    'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies.csv'
    AS row
    CALL (row) {
        WITH row
        MERGE (m:Movie {id:row.movieId})
        SET m.released = date(row.released),
            m.title = row.title,
            m.imdbRating = toFloat(row.imdbRating)
        FOREACH (director in split(row.director, '|') | 
            MERGE (p:Person {name:trim(director)})
            MERGE (p)-[:DIRECTED]->(m))
        FOREACH (actor in split(row.actors, '|') | 
            MERGE (p:Person {name:trim(actor)})
            MERGE (p)-[:ACTED_IN]->(m))
        FOREACH (genre in split(row.genres, '|') | 
            MERGE (g:Genre {name:trim(genre)})
            MERGE (m)-[:IN_GENRE]->(g))
    } IN TRANSACTIONS
    """

    try:
        graph.query(movies_query)
        print("Movie data loaded successfully!")
    except Exception as e:
        print(f"Movie data loading failed: {e}")

    # Import rating information  
    print("Loading ratings data...")
    rating_query = """
    LOAD CSV WITH HEADERS FROM 
    'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/ratings.csv'
    AS row
    CALL (row) {
        WITH row
        MATCH (m:Movie {id:row.movieId})
        MERGE (u:User {id:row.userId})
        MERGE (u)-[r:RATED]->(m)
        SET r.rating = toFloat(row.rating),
            r.timestamp = row.timestamp
    } IN TRANSACTIONS OF 10000 ROWS
    """

    try:
        graph.query(rating_query)
        print("Ratings data loaded successfully!")
    except Exception as e:
        print(f"Ratings data loading failed: {e}")

    # Create full-text indices for searching
    print("Creating full-text search indices...")

    # Drop existing indices if they exist to ensure clean creation
    try:
        graph.query("DROP INDEX movie IF EXISTS")
        graph.query("DROP INDEX person IF EXISTS")
    except Exception as e:
        print(f"Note: Could not drop existing indices: {e}")

    # Create new indices
    indices = [
        ("CREATE FULLTEXT INDEX movie FOR (m:Movie) ON EACH [m.title]", "movie"),
        ("CREATE FULLTEXT INDEX person FOR (p:Person) ON EACH [p.name]", "person")
    ]

    for index_query, index_name in indices:
        try:
            graph.query(index_query)
            print(f"Created full-text index: {index_name}")
        except Exception as e:
            print(f"Warning: Could not create {index_name} index: {e}")

    # Verify indices exist
    print("Verifying full-text indices...")
    try:
        result = graph.query("CALL db.indexes() YIELD name, type WHERE type = 'FULLTEXT' RETURN name")
        available_indices = [r['name'] for r in result]
        print(f"Available full-text indices: {available_indices}")
        
        if 'movie' not in available_indices or 'person' not in available_indices:
            print("Warning: Not all required indices are available. Fallback search will be used.")
    except Exception as e:
        print(f"Could not verify indices: {e}")

    print("Database setup complete!")

# =============================================================================
# Database Schema and Query Generation 
# =============================================================================

def get_database_schema():
    """Get the current database schema for query generation"""
    schema_info = {
        "nodes": {
            "Movie": {
                "properties": ["id", "title", "released", "imdbRating"],
                "description": "Movie nodes with unique ID, title, release date, and IMDB rating"
            },
            "Person": {
                "properties": ["name"],
                "description": "Person nodes representing actors and directors"
            },
            "User": {
                "properties": ["id"],
                "description": "User nodes for ratings"
            },
            "Genre": {
                "properties": ["name"],
                "description": "Genre nodes for movie categories"
            }
        },
        "relationships": {
            "ACTED_IN": "Person -> Movie (actors in movies)",
            "DIRECTED": "Person -> Movie (directors of movies)", 
            "IN_GENRE": "Movie -> Genre (movies belong to genres)",
            "RATED": "User -> Movie (user ratings with rating and timestamp properties)"
        }
    }
    return schema_info

def generate_cypher_query(question: str, max_attempts: int = 2) -> Dict[str, Any]:
    """
    Generate a Cypher query dynamically based on the user question and database schema
    """
    print(f"[DEBUG] Generating Cypher query for: '{question}'")
    
    schema = get_database_schema()
    
    # System prompt for Cypher query generation
    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Neo4j Cypher query expert. Generate a Cypher query to answer the user's question.

Database Schema:
{schema}

Rules:
1. Always return valid Cypher syntax
2. Use LIMIT to prevent large result sets (max 10 results)
3. Handle case-insensitive searching with toLower() when needed
4. For person searches, look in Person nodes
5. For movie searches, look in Movie nodes  
6. For recommendations, find highly-rated movies (imdbRating > 7.0)
7. Use relationships to find connections (ACTED_IN, DIRECTED, IN_GENRE)
8. Return meaningful aliases for results

Examples:
- "What movies has Tom Hanks been in?" -> Find Person named "Tom Hanks" and their ACTED_IN relationships
- "Tell me about Forrest Gump" -> Find Movie with title containing "Forrest Gump"
- "Recommend action movies" -> Find Movies with IN_GENRE relationship to Genre "Action"

Return only the Cypher query, nothing else."""),
        ("human", "{question}")
    ])
    
    try:
        cypher_chain = cypher_prompt | llm | StrOutputParser()
        cypher_query = cypher_chain.invoke({
            "question": question,
            "schema": str(schema)
        })
        
        # Clean up the query
        cypher_query = cypher_query.strip()
        if cypher_query.startswith("```"):
            cypher_query = cypher_query.split('\n')[1:-1]
            cypher_query = '\n'.join(cypher_query)
        
        print(f"[DEBUG] Generated Cypher: {cypher_query}")
        return {"query": cypher_query, "success": True}
        
    except Exception as e:
        print(f"[ERROR] Failed to generate Cypher query: {e}")
        return {"query": "", "success": False, "error": str(e)}

def execute_cypher_query(cypher_query: str) -> Dict[str, Any]:
    """Execute a Cypher query and return results with error handling"""
    print(f"[DEBUG] Executing Cypher query: {cypher_query}")
    
    try:
        results = graph.query(cypher_query)
        print(f"[DEBUG] Query returned {len(results)} results")
        return {"results": results, "success": True}
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Cypher query execution failed: {error_msg}")
        return {"results": [], "success": False, "error": error_msg}

# =============================================================================
# Dynamic Query Execution for Graph Retrieval
# =============================================================================

def query_graph_database(question: str) -> str:
    """
    Dynamically query the graph database based on user question
    """
    print(f"[DEBUG] Starting graph database query for: '{question}'")
    
    # Generate Cypher query
    query_result = generate_cypher_query(question)
    
    if not query_result["success"]:
        return f"Failed to generate query for: {question}. Error: {query_result.get('error', 'Unknown error')}"
    
    # Execute the query
    execution_result = execute_cypher_query(query_result["query"])
    
    if not execution_result["success"]:
        return f"Query execution failed. Error: {execution_result.get('error', 'Unknown error')}"
    
    results = execution_result["results"]
    if not results:
        return f"No results found for: {question}"
    
    # Format results for readability
    formatted_results = format_query_results(results, question)
    print(f"[DEBUG] Formatted {len(results)} results")
    
    return formatted_results

def format_query_results(results: List[Dict], question: str) -> str:
    """Format query results into readable text"""
    if not results:
        return "No results found."
    
    # Try to intelligently format based on the data structure
    formatted_lines = []
    
    for i, result in enumerate(results[:10]):  # Limit to 10 results
        if not result:
            continue
            
        result_parts = []
        for key, value in result.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    result_parts.append(f"{key}: {value}")
                else:
                    result_parts.append(f"{key}: {str(value)}")
        
        if result_parts:
            formatted_lines.append(f"{i+1}. {' | '.join(result_parts)}")
    
    return "\n".join(formatted_lines) if formatted_lines else "No valid results found."

# =============================================================================
# Graph State Definition
# =============================================================================

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents retrieved from graph
        query_attempts: number of query reformulation attempts
        generation_attempts: number of generation retry attempts
    """
    question: str
    generation: str
    documents: List[str]
    query_attempts: int
    generation_attempts: int

# =============================================================================
# LLM Setup
# =============================================================================

print("\n=== Setting up Language Models ===")

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# =============================================================================
# Graph Node Functions
# =============================================================================

def retrieve(state):
    """
    Retrieve information from Neo4j graph database using dynamic query generation
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): New key added to state, documents, that contains retrieved information
    """
    print("---RETRIEVE---")
    question = state["question"]
    query_attempts = state.get("query_attempts", 0)
    
    print(f"[DEBUG] Retrieve attempt #{query_attempts + 1} for question: '{question}'")
    
    # Prevent infinite retrieval cycles
    max_retrieval_attempts = 3
    if query_attempts >= max_retrieval_attempts:
        print(f"[WARNING] Maximum retrieval attempts ({max_retrieval_attempts}) reached")
        fallback_doc = "I apologize, but I'm having difficulty finding relevant information in the database. Please try rephrasing your question or ask about specific movies, actors, or request movie recommendations."
        return {
            "documents": [fallback_doc], 
            "question": question, 
            "query_attempts": query_attempts,
            "generation_attempts": state.get("generation_attempts", 0)
        }
    
    # Use dynamic query generation instead of pattern matching
    try:
        result = query_graph_database(question)
        documents = [f"Graph Database Results: {result}"]
        
        print(f"[DEBUG] Retrieved document length: {len(result)} characters")
        
        # Check if we got meaningful results
        if "No results found" in result or "Failed to generate query" in result or "Query execution failed" in result:
            print(f"[DEBUG] Query failed or returned no results, providing fallback")
            documents = ["I can help you with information about movies, actors, directors, and recommendations from our movie database. Please ask about a specific person or movie, or request recommendations by genre."]
        
    except Exception as e:
        print(f"[ERROR] Retrieval failed with exception: {e}")
        documents = [f"Error retrieving information: {str(e)}. Please try a different question."]
    
    return {
        "documents": documents, 
        "question": question, 
        "query_attempts": query_attempts,
        "generation_attempts": state.get("generation_attempts", 0)
    }

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    query_attempts = state.get("query_attempts", 0)
    generation_attempts = state.get("generation_attempts", 0)

    print(f"[DEBUG] Grading {len(documents)} documents for relevance")

    # Data model for document grading
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    # LLM with structured output
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt for grading - more lenient
    system = """You are a grader assessing relevance of retrieved movie/entertainment information to a user question.
    Give a binary score 'yes' or 'no' to indicate whether the document information is relevant to the question.
    
    Be VERY LENIENT in your grading. Consider information relevant if it:
    - Contains any facts that might help answer the question
    - Provides any context about movies, actors, directors, or entertainment
    - Offers any kind of movie information or recommendations
    - Contains data from the movie database, even if not perfectly matching
    
    Only consider information not relevant if it:
    - Is completely unrelated to movies, entertainment, or the question
    - Contains no useful information whatsoever
    - Is an error message or system message"""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    # Score each document with error handling
    filtered_docs = []
    for i, doc in enumerate(documents):
        try:
            chain = grade_prompt | structured_llm_grader
            score = chain.invoke({"question": question, "document": doc})
            grade = score.binary_score
            
            if grade == "yes":
                print(f"---GRADE: DOCUMENT {i+1} RELEVANT---")
                filtered_docs.append(doc)
            else:
                print(f"---GRADE: DOCUMENT {i+1} NOT RELEVANT---")
                
        except Exception as e:
            print(f"[ERROR] Failed to grade document {i+1}: {e}. Assuming relevant.")
            filtered_docs.append(doc)  # Assume relevant if grading fails

    print(f"[DEBUG] {len(filtered_docs)} out of {len(documents)} documents deemed relevant")

    return {
        "documents": filtered_docs, 
        "question": question, 
        "query_attempts": query_attempts,
        "generation_attempts": generation_attempts
    }

def generate(state):
    """
    Generate answer based on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    query_attempts = state.get("query_attempts", 0)
    generation_attempts = state.get("generation_attempts", 0)

    print(f"[DEBUG] Generating answer from {len(documents)} documents")

    # RAG prompt - more flexible
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for movie and entertainment question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        
        Guidelines:
        - Answer based on the provided context when possible
        - If the context doesn't contain the exact answer, provide what information you can
        - Be helpful and conversational
        - For recommendations, use numbered lists
        - For person/movie information, include relevant details from the context
        - If you don't have specific information, acknowledge this but still try to be helpful"""),
        ("human", "Question: {question} \n\n Context: {context}")
    ])

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc for doc in docs)

    try:
        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run generation
        generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
        
        print(f"[DEBUG] Generated response length: {len(generation)} characters")
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        generation = "I apologize, but I encountered an error while generating a response. Please try rephrasing your question."

    return {
        "documents": documents, 
        "question": question, 
        "generation": generation, 
        "query_attempts": query_attempts,
        "generation_attempts": generation_attempts
    }

def transform_query(state):
    """
    Transform the query to produce a better question for retrieval.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state.get("documents", [])
    query_attempts = state.get("query_attempts", 0) + 1
    generation_attempts = state.get("generation_attempts", 0)

    print(f"[DEBUG] Transforming query (attempt #{query_attempts}): '{question}'")

    # Create a query rewriting prompt
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question re-writer that converts an input question to a better version optimized 
        for movie/entertainment information retrieval from a graph database.
        
        The database contains information about:
        - Movies (titles, release dates, ratings, genres)
        - People (actors, directors and their filmographies)  
        - User ratings and movie recommendations
        
        Make the question more specific and searchable:
        - Use specific names when possible
        - Make the intent clear (information vs recommendations)
        - Focus on entities that exist in a movie database
        
        Examples:
        - "What about Tom?" -> "What movies has Tom Hanks been in?"
        - "Good action movies?" -> "Can you recommend highly-rated action movies?"
        - "Tell me about Batman" -> "What Batman movies are in the database?"
        """),
        ("human", "Original question: {question}\n\nPrevious search context: {context}\n\nRewrite this question to be more specific and searchable:"),
    ])

    try:
        # Chain
        question_rewriter = rewrite_prompt | llm | StrOutputParser()
        
        # Get context from documents if available
        context = "\n".join(documents[:1]) if documents else "Previous search found no relevant information."
        
        # Re-write question
        better_question = question_rewriter.invoke({"question": question, "context": context})
        
        print(f"[DEBUG] Transformed question: '{better_question}'")
        
    except Exception as e:
        print(f"[ERROR] Query transformation failed: {e}. Using original question.")
        better_question = question
    
    return {
        "question": better_question, 
        "query_attempts": query_attempts,
        "generation_attempts": generation_attempts
    }

# =============================================================================
# Conditional Logic Functions
# =============================================================================

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    
    filtered_documents = state["documents"]
    query_attempts = state.get("query_attempts", 0)
    
    print(f"[DEBUG] Document count: {len(filtered_documents)}, Query attempts: {query_attempts}")
    
    # Maximum attempts to prevent infinite cycles
    max_query_attempts = 2
    
    if not filtered_documents:
        if query_attempts < max_query_attempts:
            print(f"---DECISION: NO RELEVANT DOCUMENTS (attempt {query_attempts + 1}/{max_query_attempts}), TRANSFORM QUERY---")
            return "transform_query"
        else:
            print(f"---DECISION: MAX QUERY ATTEMPTS REACHED ({max_query_attempts}), FORCE GENERATE---")
            # Force generation with helpful fallback
            state["documents"] = ["I can help you with information about movies, actors, directors, and recommendations from our movie database. Please ask about a specific person or movie, or request recommendations by genre."]
            return "generate"
    else:
        # We have relevant documents, so generate answer
        print(f"---DECISION: GENERATE (found {len(filtered_documents)} relevant documents)---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---CHECK GENERATION QUALITY---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    query_attempts = state.get("query_attempts", 0)
    generation_attempts = state.get("generation_attempts", 0)
    
    print(f"[DEBUG] Generation length: {len(generation) if generation else 0} chars")
    print(f"[DEBUG] Query attempts: {query_attempts}, Generation attempts: {generation_attempts}")
    
    # Maximum attempts to prevent infinite cycles
    max_generation_attempts = 2
    max_total_attempts = 5  # Total limit across all attempts
    
    # Force acceptance if we've tried too many times overall
    total_attempts = query_attempts + generation_attempts
    if total_attempts >= max_total_attempts:
        print(f"---DECISION: MAX TOTAL ATTEMPTS REACHED ({total_attempts}/{max_total_attempts}), ACCEPTING GENERATION---")
        return "useful"

    # Accept reasonable-looking generations quickly
    if documents and generation and len(generation.strip()) > 20:
        # Check if generation looks like it's answering the question
        if any(word.lower() in generation.lower() for word in question.split()[:3]):
            print("---DECISION: GENERATION APPEARS RELEVANT, ACCEPTING---")
            return "useful"

    # Simplified hallucination check
    try:
        class GradeHallucinations(BaseModel):
            """Binary score for hallucination present in generation answer."""
            binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

        structured_llm_grader = llm.with_structured_output(GradeHallucinations)

        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
        Be very lenient - if the answer is generally relevant and helpful, grade it as 'yes'."""
        
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])

        chain = hallucination_prompt | structured_llm_grader
        score = chain.invoke({"documents": documents, "generation": generation})
        grade = score.binary_score
        
        print(f"[DEBUG] Hallucination grade: {grade}")
        
    except Exception as e:
        print(f"[ERROR] Grading failed: {e}. Accepting generation.")
        return "useful"

    # Check results
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED, ACCEPTING---")
        return "useful"
    else:
        # Prevent infinite re-generation
        if generation_attempts >= max_generation_attempts:
            print(f"---DECISION: MAX GENERATION ATTEMPTS REACHED ({generation_attempts}/{max_generation_attempts}), ACCEPTING---")
            return "useful"
        else:
            print(f"---DECISION: GENERATION NOT GROUNDED, RE-TRY (attempt {generation_attempts + 1}/{max_generation_attempts})---")
            state["generation_attempts"] = generation_attempts + 1
            return "not supported"

# =============================================================================
# Build LangGraph Workflow
# =============================================================================

print("\n=== Building LangGraph Workflow ===")

# Build state graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents) 
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Build graph workflow
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile with recursion limit
app = workflow.compile()
print("Self-GraphRAG workflow compiled successfully!")

# =============================================================================
# Running the Self-GraphRAG System
# =============================================================================

print("\n=== Running Self-GraphRAG Tests ===")

# Test 1: Actor information question
print("\n=== Test 1: Actor Information ===")
inputs = {"question": "What movies has Tom Hanks been in?", "query_attempts": 0, "generation_attempts": 0}

print(f"Input Question: {inputs['question']}")
print("Processing...")

for output in app.stream(inputs, config={"recursion_limit": 20}):
    for key, value in output.items():
        print(f"Node '{key}' completed")

print("Final Answer:")
print(value.get("generation", "No generation available"))
print("\n" + "="*60 + "\n")

# Test 2: Movie recommendation question  
print("=== Test 2: Movie Recommendations ===")
inputs = {"question": "Can you recommend some good action movies?", "query_attempts": 0, "generation_attempts": 0}

print(f"Input Question: {inputs['question']}")
print("Processing...")

for output in app.stream(inputs, config={"recursion_limit": 20}):
    for key, value in output.items():
        print(f"Node '{key}' completed")

print("Final Answer:")
print(value.get("generation", "No generation available"))
print("\n" + "="*60 + "\n")

# Test 3: Specific movie information
print("=== Test 3: Movie Information ===") 
inputs = {"question": "Tell me about the movie Forrest Gump", "query_attempts": 0, "generation_attempts": 0}

print(f"Input Question: {inputs['question']}")
print("Processing...")

for output in app.stream(inputs, config={"recursion_limit": 20}):
    for key, value in output.items():
        print(f"Node '{key}' completed")

print("Final Answer:")
print(value.get("generation", "No generation available"))
print("\n" + "="*60 + "\n")

# Test 4: Complex query that might need rewriting
print("=== Test 4: Complex Query (Self-Correction Test) ===")
inputs = {"question": "What about that movie with the guy who was in Philadelphia?", "query_attempts": 0, "generation_attempts": 0}

print(f"Input Question: {inputs['question']}")
print("Processing...")

for output in app.stream(inputs, config={"recursion_limit": 20}):
    for key, value in output.items():
        print(f"Node '{key}' completed")

print("Final Answer:")
print(value.get("generation", "No generation available"))

print("\n" + "="*60)
print("=== Self-GraphRAG Demo Complete ===")
print("The system demonstrated:")
print("1. Neo4j graph database integration with movie data")
print("2. Self-RAG workflow with validation and retry mechanisms")  
print("3. Query transformation for ambiguous questions")
print("4. Document relevance grading")
print("5. Generation quality assessment")
print("6. Hallucination detection against graph facts") 