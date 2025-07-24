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
CALL {
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
CALL {
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
indices = [
    "CREATE FULLTEXT INDEX movie IF NOT EXISTS FOR (m:Movie) ON EACH [m.title]",
    "CREATE FULLTEXT INDEX person IF NOT EXISTS FOR (p:Person) ON EACH [p.name]"
]

for index in indices:
    try:
        graph.query(index)
    except Exception as e:
        print(f"Warning: Could not create index: {e}")

print("Database setup complete!")

# =============================================================================
# Helper Functions for Neo4j Queries
# =============================================================================

def remove_lucene_chars(text: str) -> str:
    """Remove Lucene special characters for full-text search"""
    special_chars = ["+", "-", "&", "|", "!", "(", ")", "{", "}", "[", "]", "^", '"', "~", "*", "?", ":", "\\"]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()

def generate_full_text_query(input_text: str) -> str:
    """Generate a full-text search query with fuzzy matching"""
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input_text).split() if el]
    if not words:
        return ""
    for word in words[:-1]:
        full_text_query += f" {word}~0.8 AND"
    full_text_query += f" {words[-1]}~0.8"
    return full_text_query.strip()

def get_candidates(input_text: str, entity_type: str, limit: int = 3) -> List[Dict[str, str]]:
    """Retrieve candidate entities from database using full-text search"""
    ft_query = generate_full_text_query(input_text)
    if not ft_query:
        return []
    
    candidate_query = """
    CALL db.index.fulltext.queryNodes($index, $fulltextQuery, {limit: $limit})
    YIELD node
    RETURN coalesce(node.name, node.title) AS candidate,
           [el in labels(node) WHERE el IN ['Person', 'Movie'] | el][0] AS label
    """
    
    try:
        candidates = graph.query(
            candidate_query, 
            {"fulltextQuery": ft_query, "index": entity_type, "limit": limit}
        )
        return candidates
    except Exception as e:
        print(f"Error in candidate search: {e}")
        return []

# =============================================================================
# Graph Database Tools - Information and Recommendation
# =============================================================================

def get_movie_or_person_information(entity: str, entity_type: str) -> str:
    """Get detailed information about a movie or person from the graph"""
    candidates = get_candidates(entity, entity_type)
    
    if not candidates:
        return f"No information was found about the {entity_type} '{entity}' in the database."
    
    if len(candidates) > 1:
        candidates_str = "\n".join([f"- {d['candidate']} ({d['label']})" for d in candidates])
        return f"Multiple matches found for '{entity}'. Please be more specific:\n{candidates_str}"
    
    # Get detailed information for the matched entity
    description_query = """
    MATCH (entity:Movie|Person)
    WHERE entity.title = $candidate OR entity.name = $candidate
    OPTIONAL MATCH (entity)-[r:ACTED_IN|DIRECTED|IN_GENRE]-(related)
    WITH entity, type(r) as rel_type, collect(coalesce(related.name, related.title)) as related_names
    WITH entity, rel_type + ": " + reduce(s="", n IN related_names | s + n + ", ") as relationships
    WITH entity, collect(relationships) as all_relationships
    WITH entity, 
         "Type: " + labels(entity)[0] + "\n" +
         "Name: " + coalesce(entity.title, entity.name) + "\n" +
         CASE WHEN entity.released IS NOT NULL THEN "Released: " + toString(entity.released) + "\n" ELSE "" END +
         CASE WHEN entity.imdbRating IS NOT NULL THEN "IMDB Rating: " + toString(entity.imdbRating) + "\n" ELSE "" END +
         reduce(s="", rel in all_relationships | s + substring(rel, 0, size(rel)-2) + "\n") as context
    RETURN context LIMIT 1
    """
    
    try:
        result = graph.query(description_query, {"candidate": candidates[0]["candidate"]})
        return result[0]["context"] if result else f"No detailed information found for {entity}."
    except Exception as e:
        return f"Error retrieving information for {entity}: {e}"

def get_movie_recommendations(movie: Optional[str] = None, genre: Optional[str] = None, limit: int = 5) -> str:
    """Get movie recommendations based on movie similarity or genre"""
    
    # Genre-based recommendations
    if genre and not movie:
        genre_query = """
        MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
        WHERE toLower(g.name) CONTAINS toLower($genre)
        AND m.imdbRating IS NOT NULL
        WITH m
        ORDER BY m.imdbRating DESC 
        LIMIT $limit
        RETURN m.title as movie, m.imdbRating as rating
        """
        
        try:
            results = graph.query(genre_query, {"genre": genre, "limit": limit})
            if results:
                recommendations = [f"{r['movie']} (Rating: {r['rating']})" for r in results]
                return f"Top {genre} movies:\n" + "\n".join(recommendations)
            else:
                return f"No movies found for genre '{genre}'."
        except Exception as e:
            return f"Error getting recommendations: {e}"
    
    # Movie-based recommendations using collaborative filtering
    if movie:
        candidates = get_candidates(movie, "movie", 1)
        if not candidates:
            return f"Movie '{movie}' not found in database."
        
        movie_rec_query = """
        MATCH (m1:Movie)<-[r1:RATED]-()-[r2:RATED]->(m2:Movie)
        WHERE m1.title = $movie_title AND r1.rating > 3.5 AND r2.rating > 3.5
        AND m2.imdbRating IS NOT NULL
        WITH m2, count(*) as similarity_score
        ORDER BY similarity_score DESC, m2.imdbRating DESC
        LIMIT $limit
        RETURN m2.title as movie, m2.imdbRating as rating, similarity_score
        """
        
        try:
            results = graph.query(movie_rec_query, {
                "movie_title": candidates[0]["candidate"], 
                "limit": limit
            })
            if results:
                recommendations = [f"{r['movie']} (Rating: {r['rating']}, Similarity: {r['similarity_score']})" for r in results]
                return f"Movies similar to '{candidates[0]['candidate']}':\n" + "\n".join(recommendations)
            else:
                return f"No recommendations found based on '{movie}'."
        except Exception as e:
            return f"Error getting recommendations: {e}"
    
    # General high-rated recommendations
    general_query = """
    MATCH (m:Movie)
    WHERE m.imdbRating IS NOT NULL
    WITH m
    ORDER BY m.imdbRating DESC
    LIMIT $limit
    RETURN m.title as movie, m.imdbRating as rating
    """
    
    try:
        results = graph.query(general_query, {"limit": limit})
        recommendations = [f"{r['movie']} (Rating: {r['rating']})" for r in results]
        return f"Top-rated movies:\n" + "\n".join(recommendations)
    except Exception as e:
        return f"Error getting general recommendations: {e}"

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
    """
    question: str
    generation: str
    documents: List[str]
    query_attempts: int

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
    Retrieve information from Neo4j graph database
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): New key added to state, documents, that contains retrieved information
    """
    print("---RETRIEVE---")
    question = state["question"]
    
    # Enhanced retrieval logic - try to understand what the user is asking for
    documents = []
    
    # Check if asking about a specific person (actor/director)
    person_patterns = [
        r"who.*(?:is|was)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"(?:tell me about|information about|what.*know about)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:movies|films|acted|starred)"
    ]
    
    # Check if asking about a specific movie
    movie_patterns = [
        r"(?:movie|film)\s+['\"]([^'\"]+)['\"]",
        r"['\"]([^'\"]+)['\"].*(?:movie|film)",
        r"(?:what.*about|tell me about|information about)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:\?|$)"
    ]
    
    # Check for recommendation requests
    recommendation_patterns = [
        r"recommend.*(?:movie|film)",
        r"suggest.*(?:movie|film)", 
        r"what.*(?:should|good).*(?:watch|movie|film)",
        r"best.*(?:movie|film)"
    ]
    
    # Check for genre-based queries
    genre_patterns = [
        r"(?:action|comedy|drama|horror|sci-fi|thriller|romance|animation|adventure|crime|fantasy|mystery|western).*(?:movie|film)"
    ]
    
    # Try person-based retrieval
    for pattern in person_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            person_name = match.group(1)
            info = get_movie_or_person_information(person_name, "person")
            documents.append(f"Person Information: {info}")
            break
    
    # Try movie-based retrieval  
    if not documents:
        for pattern in movie_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                movie_name = match.group(1)
                info = get_movie_or_person_information(movie_name, "movie")
                documents.append(f"Movie Information: {info}")
                break
    
    # Try recommendation-based retrieval
    if not documents:
        for pattern in recommendation_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                # Extract genre if mentioned
                genre = None
                for genre_name in ["action", "comedy", "drama", "horror", "sci-fi", "thriller", "romance", "animation", "adventure", "crime", "fantasy", "mystery", "western"]:
                    if genre_name in question.lower():
                        genre = genre_name
                        break
                
                recommendations = get_movie_recommendations(genre=genre)
                documents.append(f"Recommendations: {recommendations}")
                break
    
    # Fallback: try to extract any capitalized words as potential entities
    if not documents:
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        for word in words[:2]:  # Try first 2 capitalized entities
            # Try as person first
            person_info = get_movie_or_person_information(word, "person")
            if "No information was found" not in person_info:
                documents.append(f"Person Information: {person_info}")
                break
            
            # Try as movie
            movie_info = get_movie_or_person_information(word, "movie")
            if "No information was found" not in movie_info:
                documents.append(f"Movie Information: {movie_info}")
                break
    
    # If still no documents, provide general help
    if not documents:
        documents.append("I can help you with information about movies, actors, directors, and recommendations. Please ask about a specific person or movie, or request recommendations by genre.")
    
    return {"documents": documents, "question": question, "query_attempts": state.get("query_attempts", 0)}

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

    # Data model for document grading
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    # LLM with structured output
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt for grading
    system = """You are a grader assessing relevance of retrieved movie/entertainment information to a user question.
    Give a binary score 'yes' or 'no' to indicate whether the document information is relevant to the question.
    
    Consider information relevant if it:
    - Contains facts that help answer the question
    - Provides context about movies, actors, directors mentioned in the question  
    - Offers recommendations when user asks for suggestions
    - Gives background information that supports answering the question
    
    Consider information not relevant if it:
    - Is completely unrelated to the question topic
    - Does not contain any useful facts to answer the question
    - Is generic help text when specific information was requested"""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    chain = grade_prompt | structured_llm_grader

    # Score each document
    filtered_docs = []
    for doc in documents:
        score = chain.invoke({"question": question, "document": doc})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {"documents": filtered_docs, "question": question, "query_attempts": state.get("query_attempts", 0)}

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

    # RAG prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for movie and entertainment question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise and conversational.
        
        When providing recommendations, format them as a numbered list.
        When providing information about a person or movie, include the most relevant details."""),
        ("human", "Question: {question} \n\n Context: {context}")
    ])

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    return {"documents": documents, "question": question, "generation": generation, "query_attempts": state.get("query_attempts", 0)}

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

    # Create a query rewriting prompt
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question re-writer that converts an input question to a better version that is optimized 
        for movie/entertainment information retrieval from a graph database. Look at the input and try to reason about the underlying semantic intent / meaning.
        
        The database contains information about:
        - Movies (titles, release dates, ratings, genres)
        - People (actors, directors and their filmographies)  
        - User ratings and preferences
        - Movie recommendations
        
        Rewrite the question to be more specific and clear. If the original question is ambiguous, make it more precise.
        Focus on specific entities (movie titles, person names) or clear intent (recommendations, information lookup).
        
        Examples:
        - "What about Tom?" -> "What movies has Tom Hanks appeared in?"
        - "Good action movies?" -> "What are some highly-rated action movies?"
        - "Tell me about Batman" -> "What information do you have about Batman movies?"
        """),
        ("human", "Original question: {question}\n\nContext from previous search: {context}\n\nRewrite this question to be more specific and searchable:"),
    ])

    # Chain
    question_rewriter = rewrite_prompt | llm | StrOutputParser()
    
    # Get context from documents if available
    context = "\n".join(documents[:2]) if documents else "No relevant information found in previous search."
    
    # Re-write question
    better_question = question_rewriter.invoke({"question": question, "context": context})
    
    return {"question": better_question, "query_attempts": query_attempts}

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
    
    if not filtered_documents:
        # If no relevant documents and we haven't tried too many times, transform query
        if query_attempts < 2:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: MAX QUERY ATTEMPTS REACHED, GENERATE WITH AVAILABLE INFO---")
            return "generate"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    query_attempts = state.get("query_attempts", 0)

    # Data model for hallucination grading
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""
        binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    # LLM with structured output
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])

    chain = hallucination_prompt | structured_llm_grader
    score = chain.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering quality
        print("---GRADE GENERATION vs QUESTION---")
        
        # Data model for answer quality grading
        class GradeAnswer(BaseModel):
            """Binary score to assess answer addresses question."""
            binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

        # LLM with structured output
        structured_llm_grader = llm.with_structured_output(GradeAnswer)

        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])

        chain = answer_prompt | structured_llm_grader
        score = chain.invoke({"question": question, "generation": generation})
        grade = score.binary_score

        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            if query_attempts < 2:
                return "not useful"
            else:
                print("---MAX QUERY ATTEMPTS REACHED, ENDING---")
                return "useful"  # End with current generation
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
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
inputs = {"question": "What movies has Tom Hanks been in?", "query_attempts": 0}

print(f"Input Question: {inputs['question']}")
print("Processing...")

for output in app.stream(inputs, config={"recursion_limit": 10}):
    for key, value in output.items():
        print(f"Node '{key}' completed")

print("Final Answer:")
print(value.get("generation", "No generation available"))
print("\n" + "="*60 + "\n")

# Test 2: Movie recommendation question  
print("=== Test 2: Movie Recommendations ===")
inputs = {"question": "Can you recommend some good action movies?", "query_attempts": 0}

print(f"Input Question: {inputs['question']}")
print("Processing...")

for output in app.stream(inputs, config={"recursion_limit": 10}):
    for key, value in output.items():
        print(f"Node '{key}' completed")

print("Final Answer:")
print(value.get("generation", "No generation available"))
print("\n" + "="*60 + "\n")

# Test 3: Specific movie information
print("=== Test 3: Movie Information ===") 
inputs = {"question": "Tell me about the movie Forrest Gump", "query_attempts": 0}

print(f"Input Question: {inputs['question']}")
print("Processing...")

for output in app.stream(inputs, config={"recursion_limit": 10}):
    for key, value in output.items():
        print(f"Node '{key}' completed")

print("Final Answer:")
print(value.get("generation", "No generation available"))
print("\n" + "="*60 + "\n")

# Test 4: Complex query that might need rewriting
print("=== Test 4: Complex Query (Self-Correction Test) ===")
inputs = {"question": "What about that movie with the guy who was in Philadelphia?", "query_attempts": 0}

print(f"Input Question: {inputs['question']}")
print("Processing...")

for output in app.stream(inputs, config={"recursion_limit": 10}):
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