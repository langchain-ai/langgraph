#!/usr/bin/env python3
"""
Self-GraphRAG with Neo4j

This script demonstrates an agentic Self-RAG implementation using a Neo4j graph database 
instead of traditional vector search. We adapt the Self-RAG strategy to work with graph data, 
incorporating self-reflection and self-grading on retrieved graph data and generations.

The Self-RAG workflow includes several key decisions:
1. Should I retrieve from the graph database? - Decides when to retrieve data from Neo4j
2. Are the retrieved graph results relevant? - Grades relevance of graph query results
3. Is the LLM generation supported by the graph data? - Checks for hallucinations against graph facts
4. Is the generation useful for the question? - Evaluates response quality

We use a movie dataset with actors, directors, genres, and user ratings stored in Neo4j,
demonstrating how graph relationships enhance retrieval quality.

Usage:
    pip install langchain-community langchain-openai langgraph langchain-neo4j neo4j pandas requests python-dotenv
    
    Create a .env file with your configuration:
    OPENAI_API_KEY=your_openai_api_key
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_neo4j_password
    
    python self_graphrag.py
"""

# =============================================================================
# Setup and Imports
# =============================================================================

import os
import getpass
from typing import List, Dict, Any
from typing_extensions import TypedDict
from pprint import pprint

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

print("=== Self-GraphRAG with Neo4j ===")
print("Setting up environment and dependencies...")

# =============================================================================
# Environment Setup
# =============================================================================

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}: ")

# Set up required environment variables
_set_env("OPENAI_API_KEY")
_set_env("NEO4J_URI")  # e.g., "bolt://localhost:7687" for local Docker
_set_env("NEO4J_USERNAME")  # e.g., "neo4j"
_set_env("NEO4J_PASSWORD")  # Your Neo4j password

print("Environment variables configured!")

# =============================================================================
# Neo4j Setup and Data Ingestion
# =============================================================================

print("\n=== Neo4j Connection and Setup ===")

# Initialize Neo4j connection
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
)

# Test connection
try:
    # Use a simple query that works across all Neo4j versions
    result = graph.query("RETURN 1 as test")
    print("Neo4j connection established!")
    print("Connection test successful:", result)
except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)

# Create constraints for better performance
print("\nCreating database constraints...")
constraints = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE", 
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE"
]

for constraint in constraints:
    graph.query(constraint)

print("Constraints created successfully!")

# Import movie data using Cypher LOAD CSV
print("\nImporting movie data from tomasonjo/blog-datasets...")

movies_query = """
LOAD CSV WITH HEADERS 
FROM 'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies.csv' AS row
CALL (row) {
  MERGE (m:Movie {id: toInteger(row.movieId)})
  SET m.released = toInteger(row.released),
      m.title = row.title,
      m.imdbRating = toFloat(row.imdbRating)
  
  WITH m, row
  UNWIND split(row.director, '|') AS d
  MERGE (person:Person {name: trim(d)})
  MERGE (person)-[:DIRECTED]->(m)
  
  WITH m, row
  UNWIND split(row.actors, '|') AS a
  MERGE (person:Person {name: trim(a)})
  MERGE (person)-[:ACTED_IN]->(m)
  
  WITH m, row
  UNWIND split(row.genres, '|') AS g
  MERGE (genre:Genre {name: trim(g)})
  MERGE (m)-[:IN_GENRE]->(genre)
} IN TRANSACTIONS OF 1000 ROWS
"""

print("Importing movies data...")
graph.query(movies_query)
print("Movies data imported!")

# Import ratings data
ratings_query = """
LOAD CSV WITH HEADERS
FROM 'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/ratings.csv' AS row
CALL (row) {
  MERGE (u:User {id: toInteger(row.userId)})
  MERGE (m:Movie {id: toInteger(row.movieId)})
  MERGE (u)-[r:RATED]->(m)
  SET r.rating = toFloat(row.rating),
      r.timestamp = toInteger(row.timestamp)
} IN TRANSACTIONS OF 1000 ROWS
"""

print("Importing ratings data...")
graph.query(ratings_query)
print("Ratings data imported!")

# Create fulltext indices for better search
print("Creating fulltext indices...")
indices = [
    "CREATE FULLTEXT INDEX movie_index IF NOT EXISTS FOR (m:Movie) ON EACH [m.title]",
    "CREATE FULLTEXT INDEX person_index IF NOT EXISTS FOR (p:Person) ON EACH [p.name]"
]

for index in indices:
    graph.query(index)

print("Fulltext indices created!")

# Verify data import
stats = graph.query("""
MATCH (m:Movie) WITH count(m) AS movies
MATCH (p:Person) WITH movies, count(p) AS people
MATCH (u:User) WITH movies, people, count(u) AS users
MATCH (g:Genre) WITH movies, people, users, count(g) AS genres
MATCH ()-[r:RATED]->() WITH movies, people, users, genres, count(r) AS ratings
RETURN movies, people, users, genres, ratings
""")

print("Data import complete!")
if stats:
    print(f"Movies: {stats[0]['movies']}, People: {stats[0]['people']}, Users: {stats[0]['users']}")
    print(f"Genres: {stats[0]['genres']}, Ratings: {stats[0]['ratings']}")

# =============================================================================
# Graph Retrieval Tools
# =============================================================================

print("\n=== Setting Up Graph Retrieval Tools ===")

def remove_lucene_chars(text: str) -> str:
    """Remove Lucene special characters from search text."""
    special_chars = "+-&|!(){}[]^\"~*?:\\/"
    for char in special_chars:
        text = text.replace(char, " ")
    return text.strip()

def generate_full_text_query(input_text: str) -> str:
    """Generate a full-text search query with fuzzy matching."""
    full_text_query = ""
    words = remove_lucene_chars(input_text).split()
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def extract_person_names(question: str) -> str:
    """Extract potential person names from question, filtering out generic terms."""
    # Generic terms that are not person names
    generic_terms = {'movies', 'films', 'actor', 'actress', 'director', 'starred', 'starring', 
                    'directed', 'what', 'who', 'when', 'where', 'how', 'are', 'is', 'was', 
                    'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                    'for', 'of', 'with', 'by', 'popular', 'best', 'good', 'great', 'some'}
    
    words = remove_lucene_chars(question.lower()).split()
    person_words = [word for word in words if word not in generic_terms and len(word) > 2]
    
    if not person_words:
        return question  # Fallback to original question
    
    # Create fuzzy query for person names
    query_parts = [f"{word}~2" for word in person_words]
    return " AND ".join(query_parts)

def extract_movie_titles(question: str) -> str:
    """Extract potential movie-related terms from question."""
    # Keep movie-related terms but filter out pure question words
    movie_stop_words = {'what', 'who', 'when', 'where', 'how', 'are', 'is', 'was', 'were', 
                       'the', 'a', 'an', 'and', 'or', 'but', 'to', 'for', 'of', 'with', 'by'}
    
    words = remove_lucene_chars(question.lower()).split()
    movie_words = [word for word in words if word not in movie_stop_words and len(word) > 2]
    
    if not movie_words:
        return question  # Fallback to original question
    
    # Create fuzzy query
    query_parts = [f"{word}~2" for word in movie_words]
    return " AND ".join(query_parts)

def graph_retriever(question: str, limit: int = 5):
    """Retrieve relevant information from Neo4j graph based on question."""
    # Extract different types of queries for different entity types
    movie_query = extract_movie_titles(question)
    person_query = extract_person_names(question)
    
    print(f"DEBUG: Movie query: '{movie_query}'")
    print(f"DEBUG: Person query: '{person_query}'")
    
    # Search for movies
    movie_results = graph.query(
        """
        CALL db.index.fulltext.queryNodes('movie_index', $query) 
        YIELD node, score
        WITH node as m, score
        ORDER BY score DESC LIMIT $limit
        MATCH (m:Movie)
        OPTIONAL MATCH (m)-[:IN_GENRE]->(g:Genre)
        OPTIONAL MATCH (p:Person)-[:DIRECTED]->(m)
        OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)
        OPTIONAL MATCH (u:User)-[r:RATED]->(m)
        WITH m, collect(DISTINCT g.name) AS genres, 
             collect(DISTINCT p.name) AS directors,
             collect(DISTINCT a.name) AS actors,
             AVG(r.rating) AS avgRating,
             COUNT(r) AS numRatings, score
        RETURN 'Movie' as type, m.title AS title, m.released AS released, 
               m.imdbRating AS imdbRating, genres, directors, actors, 
               avgRating, numRatings, score
        """,
        {"query": movie_query, "limit": limit}
    )
    
    # Search for people
    person_results = graph.query(
        """
        CALL db.index.fulltext.queryNodes('person_index', $query) 
        YIELD node, score
        WITH node as p, score
        ORDER BY score DESC LIMIT $limit
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:DIRECTED]->(m:Movie)
        OPTIONAL MATCH (p)-[:ACTED_IN]->(am:Movie)
        WITH p, collect(DISTINCT m.title) AS directed_movies,
             collect(DISTINCT am.title) AS acted_movies, score
        RETURN 'Person' as type, p.name AS name, directed_movies, 
               acted_movies, score
        """,
        {"query": person_query, "limit": limit}
    )
    
    # Combine and format results
    all_results = []
    
    for result in movie_results + person_results:
        if result['type'] == 'Movie':
            content = f"Movie: {result['title']} ({result['released']})\n"
            content += f"IMDB Rating: {result['imdbRating']}\n"
            if result['avgRating']:
                content += f"User Rating: {result['avgRating']:.2f} ({result['numRatings']} ratings)\n"
            if result['directors']:
                content += f"Directors: {', '.join(result['directors'])}\n"
            if result['actors']:
                content += f"Actors: {', '.join(result['actors'][:5])}\n"  # Limit actors
            if result['genres']:
                content += f"Genres: {', '.join(result['genres'])}\n"
        else:  # Person
            content = f"Person: {result['name']}\n"
            if result['directed_movies']:
                content += f"Directed: {', '.join(result['directed_movies'][:3])}\n"
            if result['acted_movies']:
                content += f"Acted in: {', '.join(result['acted_movies'][:3])}\n"
        
        all_results.append({
            'page_content': content,
            'score': result['score'],
            'type': result['type']
        })
    
    # Sort by score and return top results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:limit]

# Test the retriever
print("Testing graph retriever...")
test_results = graph_retriever("Tom Hanks movies")
print(f"Found {len(test_results)} results for 'Tom Hanks movies':")
for i, result in enumerate(test_results[:2]):
    print(f"\n{i+1}. {result['type']} (score: {result['score']:.2f})")
    print(result['page_content'])

# =============================================================================
# LLM Graders for Graph Data
# =============================================================================

print("\n=== Setting Up LLM Graders ===")

# Data models for structured output
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved graph data."""
    binary_score: str = Field(description="Graph data is relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the graph data facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Retrieval Grader
print("Setting up retrieval grader...")
retrieval_system = """You are a grader assessing relevance of retrieved graph data to a user question about movies, actors, or entertainment. 
The graph data contains information about movies, people (actors/directors), genres, and ratings.
If the graph data contains information related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' to indicate whether the graph data is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", retrieval_system),
    ("human", "Retrieved graph data: \n\n {document} \n\n User question: {question}"),
])

retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)

# Test the grader
question = "What movies did Tom Hanks star in?"
docs = graph_retriever(question)
if docs:
    doc_content = docs[0]['page_content']
    grade_result = retrieval_grader.invoke({"question": question, "document": doc_content})
    print(f"Question: {question}")
    print(f"Retrieved data relevance: {grade_result.binary_score}")

# Generation Chain
print("Setting up generation chain...")
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an assistant for question-answering tasks about movies, actors, and entertainment.
     Use the following retrieved graph data to answer the question. The data comes from a movie database 
     containing information about films, actors, directors, genres, and user ratings.
     If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""),
    ("human", "Question: {question} \n\nGraph Data: {context} \n\nAnswer:"),
])

def format_docs(docs):
    return "\n\n".join([doc['page_content'] for doc in docs])

rag_chain = generation_prompt | llm | StrOutputParser()

# Test generation
if docs:
    generation = rag_chain.invoke({"context": format_docs(docs), "question": question})
    print(f"Generated answer: {generation}")

# Hallucination Grader
print("Setting up hallucination grader...")
hallucination_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved graph data facts about movies and entertainment. 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the graph data facts."""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", hallucination_system),
    ("human", "Graph data facts: \n\n {documents} \n\n LLM generation: {generation}"),
])

hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucinations)

# Test hallucination grader
if docs and 'generation' in locals():
    hallucination_result = hallucination_grader.invoke({"documents": format_docs(docs), "generation": generation})
    print(f"Hallucination check: {hallucination_result.binary_score}")

# Answer Grader
print("Setting up answer grader...")
answer_system = """You are a grader assessing whether an answer addresses / resolves a question about movies, actors, or entertainment.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_system),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])

answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)

# Test answer grader
if 'generation' in locals():
    answer_result = answer_grader.invoke({"question": question, "generation": generation})
    print(f"Answer quality: {answer_result.binary_score}")

# Question Rewriter
print("Setting up question rewriter...")
rewrite_system = """You are a question re-writer that converts an input question to a better version optimized 
for graph database search about movies, actors, directors, and entertainment. 
Consider that the graph contains movies, people, genres, and ratings. 
Rephrase questions to be more specific and searchable within this domain."""

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
])

question_rewriter = re_write_prompt | llm | StrOutputParser()

# Test question rewriter
test_rewrite = question_rewriter.invoke({"question": "good action movies"})
print(f"Original: 'good action movies'")
print(f"Rewritten: '{test_rewrite}'")

# =============================================================================
# GraphRAG Workflow with LangGraph
# =============================================================================

print("\n=== Building GraphRAG Workflow ===")

# Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of retrieved graph data
        query_attempts: number of query transformation attempts
    """
    question: str
    generation: str
    documents: List[Dict[str, Any]]
    query_attempts: int

# Node Functions
def retrieve(state):
    """Retrieve data from Neo4j graph"""
    print("---RETRIEVE FROM GRAPH DATABASE---")
    question = state["question"]
    documents = graph_retriever(question)
    query_attempts = state.get("query_attempts", 0)
    return {"documents": documents, "question": question, "query_attempts": query_attempts}

def generate(state):
    """Generate answer using graph data"""
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    query_attempts = state.get("query_attempts", 0)
    
    if not documents:
        # Generate an answer indicating no relevant data was found
        generation = "I don't have sufficient information in the movie database to answer your question. Please try asking about specific movies, actors, directors, or genres that might be in our dataset."
    else:
        generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    
    return {"documents": documents, "question": question, "generation": generation, "query_attempts": query_attempts}

def grade_documents(state):
    """Determines whether the retrieved graph data is relevant to the question."""
    print("---CHECK GRAPH DATA RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    query_attempts = state.get("query_attempts", 0)
    
    # Score each piece of graph data
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d['page_content']}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: GRAPH DATA RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: GRAPH DATA NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question, "query_attempts": query_attempts}

def transform_query(state):
    """Transform the query to produce a better question for graph search."""
    print("---TRANSFORM QUERY FOR GRAPH SEARCH---")
    question = state["question"]
    documents = state["documents"]
    query_attempts = state.get("query_attempts", 0) + 1
    
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question, "query_attempts": query_attempts}

# Edge Functions
def decide_to_generate(state):
    """Determines whether to generate an answer, or re-generate a question."""
    print("---ASSESS GRADED GRAPH DATA---")
    filtered_documents = state["documents"]
    query_attempts = state.get("query_attempts", 0)
    
    print(f"DEBUG: Found {len(filtered_documents)} documents, query_attempts: {query_attempts}")
    
    if not filtered_documents:
        # Check if we've tried too many query transformations
        if query_attempts >= 2:  # Reduced from 3 to 2 for faster exit
            print("---DECISION: MAX QUERY ATTEMPTS REACHED, GENERATE WITH NO DATA---")
            return "generate"
        else:
            # All graph data has been filtered - re-generate a new query
            print("---DECISION: ALL GRAPH DATA NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
    else:
        # We have relevant graph data, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """Determines whether the generation is grounded in the graph data and answers question."""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    query_attempts = state.get("query_attempts", 0)
    
    # If we have no documents, just check if the generation addresses the question
    if not documents:
        print("---NO DOCUMENTS TO CHECK AGAINST, EVALUATING ANSWER QUALITY---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "useful"  # Still return useful to avoid loops when no docs available
    
    score = hallucination_grader.invoke(
        {"documents": format_docs(documents), "generation": generation}
    )
    grade = score.binary_score
    
    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN GRAPH DATA---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            # Avoid infinite loops - if we've tried multiple times, accept the answer
            if query_attempts >= 2:
                print("---MAX ATTEMPTS REACHED, ACCEPTING ANSWER---")
                return "useful"
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN GRAPH DATA, RE-TRY---")
        # Avoid infinite regeneration
        if query_attempts >= 2:
            print("---MAX ATTEMPTS REACHED, ACCEPTING ANSWER---")
            return "useful"
        return "not supported"

# Build Graph Workflow
print("Building LangGraph workflow...")
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
print("GraphRAG workflow compiled successfully!")

# =============================================================================
# Running the GraphRAG System
# =============================================================================

print("\n=== Running Self-GraphRAG Tests ===")

# Test 1: Actor filmography question
print("\n=== Test 1: Actor Filmography ===")
inputs = {"question": "What are some popular movies starring Tom Hanks?", "query_attempts": 0}

for output in app.stream(inputs, config={"recursion_limit": 10}):
    for key, value in output.items():
        print(f"Node '{key}':")
    print("\n---\n")

# Final generation
print("Final Answer:")
pprint(value["generation"])
print("\n" + "="*50 + "\n")

# Test 2: Genre and recommendation question
print("=== Test 2: Genre Recommendations ===")
inputs = {"question": "Recommend some highly-rated action movies from the 1990s", "query_attempts": 0}

for output in app.stream(inputs, config={"recursion_limit": 10}):
    for key, value in output.items():
        print(f"Node '{key}':")
    print("\n---\n")

# Final generation
print("Final Answer:")
pprint(value["generation"])
print("\n" + "="*50 + "\n")

# Test 3: Director and collaboration question
print("=== Test 3: Director Collaborations ===")
inputs = {"question": "What movies did Christopher Nolan direct and who were the main actors?", "query_attempts": 0}

for output in app.stream(inputs, config={"recursion_limit": 10}):
    for key, value in output.items():
        print(f"Node '{key}':")
    print("\n---\n")

# Final generation
print("Final Answer:")
pprint(value["generation"])
print("\n" + "="*50 + "\n")

# Test 4: A challenging query to test the self-correction mechanism
print("=== Test 4: Challenging Query (Self-Correction Test) ===")
inputs = {"question": "space adventures", "query_attempts": 0}  # Vague query to trigger query transformation

for output in app.stream(inputs, config={"recursion_limit": 10}):
    for key, value in output.items():
        print(f"Node '{key}':")
    print("\n---\n")

# Final generation
print("Final Answer:")
pprint(value["generation"])
print("\n" + "="*50 + "\n")

# =============================================================================
# Advanced Graph Queries
# =============================================================================

print("=== Advanced Graph Queries ===")

# Find actors who worked together in multiple movies
collaboration_query = """
MATCH (a1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Person)
WHERE a1.name < a2.name  // Avoid duplicates
WITH a1, a2, collect(m.title) AS movies
WHERE size(movies) >= 2
RETURN a1.name AS actor1, a2.name AS actor2, movies
ORDER BY size(movies) DESC
LIMIT 5
"""

print("\nActor Collaborations:")
collaboration_results = graph.query(collaboration_query)
for i, result in enumerate(collaboration_results):
    print(f"{i+1}. {result['actor1']} & {result['actor2']}")
    print(f"   Movies: {', '.join(result['movies'])}")

# Movies recommendation based on user ratings
recommendation_query = """
MATCH (m:Movie)<-[r:RATED]-(u:User)
WHERE r.rating >= 4.0
WITH m, avg(r.rating) AS avgRating, count(r) AS numRatings
WHERE numRatings >= 10
MATCH (m)-[:IN_GENRE]->(g:Genre)
RETURN m.title AS movie, m.released AS year, avgRating, numRatings, collect(g.name) AS genres
ORDER BY avgRating DESC, numRatings DESC
LIMIT 5
"""

print("\nTop User-Rated Movies:")
rating_results = graph.query(recommendation_query)
for i, result in enumerate(rating_results):
    print(f"{i+1}. {result['movie']} ({result['year']})")
    print(f"   Rating: {result['avgRating']:.2f}/5.0 ({result['numRatings']} ratings)")
    print(f"   Genres: {', '.join(result['genres'])}")

# =============================================================================
# System Performance Analysis
# =============================================================================

print("\n=== System Performance Analysis ===")

analysis_queries = {
    "Graph Statistics": """
        MATCH (n) 
        WITH labels(n) AS nodeType, count(*) AS count
        RETURN nodeType[0] AS nodeType, count
        ORDER BY count DESC
    """,
    
    "Relationship Statistics": """
        MATCH ()-[r]->() 
        WITH type(r) AS relType, count(*) AS count
        RETURN relType, count
        ORDER BY count DESC
    """,
    
    "Most Connected Actors": """
        MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
        WITH p, count(m) AS movieCount
        WHERE movieCount >= 3
        RETURN p.name AS actor, movieCount
        ORDER BY movieCount DESC
        LIMIT 10
    """
}

for title, query in analysis_queries.items():
    print(f"\n{title}:")
    results = graph.query(query)
    for result in results[:5]:
        print(f"  {result}")

# =============================================================================
# Conclusion
# =============================================================================

print("\n=== Self-GraphRAG Demo Complete ===")
print("""
This script demonstrated a Self-GraphRAG implementation that combines:

Key Features:
1. Graph-Based Retrieval: Uses Neo4j to store and query complex relationships
2. Self-Reflection Mechanisms: Implements grading for relevance, hallucination detection, and answer quality
3. Adaptive Query Processing: Automatically rewrites questions for better graph search results
4. Agentic Workflow: Uses LangGraph to orchestrate the self-correcting RAG pipeline

Advantages over Vector RAG:
- Structured Relationships: Can answer complex queries about collaborations and connections
- Factual Accuracy: Graph structure enforces data integrity and relationships
- Rich Context: Provides detailed relationship information
- Scalable Queries: Cypher queries can efficiently traverse large graph structures

Self-RAG Benefits:
- Quality Control: Multiple validation steps ensure accurate and relevant responses
- Adaptive Behavior: System can recover from poor initial queries through rephrasing
- Hallucination Detection: Validates answers against graph data
- Iterative Improvement: Continues refining until satisfactory results are achieved
""")

print("Script execution completed successfully!")