# LangGraph GraphRAG Implementation

**Generate a Python Jupyter notebook for an agentic GraphRAG implementation using LangGraph**

### Project Goal
The primary objective is to **adapt an existing agentic RAG workflow**, specifically a **self-RAG workflow** from a LangGraph Jupyter notebook, to **utilize a graph database instead of a vector database** for retrieval and response generation [12, 00:23:50]. This implementation should fully leverage the robust principles of agentic RAG, including built-in validation, retry mechanisms, the ability to rephrase user questions, and internal sanity checks [12, 00:19:36, 00:30:07, 01:03:42].

### Leveraging Data and Structure from `tomasonjo/llm-movieagent` Repository (here /Users/spolischook/www/llm-movieagent)
*   **Data Source:** Use the **movie dataset from the `tomasonjo/llm-movieagent` GitHub repository** [11, 12, 00:44:21]. This repository implements a "semantic layer on top of a graph database" [1], using Neo4j to store information about actors, movies, and their ratings [2]. The dataset is based on the MovieLens dataset [3].
*   **Data Structure:** The data is already well-structured with defined schemas for **movies, users, persons (such as actors and directors), and genres** [12, 01:00:14]. This existing structure should serve as the foundation for the graph database.
*   **Existing Tools:** The `llm-movieagent` project already provides agents with a suite of robust tools for interacting with the graph database, including an **Information Tool, Recommendation Tool, and Memory Tool** [4].
*   **Data Ingestion:** The data can be easily populated into the graph database from **CSV files**, with data preparation requiring approximately 60 lines of code [12, 01:00:14, 01:03:42]. The repository is under the MIT license, meaning no attribution is explicitly required [5, 12, 01:02:08].
*   **Relevant Components:** The UI from the `llm-movieagent` repository is not relevant for our project, but the data import logic (in `ingest.py`) and agentic RAG code (in the `neo4j-semantic-layer` package) are highly relevant and should be adapted for our implementation.

### Graph Database Integration
*   **Database Choice:** The integration must be performed with the **Neo4j graph database** [2, 5].
*   **Accessibility for Testing:** For development and testing purposes, utilize the **free cloud version of Neo4j named "AuraDB,"** which does not require credit card details for access [11, 12, 00:37:46, 00:39:34].
*   **Workflow Adaptation:** Crucially, within the chosen self-RAG (/Users/spolischook/www/langgraph/docs/docs/tutorials/rag/langgraph_self_rag.ipynb) workflow, the **vector database component must be replaced** with direct calls to the Neo4j graph database for retrieval [12, 00:20:59, 00:23:50].

### LangGraph Workflow and Agentic Principles
*   The notebook should demonstrate the **agentic self-RAG workflow**'s capabilities, including its validation and retry mechanisms. This allows the system to rephrase user questions and refine answers based on internal validity checks, enhancing the overall reliability and accuracy of responses [12, 00:19:36, 00:30:07].

### General Requirements and Simplicity
*   **Simplicity is Paramount:** The entire project should be designed to be as **simple as possible**, deliberately avoiding unnecessary complexity and excessive detail [12, 00:08:38, 00:09:20].
*   **No Docker for LangGraph:** For integration with LangGraph, a **very straightforward approach is required, explicitly without the use of Docker containers** [12, 00:58:43].
*   **Environment Variables:** Any necessary API keys (e.g., for LLMs or the Neo4j cloud service) should be securely stored and accessed via **environment variables** [7, 12, 00:03:59].
*   **The overall structure** must be aligned with the other LangGraph tutorials in docs/docs/tutorials/rag

### Expected Output
Provide the complete Python code for a Jupyter notebook. Each step within the notebook should be accompanied by clear, concise explanations, covering everything from data loading and interaction with the Neo4j graph database to a full demonstration of the agentic self-RAG workflow. Use other tutorials in docs/docs/tutorials/rag as a reference.

### References
1. [Tomaz Bratanic's GraphRAG implementation](https://github.com/tomasonjo/llm-movieagent)
2. [Neo4j AuraDB Free](https://neo4j.com/pricing/)
3. [LangGraph](https://langchain-ai.github.io/langgraph/)
4. [Microsoft GraphRAG](https://github.com/microsoft/graphrag/blob/main/examples_notebooks/community_contrib/neo4j/graphrag_import_neo4j_cypher.ipynb)