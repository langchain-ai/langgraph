# Self-GraphRAG Implementation

**Generate a Python Jupyter notebook for an agentic GraphRAG implementation using LangGraph**

### Project Goal
The primary goal is adapt an existing self-RAG workflow from a LangGraph Jupyter notebook (source file: `docs/docs/tutorials/rag/langgraph_self_rag.ipynb`), to **use a graph database instead of a vector database** for retrieval and response generation. This implementation should fully leverage the robust principles of agentic RAG, including built-in validation, retry mechanisms, the ability to rephrase user questions, and internal sanity checks. Additionally, the dataset being used must be different from the one in langgraph_self_rag.ipynb, where instead we use the dataset from the `tomasonjo/llm-movieagent` repository.

### Leveraging Data and Structure from `tomasonjo/llm-movieagent` repository
*   **Data Source:** Use the **movie dataset from the `tomasonjo/llm-movieagent` GitHub repository** https://github.com/tomasonjo/llm-movieagent. This repository implements a "semantic layer on top of a graph database", using Neo4j to store information about actors, movies, and their ratings. The dataset is based on the MovieLens dataset. The dataset is downloaded in the file `api/ingest.py`.
*   **Data Structure:** The data is already well-structured with defined schemas for **movies, users, persons (such as actors and directors), and genres**. This existing structure should serve as the foundation for the graph database.
*   **Existing Tools:** The `llm-movieagent` project already provides agents with a suite of robust tools for interacting with the graph database, including an **Information Tool, Recommendation Tool, and Memory Tool**.
*   **Data Ingestion:** The data can be easily populated into the graph database from **CSV files**, with data preparation requiring approximately 60 lines of code. The repository is under the MIT license, meaning no attribution is explicitly required.
*   **Relevant Components:** The UI component from the `llm-movieagent` repository is not relevant for our project, but the data import logic (in `ingest.py`) and agentic RAG code (in the `neo4j-semantic-layer` package) are highly relevant and should be adapted for our implementation.

### Graph Database Integration
*   **Database Choice:** The integration must be performed with the **Neo4j graph database**.
*   **Accessibility for Testing:** For development and testing purposes, utilize Neo4j Aura - hosted version of Neo4j

### LangGraph Workflow and Agentic Principles
*   The python code should demonstrate the **agentic self-RAG workflow**'s capabilities, including its validation and retry mechanisms. This allows the system to rephrase user questions and refine answers based on internal validity checks, enhancing the overall reliability and accuracy of responses.

### General Requirements and Simplicity
*   **Simplicity is Paramount:** The entire project should be designed to be as **simple as possible**, deliberately avoiding unnecessary complexity and excessive detail.
*   **Environment Variables:** Any necessary API keys (e.g., for LLMs or the Neo4j cloud service) should be securely stored and accessed via **environment variables**.

### Expected Output
Provide the complete documented Python code (.py, not .ipynb). Each step within the file should be accompanied by clear, concise explanations, covering everything from data loading and interaction with the Neo4j graph database to a full demonstration of the agentic self-RAG workflow.

### Definition of Done
1. The code should be able to run without errors and warnings

### Deprecation Warning
1. The class `Neo4jGraph` was deprecated in LangChain 0.3.8 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-neo4j package and should be used instead. To use it run `pip install -U :class:`~langchain-neo4j` and import as `from :class:`~langchain_neo4j import Neo4jGraph``.

### References
1. [Tomaz Bratanic's GraphRAG implementation](https://github.com/tomasonjo/llm-movieagent)
2. [LangGraph](https://langchain-ai.github.io/langgraph/)
