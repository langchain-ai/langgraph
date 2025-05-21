# Documentation for LLM Interaction with Agentic System

## 1. System Purpose
   - This system is designed to process natural language user inputs to answer questions by retrieving and reasoning over data stored in BigQuery tabular databases.
   - It operates as a graph of interconnected agents (which are themselves LLM-driven or rule-based components). The orchestration of these agents is managed by LangGraph.

## 2. Overall Interaction Flow (Input/Output)
   - **System Input**:
     - Type: Natural language string.
     - Description: Represents the end-user's query or request. This is the initial entry point into the system.
     - Example: "What were the total sales for product X last month?"
     - Handled by: `UserInputState(user_input: str)`, which is the expected structure for the input to the first agent.

   - **System Final Output**:
     - Type: String.
     - Description: The system's final answer to the user's query, or a message indicating inability to answer. This is the culmination of the agentic workflow.
     - Produced by: `AnswerState(answer: str)`. This state is populated by either the `generate_answer_agent` (if the query is answerable) or the `cannot_answer_agent` (if the query is not answerable).

## 3. Core Agent Roles and Data States
   This system uses a sequence of agents, each transforming the state of the system. If you, as an LLM, are to act as one of these agents or interpret their outputs, understanding their specific roles and data contracts is crucial.

   - **Agent 1: Question Generator (`generate_questions_agent`)**
     - Input: `UserInputState(user_input: str)`
     - Output: `QuestionState(questions: List[str])`
     - Role: To parse the initial `user_input` (a natural language query) and transform it into one or more specific, answerable questions suitable for data retrieval. This often involves disambiguation, clarification, or breaking down a complex request.
     - Current Implementation: Placeholder. It currently generates a static, predefined list of questions regardless of the `user_input`.
     - *Note for LLM*: If you are this agent, your primary goal is to analyze the user's raw text. You need to identify the core intents and entities, then rephrase these into clear, targeted questions that can be individually addressed by a data retrieval process. Consider edge cases like ambiguous queries or compound questions.

   - **Agent 2: Data Retriever (`retrieve_data_agent`)**
     - Input: `QuestionState(questions: List[str])`
     - Output: `DataState(retrieved_data: Any, can_answer: bool)`
     - Role:
       1. To understand the list of questions provided in `QuestionState`.
       2. To interact with a BigQuery database (currently simulated via placeholder functions in `bigquery_tools.py`). This involves formulating and "executing" SQL queries or other data retrieval methods based on the questions.
       3. To critically evaluate the data obtained from BigQuery and determine if it can adequately answer the posed questions. This assessment is captured in the `can_answer: bool` flag.
       4. To fetch and structure the relevant data into the `retrieved_data` field. The format of `retrieved_data` is currently a list of simulated query results (dictionaries).
     - Current Implementation: Uses placeholder tools that simulate BigQuery interaction. It makes a rule-based decision on `can_answer` based on simple keyword matching between questions and simulated table/column names.
     - *Note for LLM*: If you are this agent, you need strong capabilities in translating natural language questions into precise SQL queries. This requires understanding of SQL syntax, database schemas (even if simulated initially), and how to map question entities to database entities. After "retrieving" data, you must assess its relevance and completeness to answer the questions. The `bigquery_tools.py` file shows the current interaction patterns (e.g., `execute_sql_query`, `get_table_schema`, `list_tables`).

   - **Agent 3: Answer Generator (`generate_answer_agent`)**
     - Input: `DataState(retrieved_data: Any, can_answer: bool)` (this agent is specifically invoked when `can_answer` is `True`).
     - Output: `AnswerState(answer: str)`
     - Role: To synthesize a final, human-readable answer from the `retrieved_data` provided in `DataState`. This answer should directly address the original questions posed by the `generate_questions_agent`.
     - Current Implementation: Placeholder. It generates a static, predefined answer string if `can_answer` is true.
     - *Note for LLM*: If you are this agent, your task is to transform potentially structured or raw data from `retrieved_data` along with the context of the original questions, into a coherent, informative, and natural language response. This may involve summarization, explanation, and formatting.

   - **Auxiliary Agent: Cannot Answer (`cannot_answer_agent`)**
     - Input: `DataState(retrieved_data: Any, can_answer: bool)` (this agent is specifically invoked when `can_answer` is `False`).
     - Output: `AnswerState(answer: str)`
     - Role: To inform the user clearly and politely that their question cannot be answered with the available data or current system capabilities.
     - Current Implementation: Returns a static, predefined message.
     - *Note for LLM*: While simpler, this agent's response should be empathetic and potentially offer reasons or next steps if possible (though the current implementation is static).

## 4. BigQuery Interaction (Simulated)
   - The system is designed with the goal of interacting with Google BigQuery. Currently, this interaction is simulated by placeholder tools defined in `bigquery_tools.py`:
     - `execute_sql_query(query: str) -> dict`: Simulates running a SQL query and returns a placeholder result.
     - `get_table_schema(table_name: str) -> dict`: Simulates fetching the structure (schema) of a given table.
     - `list_tables(dataset_name: str) -> dict`: Simulates listing available tables within a dataset.
   - An LLM acting as the `Data Retriever` agent would need to generate SQL queries that are appropriate for the questions it receives and the (eventually real) database schemas it might discover using `get_table_schema` or `list_tables`.

## 5. Key Decision Point: `DataState.can_answer`
   - The `data_retriever` agent plays a critical role in the workflow by setting the `can_answer: bool` flag in the `DataState`.
   - If `DataState.can_answer` is `True`, the system proceeds to the `generate_answer_agent` to formulate an answer from the retrieved data.
   - If `DataState.can_answer` is `False`, the system diverts to the `cannot_answer_agent` to inform the user appropriately.
   - An LLM tasked with the `data_retriever` role must accurately assess whether the data obtained (simulated or real) is sufficient and relevant to answer the input questions. This is a crucial reasoning step.

## 6. Extending or Integrating with this System
   - **Replacing a Placeholder Agent with an Advanced LLM**:
     - Identify the agent function you want to upgrade in `agents.py` (e.g., `generate_questions_agent`).
     - Develop your LLM-powered logic for that agent's role.
     - Ensure your new LLM function strictly adheres to the input and output `State` Pydantic models defined for that agent (e.g., takes `UserInputState`, returns `QuestionState`). This ensures compatibility with the LangGraph workflow.
   - **Connecting to a Real BigQuery Instance**:
     - The functions in `bigquery_tools.py` are the designated points for real database interaction.
     - Update these functions to use the `google-cloud-bigquery` Python library.
     - For example, `execute_sql_query` would use `bigquery_client.query(query_string).result()` or `.to_dataframe()`.
     - `get_table_schema` would use `bigquery_client.get_table(table_ref).schema`.
     - `list_tables` would use `bigquery_client.list_tables(dataset_ref)`.
     - Maintain the existing function signatures (parameters and return type hints) if possible to minimize changes in the `retrieve_data_agent` that calls them.
     - Ensure proper GCP authentication and error handling for API calls.

This documentation should provide a foundational understanding for an LLM to effectively integrate with, operate as a component of, or be developed to enhance this agentic system.

## 7. A2A Protocol Integration (google/A2A)
The Agent to Agent (A2A) protocol provides a standardized way for software agents to communicate with each other, regardless of their underlying implementation. You can find more information about the A2A protocol at its official documentation page: [https://google.github.io/A2A/](https://google.github.io/A2A/) and its GitHub repository: [https://github.com/Dantemerlino/A2A](https://github.com/Dantemerlino/A2A).

This section describes how this LangGraph-based agentic system and its components can be integrated into a broader A2A ecosystem, both by exposing its agents as A2A services and by consuming external A2A services.

### Exposing System Agents as A2A Services
Each core agent within this LangGraph system (`Question Generator`, `Data Retriever`, `Answer Generator`) can be conceptualized as an A2A Server or Remote Agent. This involves wrapping their functionality within an A2A-compliant API, typically HTTP-based. The agent's capabilities and how to interact with it are described in an `AgentCard`.

Below are hypothetical `AgentSkill` definitions that would be part of each agent's `AgentCard`:

-   **Question Generator Agent as A2A Service**:
    -   `AgentSkill.id`: "generate-questions-from-user-input"
    -   `AgentSkill.name`: "User Input to Question Converter"
    -   `AgentSkill.description`: "Takes general user textual input and refines it into specific, answerable questions for data retrieval."
    -   `AgentSkill.inputModes`: `["text/plain"]` (maps from `UserInputState.user_input`)
    -   `AgentSkill.outputModes`: `["application/json"]` (represents `QuestionState.questions` as a JSON list of strings within an A2A `DataPart`)
    -   Relevant A2A RPC method: `message/send` (for synchronous request/response).
    -   Mapping `UserInputState` to A2A `Message`: The `user_input: str` from `UserInputState` would be placed in a `TextPart` within the `parts` array of the input A2A `Message` sent by the A2A client.
        ```json
        // Example A2A Message input for Question Generator
        {
          "parts": [
            { "text": "Tell me about orders and customer schemas." }
          ]
        }
        ```
    -   Mapping `QuestionState` from A2A `Message`: The agent's response A2A `Message` would contain a `DataPart`. The `data` field of this `DataPart` would be a JSON object corresponding to `QuestionState`.
        ```json
        // Example A2A Message response from Question Generator
        {
          "parts": [
            { "data": { "questions": ["What are the orders?", "What is the customer schema?"] } }
          ]
        }
        ```

-   **Data Retriever Agent as A2A Service**:
    -   `AgentSkill.id`: "retrieve-bigquery-data"
    -   `AgentSkill.name`: "BigQuery Data Retriever and Assessor"
    -   `AgentSkill.description`: "Receives questions (as JSON), queries a (simulated) BigQuery database, determines if questions are answerable, and returns retrieved data."
    -   `AgentSkill.inputModes`: `["application/json"]` (maps from `QuestionState.questions` via an A2A `DataPart`)
    -   `AgentSkill.outputModes`: `["application/json"]` (represents the full `DataState` via an A2A `DataPart`)
    -   Relevant A2A RPC method: `message/send` (synchronous, though could be `stream/open` if queries are long-running and intermediate results are desired).
    -   Mapping `QuestionState` to A2A `Message`: The input A2A `Message` from the client (e.g., the LangGraph orchestrator) would contain a `DataPart` with the JSON representation of `QuestionState`.
        ```json
        // Example A2A Message input for Data Retriever
        {
          "parts": [
            { "data": { "questions": ["What are the orders?", "What is the customer schema?"] } }
          ]
        }
        ```
    -   Mapping `DataState` from A2A `Message`: The agent's response A2A `Message` would contain a `DataPart` whose `data` field is the JSON representation of `DataState`.
        ```json
        // Example A2A Message response from Data Retriever
        {
          "parts": [
            { "data": { "retrieved_data": [{"query_result": "..."}], "can_answer": true } }
          ]
        }
        ```

-   **Answer Generator Agent as A2A Service**:
    -   `AgentSkill.id`: "generate-answer-from-data"
    -   `AgentSkill.name`: "Answer Synthesizer"
    -   `AgentSkill.description`: "Takes retrieved data (as JSON) and synthesizes a final textual answer."
    -   `AgentSkill.inputModes`: `["application/json"]` (maps from `DataState` via an A2A `DataPart`)
    -   `AgentSkill.outputModes`: `["text/plain"]` (represents `AnswerState.answer` as a `TextPart`)
    -   Relevant A2A RPC method: `message/send`.
    -   Mapping `DataState` to A2A `Message`: Input A2A `Message` from client contains a `DataPart` with the JSON for `DataState`.
        ```json
        // Example A2A Message input for Answer Generator
        {
          "parts": [
            { "data": { "retrieved_data": [{"query_result": "..."}], "can_answer": true } }
          ]
        }
        ```
    -   Mapping `AnswerState` from A2A `Message`: The agent's response A2A `Message` would contain a `TextPart` for the `answer` string from `AnswerState`.
        ```json
        // Example A2A Message response from Answer Generator
        {
          "parts": [
            { "text": "The answer based on the data is..." }
          ]
        }
        ```

A full `AgentCard` for each of these A2A services would also include other mandatory and optional fields like `id` (for the agent itself), `name`, `description`, `url` (the HTTP endpoint where this agent is hosted and listening for A2A requests), `capabilities` (listing its skills), `securitySchemes`, `inputModes`, `outputModes` (for the agent overall, often derived from its skills), etc., as per the A2A specification.

### Consuming External A2A Agents
The LangGraph workflow itself can be extended to act as an A2A Client, enabling it to call other external agents that expose A2A-compliant interfaces.

-   **Generic A2A Client Node**: A new, generic LangGraph node could be created (e.g., `a2a_generic_client_node`).
    -   This node would be designed to take parameters such as:
        -   The URL of the target external A2A agent's `AgentCard` (to discover its capabilities and endpoint) or directly its service URL.
        -   The specific `AgentSkill.id` to invoke on the external agent.
        -   The input A2A `Message` payload, constructed from the current LangGraph state.
    -   Internally, this node would use an HTTP client library to make the A2A call (e.g., POST to the external agent's `message/send` endpoint).
    -   It would then parse the A2A `Message` response from the external agent and transform its `parts` (e.g., `TextPart`, `DataPart`) back into a structure that can update the LangGraph state and be used by subsequent nodes in the workflow.
-   **Use Case**: This would allow replacing one of the current system's internal agents (e.g., the placeholder `generate_questions_agent`) with a call to an external, potentially more specialized or powerful, A2A agent that offers that capability.

### General Considerations for A2A
-   **Task Management**: For interactions that are more complex than a single request-response, or for long-running operations, A2A `Task` objects (`task/create`, `task/getStatus`, etc.) could be used to manage the lifecycle of the interaction. The current system's synchronous nature aligns well with `message/send`.
-   **Data Serialization**: The Pydantic state models (`UserInputState`, `QuestionState`, `DataState`, `AnswerState`) currently used in this system provide a strong foundation for the JSON structures within A2A `DataPart` objects. Pydantic's `.model_dump_json()` and `.model_validate_json()` methods can be directly used for serialization and deserialization when constructing and parsing A2A `Message` parts containing JSON data.
-   **Error Handling**: A robust A2A integration would require careful error handling, including parsing A2A-defined error responses and translating them into exceptions or error states within the LangGraph workflow.I have successfully appended the new sections regarding A2A Protocol Integration to the `LLM_DOCUMENTATION.md` file.

The changes included:
1.  Added Major Section: `## 7. A2A Protocol Integration (google/A2A)` with an introduction and links.
2.  Added Subsection: `"Exposing System Agents as A2A Services"` which details:
    *   Conceptualizing agents as A2A Servers.
    *   Hypothetical `AgentSkill` definitions for `Question Generator`, `Data Retriever`, and `Answer Generator`, including `id`, `name`, `description`, `inputModes`, `outputModes`, relevant RPC methods, and mappings of current Pydantic states to/from A2A `Message` parts with JSON examples.
    *   Mention of other `AgentCard` fields.
3.  Added Subsection: `"Consuming External A2A Agents"` explaining:
    *   How LangGraph can act as an A2A client.
    *   A proposal for a generic LangGraph node for this purpose.
    *   The use case of replacing an internal agent with an external A2A agent.
4.  Added Subsection: `"General Considerations for A2A"` covering:
    *   The potential use of A2A `Task` objects.
    *   The utility of Pydantic states for A2A `DataPart` JSON serialization.
    *   A note on error handling.

The language used is intended to be LLM-friendly, and the content focuses on structural and "how-to" aspects of A2A integration as requested.
