# Agentic System for User Input Processing and BigQuery Interaction

## 1. Project Overview
This system is designed to take user input, convert it into specific questions, query a BigQuery database (simulated for now), and then generate a comprehensive answer based on the findings. The entire workflow is orchestrated using LangGraph, allowing for a modular, stateful, and observable agentic design.

## 2. Architecture
The system employs a multi-agent setup where each agent has a distinct role in the process:

-   **Agent 1 (Question Generator - `generate_questions_agent`)**:
    -   **Role**: Processes the initial raw user input and breaks it down into one or more clear, answerable questions.
    -   **Current state**: Placeholder. It currently returns a static, predefined list of questions, irrespective of the actual user input.
-   **Agent 2 (Data Retriever - `retrieve_data_agent`)**:
    -   **Role**: Takes the generated questions, interacts with the BigQuery tools to find relevant data, and determines if the questions can be adequately answered based on the retrieved information.
    -   **Current state**: Uses placeholder BigQuery tools (`bigquery_tools.py`) that simulate API calls without actually connecting to a database. It performs simple keyword matching (based on question content and simulated table/column names) to decide if a question is answerable and what "data" to return.
-   **Agent 3 (Answer Generator - `generate_answer_agent`)**:
    -   **Role**: Synthesizes the data retrieved by the Data Retriever and formulates a final, human-readable answer to the user's original query.
    -   **Current state**: Placeholder. It returns a static answer if data is provided and the `can_answer` flag is true.
-   **`cannot_answer_agent`**:
    -   **Role**: Specifically handles scenarios where the Data Retriever (`retrieve_data_agent`) determines that it cannot find sufficient information to answer the questions. It provides a predefined message indicating this inability.

The system's codebase is organized into the following key Python files:

-   `states.py`: Defines Pydantic models (`UserInputState`, `QuestionState`, `DataState`, `AnswerState`) that represent the structure of data as it flows through the system. These models ensure type safety and provide clear data contracts between agents.
-   `agents.py`: Contains the Python functions that implement the logic for each of the agents described above. This is where the core processing for each step resides.
-   `bigquery_tools.py`: Provides placeholder functions that simulate interactions with Google BigQuery (e.g., `execute_sql_query`, `get_table_schema`, `list_tables`). These tools mimic the expected interface of actual BigQuery operations.
-   `graph.py`: Defines the LangGraph workflow. This file includes:
    -   The overall state definition (`State` TypedDict) that the graph operates on.
    -   Instantiation of the `StateGraph`.
    -   Registration of each agent function as a node within the graph.
    -   Definition of edges that connect these nodes, dictating the sequence of operations.
    -   Conditional logic for routing the workflow based on the output of the `data_retriever` node (specifically, the `can_answer` flag).

## 3. Setup and Installation

### Prerequisites
-   Python 3.8+ (or a version compatible with LangGraph and other dependencies).
-   `pip` (Python package installer).
-   `git` (for cloning the repository).

### Installation
1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository_url> 
    cd <repository_directory>
    ```
    (Replace `<repository_url>` and `<repository_directory>` with the actual URL and directory name if applicable. If you're working from an existing checkout, you can skip this step.)

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages** using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    This file includes `langgraph`, `pydantic`, and `requests` which are necessary for running the main agentic system and the LLM Connection Tester utility.

*Note on BigQuery*: For actual BigQuery integration (which is currently placeholder-based), you will also need the `google-cloud-bigquery` library. This can be installed via `pip install google-cloud-bigquery` when you're ready to implement that functionality. This package is not included in `requirements.txt` by default to keep the initial setup lightweight for users who only want to explore the placeholder logic.

## 4. How the System Works (LangGraph Flow)
The LangGraph workflow orchestrates the interaction between the agents in a defined, stateful sequence:

1.  **Initial State**: The process begins with a `UserInputState`, which contains the raw input string from the user. This input is part of the overall `State` dictionary that the graph manages.
2.  **Question Generation**: The `question_generator` node is the entry point. It processes the `user_input` from the state and populates the `questions` field in the graph's state with a list of questions.
3.  **Data Retrieval**: The flow then moves to the `data_retriever` node. This agent uses the list of `questions` and interacts with the (currently simulated) BigQuery tools. It populates the `retrieved_data` field and sets the `can_answer` boolean flag in the state.
4.  **Conditional Routing**: Based on the value of the `can_answer` flag set by `data_retriever`:
    -   If `can_answer` is `True`, the graph transitions to the `answer_generator` node.
    -   If `can_answer` is `False`, the graph transitions to the `cannot_answer_node`.
5.  **Answer Generation / Handling Inability to Answer**:
    -   If the `answer_generator` node is called, it uses the `retrieved_data` to formulate the final `answer`.
    -   If the `cannot_answer_node` is called, it provides a predefined `answer` indicating that the query could not be satisfied with the available data.
6.  **Final State**: The graph execution concludes, and the final state (which includes the `answer` from either `answer_generator` or `cannot_answer_node`) is available.

## 5. Running the System
To run the system with its current placeholder logic and observe its behavior:

1.  Navigate to the directory containing the Python files (`states.py`, `agents.py`, `bigquery_tools.py`, `graph.py`).
2.  Execute the `graph.py` script directly from your terminal:
    ```bash
    python graph.py
    ```
This command will trigger the test execution block defined within the `if __name__ == "__main__":` section of `graph.py`. This block initializes the graph with a sample user input.

**Expected Output**:
When you run the script, you will see console output demonstrating:
-   The initial sample user input provided to the system.
-   Logs from the placeholder BigQuery tools as they are "called" (e.g., "Simulating SQL query execution...", "Simulating schema retrieval...").
-   Intermediate outputs from each agent node as it processes the data. This is made visible by the `app.stream()` logging in `graph.py`, showing the state changes at each step.
-   The final answer generated by the system, which will come from either the `answer_generator` (if `can_answer` was true) or the `cannot_answer_node` (if `can_answer` was false).

## LLM Connection Tester Utility (`llm_connection_tester.py`)

This repository includes a utility script, `llm_connection_tester.py`, designed to help you test and confirm your connection to a local or remote LLM endpoint, particularly one that mimics the OpenAI API structure (like those provided by LM Studio or similar tools).

### Purpose
The main goal of this utility is to ensure that your Python environment can successfully communicate with your chosen LLM before you integrate it into more complex agentic systems. It allows you to:
- Specify the LLM's API endpoint URL.
- Optionally add custom HTTP headers (e.g., for authorization).
- Send a test request with a customizable JSON body.
- View the LLM's direct response or any error messages encountered.

### Prerequisites
- Python 3.x
- `tkinter` (usually included with standard Python installations).
- `requests` library (installed via `requirements.txt`).

### How to Use
1.  **Run the script**:
    ```bash
    python llm_connection_tester.py
    ```
2.  **GUI Interface**:
    *   **LLM API Endpoint URL**: Enter the full HTTP/HTTPS URL for your LLM's chat completions endpoint. For many local LLMs (like LM Studio), this might be something like `http://localhost:1234/v1/chat/completions` or `http://127.0.0.1:1234/v1/chat/completions`.
    *   **HTTP Headers (JSON, optional)**: If your LLM endpoint requires specific headers (e.g., an API key for a hosted service, or a different `Content-Type`), enter them here in JSON format. For example: `{"Authorization": "Bearer YOUR_API_KEY"}`. It defaults to `{"Content-Type": "application/json"}` which is then merged with any user-provided headers.
    *   **Request Body (JSON)**: This field is pre-populated with a sample request body compatible with OpenAI-like chat completion APIs. You can modify this JSON to change the model, prompt, temperature, or other parameters as needed for your test.
    *   **Test Connection Button**: Click this to send the request to the LLM.
    *   **Response / Status Pane**: This area will display:
        - Confirmation of the request details being sent.
        - "Connection Successful!" followed by the LLM's JSON response if the request is successful.
        - Detailed error messages if any issues occur (e.g., connection errors, HTTP errors, timeouts, JSON parsing failures).

### Example (LM Studio)
- If your LM Studio server is running on `http://localhost:1234`:
    - **LLM API Endpoint URL**: `http://localhost:1234/v1/chat/completions`
    - **HTTP Headers**: Usually not needed for a default LM Studio setup, so you can leave this blank unless you've configured API keys.
    - **Request Body**: You can use the default or modify the `model` field if your loaded model has a specific identifier recognized by LM Studio (often not strictly required by LM Studio; it uses the model loaded in the UI).
    - Click "Test Connection". You should see a response from the model.

### Troubleshooting
*   **"Connection Error"**:
    - Ensure your LLM server (e.g., LM Studio) is running.
    - Verify the IP address and port in the URL are correct and accessible from where you're running the script.
    - Check your firewall settings if you're trying to access an endpoint on a different machine or a virtual machine.
    - If using an ngrok URL, ensure the ngrok tunnel is active.
*   **"HTTP Error" (e.g., 401, 403, 404, 500)**:
    - **401/403 (Unauthorized/Forbidden)**: Your LLM endpoint might require an API key or authentication token. Provide it in the "HTTP Headers" field (e.g., `{"Authorization": "Bearer YOUR_KEY"}`).
    - **404 (Not Found)**: Double-check the path part of your URL (e.g., `/v1/chat/completions`). It might be incorrect for your specific LLM server.
    - **500 (Internal Server Error)**: The LLM server itself encountered an error. Check the LLM server's logs.
*   **"Timeout Error"**:
    - Your LLM might be taking too long to respond. You can try increasing the timeout in the script (currently 20 seconds), or check if the LLM is overloaded.
    - Network connectivity issues can also cause timeouts.
*   **"Invalid JSON in Headers/Request Body"**: Ensure the text you've entered in these fields is valid JSON. Online JSON validators can help.
*   **"JSON Decode Error" (for response)**: The LLM responded, but its response was not valid JSON. The raw response text will be shown, which might give clues. This could indicate an issue with the LLM server's output or an unexpected HTML error page.

## 6. Development and Future Enhancements

-   **Placeholder Components**: It is crucial to understand that the current system relies heavily on placeholders for core functionality:
    -   **LLM Logic**: The agent functions within `agents.py` (specifically `generate_questions_agent` and `generate_answer_agent`) do not yet contain calls to Large Language Models (LLMs). Their current logic is hardcoded to return static values for simulation purposes.
    -   **BigQuery Tools**: The functions in `bigquery_tools.py` only simulate BigQuery interactions. They print messages and return static, hardcoded data without actually connecting to or querying any database.

-   **Integrating Real LLMs**:
    -   To make the agents intelligent and responsive to varied inputs, you will need to modify the respective agent functions in `agents.py`.
    -   For example, `generate_questions_agent` would need to use an LLM to parse the `user_input` string and generate a relevant list of questions. Similarly, `generate_answer_agent` would use an LLM to synthesize a coherent answer from the potentially complex `retrieved_data`.
    -   This typically involves using an LLM client library (such as LangChain's integrations with models from OpenAI, Anthropic, Google, etc.) to make API calls to the chosen language model.

-   **Integrating Real BigQuery**:
    -   The placeholder functions in `bigquery_tools.py` (`execute_sql_query`, `get_table_schema`, `list_tables`) must be replaced with actual calls to the Google Cloud BigQuery API.
    -   This involves:
        -   Initializing the `google.cloud.bigquery.Client` typically at the beginning of your tool functions or in a shared context.
        -   For `execute_sql_query`, you would replace the placeholder with code like `client.query(your_sql_query).to_dataframe()` or by iterating over the results using `client.query(your_sql_query).result()`.
        -   For `get_table_schema`, you would use `client.get_table(table_id).schema`.
        -   For `list_tables`, you would use `client.list_tables(dataset_id)`.
    -   Proper authentication (e.g., using Application Default Credentials or service account keys) and robust error handling for API calls will be necessary.

-   **Data Input**:
    -   The original problem statement mentioned, "These data will be provided later." Integrating real BigQuery (as detailed above) will become critical once these actual data sources, table structures, and schemas are defined and accessible. This will allow the `retrieve_data_agent` to work with live, meaningful data, making the entire system fully functional and capable of answering real user queries.

### Interoperability with A2A Protocol

The [Agent2Agent (A2A) Protocol](https://google.github.io/A2A/) (see also [Dantemerlino/A2A.git](https://github.com/Dantemerlino/A2A)) is an open standard for enabling AI agents to communicate and collaborate. This system can be enhanced for greater interoperability within an A2A ecosystem:

*   **Exposing Agents as A2A Services**: The core agents of this system (Question Generator, Data Retriever, Answer Generator) could be wrapped as A2A-compliant services. Each would publish an `AgentCard` describing its capabilities and expose A2A-standard JSON-RPC methods (e.g., `message/send`). This would allow other A2A-compliant agents to discover and utilize their specific skills.
*   **Consuming External A2A Agents**: The LangGraph workflow can be extended to act as an A2A client. A generic node could be developed to interact with external A2A agents, allowing this system to delegate tasks or leverage specialized capabilities from the broader A2A ecosystem.
*   **Enhanced Modularity**: Adopting A2A can make the overall architecture more modular, allowing individual components (agents) to be developed, deployed, and scaled independently while maintaining standardized communication interfaces.

For detailed information on how this system's components map to A2A concepts and how LLMs can interact with it in an A2A context, please refer to `LLM_DOCUMENTATION.md`.
