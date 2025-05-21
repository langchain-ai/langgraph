from states import UserInputState, QuestionState, DataState, AnswerState
from bigquery_tools import execute_sql_query, get_table_schema, list_tables # Import BQ tools
from typing import List # For type hinting

def generate_questions_agent(state: UserInputState) -> QuestionState:
    """
    Generates questions based on user input.
    Placeholder implementation.
    """
    # For now, return a static list of questions
    return QuestionState(questions=["What is the capital of France?", "How does photosynthesis work?"])

def retrieve_data_agent(state: QuestionState) -> DataState:
    """
    Retrieves data based on the generated questions.
    Placeholder implementation.
    """
    # Get "available" tables and schema for the first table (simulation)
    simulated_tables_response = list_tables()
    simulated_tables: List[str] = simulated_tables_response.get('tables', [])

    if not simulated_tables:
        # No tables found, cannot answer
        return DataState(retrieved_data="No tables found in the simulated BigQuery environment.", can_answer=False)

    # For simplicity, use the first table for schema checks and queries
    # In a real scenario, logic would be more complex to choose the right table.
    primary_table_for_simulation = simulated_tables[0]
    simulated_schema_response = get_table_schema(primary_table_for_simulation)
    simulated_schema: dict = simulated_schema_response.get('schema', {})
    simulated_schema_columns: List[str] = list(simulated_schema.keys())

    retrieved_data_for_all_questions = []
    any_question_answerable = False

    for q_text in state.questions:
        q_text_lower = q_text.lower() # For case-insensitive matching
        question_seems_answerable = False

        # Simple check: does the question mention a known table name (case-insensitive)?
        if any(table.lower() in q_text_lower for table in simulated_tables):
            question_seems_answerable = True
        
        # Simple check: does the question mention a known column name (case-insensitive)?
        if not question_seems_answerable: # only check columns if table check failed
            if any(col.lower() in q_text_lower for col in simulated_schema_columns):
                question_seems_answerable = True
        
        if question_seems_answerable:
            # Simulate a query for this question
            # In a real scenario, an LLM would generate this query
            # Using the primary_table_for_simulation and its first column for simplicity
            first_column_name = simulated_schema_columns[0] if simulated_schema_columns else "unknown_column"
            simulated_query = f"SELECT * FROM {primary_table_for_simulation} WHERE {first_column_name} CONTAINS '{q_text[:20]}...'" # Example query
            
            print(f"RetrieveDataAgent: Attempting simulated query: {simulated_query} for question: {q_text}")
            query_result = execute_sql_query(simulated_query)
            
            # Store the actual result from the BQ tool, which is a dict like {"result": [...]}
            retrieved_data_for_all_questions.append({
                "question": q_text,
                "query_attempted": simulated_query,
                "data": query_result.get("result", "No data from query")
            })
            any_question_answerable = True # Mark that at least one question could be addressed
        else:
            retrieved_data_for_all_questions.append({
                "question": q_text,
                "query_attempted": None,
                "data": f"Could not find relevant table/column for question in simulated environment: {q_text}"
            })

    if any_question_answerable:
        return DataState(retrieved_data=retrieved_data_for_all_questions, can_answer=True)
    else:
        return DataState(retrieved_data="No relevant data found for any question based on simulated table/column checks.", can_answer=False)

def generate_answer_agent(state: DataState) -> AnswerState:
    """
    Generates an answer based on the retrieved data.
    Placeholder implementation.
    """
    # For now, return a static answer
    if state.can_answer:
        return AnswerState(answer="The retrieved data suggests the answer is 42.")
    else:
        return AnswerState(answer="I could not find enough information to answer the questions.")

def cannot_answer_agent(state: DataState) -> AnswerState:
    """
    Generates a specific message when the agent cannot answer.
    """
    # This agent is called when retrieve_data_agent sets can_answer to False
    return AnswerState(answer="Sorry, I cannot answer the question with the available data.")
