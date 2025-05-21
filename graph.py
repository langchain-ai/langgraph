from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

# Import agent functions and state models
from agents import generate_questions_agent, retrieve_data_agent, generate_answer_agent, cannot_answer_agent
from states import UserInputState, QuestionState, DataState, AnswerState

# Define the aggregate State TypedDict for the graph
class State(TypedDict):
    user_input: str
    questions: List[str]
    retrieved_data: Any
    can_answer: bool
    answer: str
    # Potentially, we can add intermediate states or error states here later

class Workflow(StateGraph):
    def __init__(self):
        super().__init__(State)

        # Add agent nodes to the graph
        # For now, we are adding the agent functions directly.
        # We might need to wrap them later if their input/output signatures
        # are not directly compatible with what StateGraph expects
        # (i.e., taking the whole State dict and returning a partial State dict).
        self.add_node("question_generator", generate_questions_agent)
        self.add_node("data_retriever", retrieve_data_agent)
        self.add_node("answer_generator", generate_answer_agent)
        self.add_node("cannot_answer_node", cannot_answer_agent) # New node

        # Define the entry point
        self.set_entry_point("question_generator")

        # Define the regular edges
        self.add_edge("question_generator", "data_retriever")

        # Define the conditional edge from data_retriever
        self.add_conditional_edges(
            "data_retriever",
            self.should_generate_answer,
            {
                "generate_answer": "answer_generator",
                "cannot_answer": "cannot_answer_node"
            }
        )

        # Define edges to the end
        self.add_edge("answer_generator", END)
        self.add_edge("cannot_answer_node", END)

    def should_generate_answer(self, state: State) -> str:
        """
        Determines the next step based on whether the agent can answer.
        The 'can_answer' field is expected to be set by the 'data_retriever' node.
        """
        if state.get("can_answer"): # Check if can_answer is True
            return "generate_answer"
        else:
            return "cannot_answer"

# Instantiate the Workflow and compile it
# This will likely fail if the agent functions' signatures are not
# (state: State) -> Partial[State], but the task is to define the graph structure.
# We will address agent compatibility in a later step.
app = Workflow().compile()

# To make this file runnable for basic inspection, though execution will fail
# if agent signatures are not yet adapted:
if __name__ == "__main__":
    print("Graph compiled. 'app' instance is ready.")
    print("Attempting to run the graph with a sample input...\n")

    user_input_string = "Tell me about orders from last week, and also what is the schema of the customers table?"
    initial_state = {"user_input": user_input_string}

    print(f"--- Initial User Input ---")
    print(initial_state['user_input'])
    print("\n=====================\n")

    print("--- Streaming Events ---")
    final_output = None # To store the result from invoke later
    try:
        for event_count, event in enumerate(app.stream(initial_state)):
            for key, value in event.items():
                print(f"--- Event {event_count + 1}: Node '{key}' Output ---")
                print(f"Raw output: {value}") # Print the raw output of the node
                
                # If the value is a Pydantic model, it's helpful to see its dict representation
                if hasattr(value, 'model_dump'):
                    print(f"Pydantic model dump: {value.model_dump()}")
                elif isinstance(value, dict): # Or if it's already a dict (like the overall state)
                     print(f"State snapshot: {value}")


            print("\n=====================\n")
        
        # After streaming, get the final accumulated state
        # Note: The `final_output` from `invoke` is usually more direct for just the end state.
        # The stream provides intermediate steps. Let's use invoke to get the final state cleanly.
        print("--- Invoking Graph for Final Output ---")
        final_output = app.invoke(initial_state)
        print(f"\n--- Final Output (from invoke) ---")
        if final_output:
            if final_output.get("answer"):
                 print(f"Answer: {final_output['answer']}")
            else:
                print(f"Full final state: {final_output}")
        else:
            print("No final output from invoke.")

    except Exception as e:
        print(f"Error during graph execution: {e}")
        print("This might be due to incompatible agent signatures or issues within agent logic.")
        import traceback
        traceback.print_exc()
