import sys
import os
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout, 
    QLabel,
    QTextEdit,
    QPushButton,
    QLineEdit,   
    QFileDialog,
    QSplitter,    # For resizable panes
    QFormLayout   # For performance metrics layout
)
from PySide6.QtCore import QThread, Signal, Qt # For Qt.Vertical, Qt.Horizontal
from dotenv import load_dotenv
import json 

# Attempt to import the LangGraph app and states
# This will be used by the GraphWorker
try:
    from graph import app as langgraph_app
    from states import UserInputState # UserInputState might not be directly used by worker if graph.py handles it
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"Error importing LangGraph components: {e}. Graph execution will be simulated.")
    langgraph_app = None
    UserInputState = None
    LANGGRAPH_AVAILABLE = False


# Expected BigQuery environment variables
EXPECTED_BQ_VARS = ["GOOGLE_APPLICATION_CREDENTIALS", "BIGQUERY_PROJECT_ID", "BIGQUERY_DATASET_ID"]


# --- GraphWorker QThread ---
class GraphWorker(QThread):
    new_event = Signal(object)  # Emits each event from app.stream()
    # finished_successfully = Signal(object) # Optional: for final app.invoke() result
    error_occurred = Signal(str) # Emits error messages
    execution_done = Signal()    # Signals that graph execution (success or fail) has concluded

    def __init__(self, user_input: str, env_vars: dict):
        super().__init__()
        self.user_input = user_input
        self.env_vars = env_vars # Though not directly used by graph.py yet, good to pass
        self._is_running = True

    def run(self):
        """Executes the LangGraph app."""
        if not LANGGRAPH_AVAILABLE:
            self.error_occurred.emit("LangGraph components (graph.py, states.py) not found or importable.")
            self.execution_done.emit()
            return

        try:
            # Here, we assume that os.environ has been appropriately set by load_dotenv
            # before this worker is instantiated if graph.py or its components rely on it.
            # The self.env_vars could be used to set os.environ within this thread if needed,
            # but direct use of os.environ by graph.py is simpler if .env is loaded globally.
            
            initial_state = {"user_input": self.user_input}
            
            # Stream events
            for event in langgraph_app.stream(initial_state):
                if not self._is_running: # Check if stop was requested
                    break
                self.new_event.emit(event) # Emit each event
            
            # If you also want to emit the final result from invoke separately:
            # final_output = langgraph_app.invoke(initial_state)
            # self.finished_successfully.emit(final_output)

        except Exception as e:
            self.error_occurred.emit(f"Error during graph execution: {str(e)}\n"
                                     f"Traceback: {e.__traceback__}") # Basic traceback info
        finally:
            self.execution_done.emit()

    def stop(self):
        self._is_running = False


class InteractiveLauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interactive Agent Launcher")
        self.setGeometry(100, 100, 1000, 800) # Increased window size for more panes
        
        self.loaded_env_file_path = None
        self.graph_worker = None 

        # --- Main Vertical Splitter (Top: Inputs, Bottom: Outputs) ---
        main_v_splitter = QSplitter(Qt.Vertical)
        self.setCentralWidget(main_v_splitter)

        # --- Top Widget (Inputs and Controls) ---
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # .env File Path Area
        env_file_layout = QHBoxLayout()
        env_label = QLabel(".env File Path:")
        env_file_layout.addWidget(env_label)

        self.env_file_path_line_edit = QLineEdit(".env")
        env_file_layout.addWidget(self.env_file_path_line_edit)

        browse_env_button = QPushButton("Browse...")
        browse_env_button.clicked.connect(self.handle_browse_env_file_click)
        env_file_layout.addWidget(browse_env_button)
        top_layout.addLayout(env_file_layout)

        # User Input Area
        input_label = QLabel("Enter your query:")
        top_layout.addWidget(input_label)
        self.user_input_area = QTextEdit()
        self.user_input_area.setPlaceholderText("Type your query for the agentic system here...")
        top_layout.addWidget(self.user_input_area)

        # Buttons Layout (Execute and Clear)
        buttons_layout = QHBoxLayout()
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.handle_execute_button)
        buttons_layout.addWidget(self.execute_button)

        self.clear_button = QPushButton("Clear/Reset")
        self.clear_button.clicked.connect(self.handle_clear_button_click)
        buttons_layout.addWidget(self.clear_button)
        top_layout.addLayout(buttons_layout)
        
        top_widget.setLayout(top_layout) # Set layout for top_widget
        main_v_splitter.addWidget(top_widget)


        # --- Bottom Outputs Splitter (Horizontal: Left and Right Output Sections) ---
        bottom_h_splitter = QSplitter(Qt.Horizontal)

        # Left Outputs Splitter (Vertical: Intermediate Steps and SQL Query)
        left_v_splitter = QSplitter(Qt.Vertical)
        
        # Pane 1: Intermediate Steps / Full Log
        intermediate_steps_widget = QWidget()
        intermediate_steps_layout = QVBoxLayout(intermediate_steps_widget)
        intermediate_steps_label = QLabel("Intermediate Steps / Full Log:")
        intermediate_steps_layout.addWidget(intermediate_steps_label)
        self.response_area = QTextEdit() # This is the existing response_area
        self.response_area.setReadOnly(True)
        intermediate_steps_layout.addWidget(self.response_area)
        left_v_splitter.addWidget(intermediate_steps_widget)

        # Pane 2: SQL Query
        sql_query_widget = QWidget()
        sql_query_layout = QVBoxLayout(sql_query_widget)
        sql_query_label = QLabel("SQL Query:")
        sql_query_layout.addWidget(sql_query_label)
        self.sql_query_area = QTextEdit()
        self.sql_query_area.setReadOnly(True)
        sql_query_layout.addWidget(self.sql_query_area)
        left_v_splitter.addWidget(sql_query_widget)
        
        bottom_h_splitter.addWidget(left_v_splitter)

        # Right Outputs Splitter (Vertical: Final Answer and Performance/Feedback)
        right_v_splitter = QSplitter(Qt.Vertical)

        # Pane 3: Final LLM Answer
        final_answer_widget = QWidget()
        final_answer_layout = QVBoxLayout(final_answer_widget)
        final_answer_label = QLabel("Final LLM Answer:")
        final_answer_layout.addWidget(final_answer_label)
        self.final_answer_area = QTextEdit()
        self.final_answer_area.setReadOnly(True)
        final_answer_layout.addWidget(self.final_answer_area)
        right_v_splitter.addWidget(final_answer_widget)

        # Pane 4: Performance & Feedback
        self.perf_feedback_widget = QWidget()
        perf_feedback_main_layout = QVBoxLayout(self.perf_feedback_widget)
        
        perf_feedback_label = QLabel("Performance & Feedback:")
        perf_feedback_main_layout.addWidget(perf_feedback_label)

        # Performance Sub-section
        perf_form_layout = QFormLayout()
        self.total_time_label_val = QLabel("N/A")
        self.q_gen_time_label_val = QLabel("N/A")
        self.data_retr_time_label_val = QLabel("N/A")
        self.ans_gen_time_label_val = QLabel("N/A")
        perf_form_layout.addRow("Total Execution Time:", self.total_time_label_val)
        perf_form_layout.addRow("Question Gen Time:", self.q_gen_time_label_val)
        perf_form_layout.addRow("Data Retrieval Time:", self.data_retr_time_label_val)
        perf_form_layout.addRow("Answer Gen Time:", self.ans_gen_time_label_val)
        perf_feedback_main_layout.addLayout(perf_form_layout)

        # Feedback Sub-section
        feedback_buttons_layout = QHBoxLayout()
        self.like_button = QPushButton("👍 Like") # Placeholder
        self.dislike_button = QPushButton("👎 Dislike") # Placeholder
        feedback_buttons_layout.addWidget(self.like_button)
        feedback_buttons_layout.addWidget(self.dislike_button)
        perf_feedback_main_layout.addLayout(feedback_buttons_layout)

        self.feedback_comment_area = QTextEdit()
        self.feedback_comment_area.setPlaceholderText("Optional comments...")
        self.feedback_comment_area.setFixedHeight(60) # Smaller height
        perf_feedback_main_layout.addWidget(self.feedback_comment_area)

        self.save_session_button = QPushButton("Save Session") # Placeholder
        perf_feedback_main_layout.addWidget(self.save_session_button)
        
        perf_feedback_main_layout.addStretch() # Push elements to top
        right_v_splitter.addWidget(self.perf_feedback_widget)

        bottom_h_splitter.addWidget(right_v_splitter)
        
        # Add splitters to main vertical splitter
        main_v_splitter.addWidget(bottom_h_splitter)

        # Set initial splitter sizes (adjust as needed)
        main_v_splitter.setSizes([250, 550]) # Top area, Bottom area
        bottom_h_splitter.setSizes([500, 500]) # Left section, Right section
        left_v_splitter.setSizes([400,150]) # Intermediate steps, SQL query
        right_v_splitter.setSizes([250,300]) # Final Answer, Perf/Feedback

        # Initial load of .env configuration (messages go to self.response_area)
        self.load_dot_env_config(self.env_file_path_line_edit.text())


    def handle_browse_env_file_click(self):
        """Opens a file dialog to select a .env file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select .env File", 
            "",  # Start directory (current if empty)
            "Env files (*.env);;All files (*)"
        )
        if file_path:
            self.env_file_path_line_edit.setText(file_path)
            self.load_dot_env_config(file_path) # Automatically load after selection

    def load_dot_env_config(self, env_path: str):
        """Loads configuration from the specified .env file."""
        self.response_area.setReadOnly(False) # Allow editing for status messages
        
        # Clear previous .env status messages if any (optional, or use a dedicated status bar)
        # For now, just append to the main response area.
        self.response_area.append(f"\n--- .env Configuration Loading ---")
        
        if not os.path.exists(env_path):
            self.response_area.append(f"Error: .env file not found at '{env_path}'.")
            self.loaded_env_file_path = None
            self.response_area.setReadOnly(True)
            return

        try:
            # load_dotenv will return True if it found and loaded the file, False otherwise.
            # It doesn't raise an exception for a missing file by default.
            success = load_dotenv(dotenv_path=env_path, override=True)
            if success:
                self.loaded_env_file_path = env_path
                found_vars = [var for var in EXPECTED_BQ_VARS if os.getenv(var)]
                missing_vars = [var for var in EXPECTED_BQ_VARS if not os.getenv(var)]

                if len(found_vars) == len(EXPECTED_BQ_VARS):
                    self.response_area.append(f"Successfully loaded .env file: {env_path}")
                    self.response_area.append(f"Found BigQuery variables: {', '.join(found_vars)}")
                elif found_vars:
                    self.response_area.append(f"Warning: .env file loaded from {env_path}, but some BigQuery variables are missing.")
                    self.response_area.append(f"Found: {', '.join(found_vars)}")
                    self.response_area.append(f"Missing: {', '.join(missing_vars)}")
                else:
                    self.response_area.append(f"Warning: .env file loaded from {env_path}, but ALL expected BigQuery variables are missing.")
                    self.response_area.append(f"Missing: {', '.join(missing_vars)}")
            else:
                # This case might be redundant if os.path.exists is checked first,
                # but load_dotenv might return False for other reasons (e.g., empty file, permissions).
                self.response_area.append(f"Error: Failed to load .env file from '{env_path}'. File might be empty or unreadable.")
                self.loaded_env_file_path = None
        except Exception as e:
            self.response_area.append(f"An unexpected error occurred while loading .env file '{env_path}': {e}")
            self.loaded_env_file_path = None
        finally:
            self.response_area.append("----------------------------------\n")
            self.response_area.setReadOnly(True)
            if hasattr(self, 'response_area'): # Ensure response_area is initialized
                self.response_area.verticalScrollBar().setValue(self.response_area.verticalScrollBar().maximum())


    def handle_clear_button_click(self):
        """Clears all input and output areas."""
        self.user_input_area.clear()
        
        # Clear output panes
        self.response_area.setReadOnly(False)
        self.response_area.clear()
        self.response_area.setReadOnly(True)

        self.sql_query_area.setReadOnly(False)
        self.sql_query_area.clear()
        self.sql_query_area.setReadOnly(True)

        self.final_answer_area.setReadOnly(False)
        self.final_answer_area.clear()
        self.final_answer_area.setReadOnly(True)
        
        self.feedback_comment_area.clear()
        
        # Reset performance labels
        self.total_time_label_val.setText("N/A")
        self.q_gen_time_label_val.setText("N/A")
        self.data_retr_time_label_val.setText("N/A")
        self.ans_gen_time_label_val.setText("N/A")
        
        # Optionally, re-display initial .env loading status
        self.response_area.setReadOnly(False)
        self.response_area.append("--- UI Cleared ---")
        if self.loaded_env_file_path:
             self.response_area.append(f"Current .env config: {self.loaded_env_file_path}\n")
        else:
             self.response_area.append("No .env file currently loaded.\n")
        self.response_area.setReadOnly(True)


    def handle_execute_button(self):
        """Handles the execute button click by starting the GraphWorker thread."""
        user_input = self.user_input_area.toPlainText().strip()
        
        # Clear previous run's specific outputs before starting new one
        self.sql_query_area.setReadOnly(False); self.sql_query_area.clear(); self.sql_query_area.setReadOnly(True)
        self.final_answer_area.setReadOnly(False); self.final_answer_area.clear(); self.final_answer_area.setReadOnly(True)
        self.total_time_label_val.setText("N/A"); self.q_gen_time_label_val.setText("N/A"); 
        self.data_retr_time_label_val.setText("N/A"); self.ans_gen_time_label_val.setText("N/A")

        self.response_area.setReadOnly(False) # For full log
        self.response_area.clear() 
        
        if not user_input:
            self.response_area.append("Error: User input cannot be empty.")
            self.response_area.setReadOnly(True)
            return

        if self.loaded_env_file_path:
             self.response_area.append(f"Using .env config from: {self.loaded_env_file_path}\n")
        else:
             self.response_area.append("Warning: No .env file loaded. Proceeding with current environment.\n")
        self.response_area.append("--- Starting Graph Execution ---\n")
        self.response_area.setReadOnly(True)

        self.execute_button.setEnabled(False)
        self.clear_button.setEnabled(False)

        # Pass a copy of the current environment variables.
        # load_dotenv modifies os.environ, so graph.py should pick up changes automatically
        # if it also uses os.environ. Passing them explicitly is also an option.
        current_env_vars = dict(os.environ) 
        self.graph_worker = GraphWorker(user_input, current_env_vars)
        
        self.graph_worker.new_event.connect(self.handle_graph_event)
        self.graph_worker.error_occurred.connect(self.handle_graph_error)
        self.graph_worker.execution_done.connect(self.handle_execution_finished)
        # If using finished_successfully:
        # self.graph_worker.finished_successfully.connect(self.handle_graph_success)
        
        self.graph_worker.start()

    def handle_graph_event(self, event_data):
        """Handles new events from the graph worker (Intermediate Steps/Full Log)."""
        self.response_area.setReadOnly(False)
        try:
            # Attempt to pretty-print if it's a dictionary (common for LangGraph events)
            if isinstance(event_data, dict):
                formatted_event = json.dumps(event_data, indent=2)
            else:
                formatted_event = str(event_data)
            self.response_area.append(f"\n--- New Event ---\n{formatted_event}\n")
        except Exception as e:
            self.response_area.append(f"\n--- New Event (raw) ---\n{str(event_data)}\nError formatting event: {e}\n")
        self.response_area.setReadOnly(True)
        self.response_area.verticalScrollBar().setValue(self.response_area.verticalScrollBar().maximum())

        # --- Logic to display final answer in its dedicated pane ---
        if isinstance(event_data, dict):
            final_answer_str = None
            # Check for output from the answer_generator node
            if "answer_generator" in event_data:
                answer_output = event_data.get("answer_generator")
                # The output of a node is directly its return value (the Pydantic model in this case)
                # LangGraph events often look like: { "node_name": {"key_in_state": value_from_node_pydantic_field} }
                # or { "node_name": PydanticModelOutput }
                # Given our graph structure, the node output is the Pydantic model itself.
                # So, if state['answer'] is updated by 'answer_generator', it's event_data['answer_generator']['answer']
                # If 'answer_generator' returns an AnswerState model, it's event_data['answer_generator'].answer
                if isinstance(answer_output, dict) and "answer" in answer_output: # Check if it's a dict (like AnswerState.model_dump())
                    final_answer_str = str(answer_output["answer"])
                elif hasattr(answer_output, 'answer'): # Check if it's an AnswerState Pydantic object
                    final_answer_str = str(answer_output.answer)
                    
            # Check for output from the cannot_answer_node
            elif "cannot_answer_node" in event_data:
                cannot_answer_output = event_data.get("cannot_answer_node")
                if isinstance(cannot_answer_output, dict) and "answer" in cannot_answer_output:
                    final_answer_str = str(cannot_answer_output["answer"])
                elif hasattr(cannot_answer_output, 'answer'): # Check if it's an AnswerState Pydantic object
                    final_answer_str = str(cannot_answer_output.answer)

            if final_answer_str is not None:
                self.final_answer_area.setReadOnly(False)
                self.final_answer_area.clear() # Clear previous final answer
                self.final_answer_area.append(final_answer_str)
                self.final_answer_area.setReadOnly(True)
                self.final_answer_area.verticalScrollBar().setValue(self.final_answer_area.verticalScrollBar().maximum())

            # --- Logic to display SQL queries in its dedicated pane ---
            if "data_retriever" in event_data:
                data_retriever_output = event_data.get("data_retriever")
                # data_retriever_output is the DataState Pydantic model or its dict representation
                
                retrieved_items_data = None
                if isinstance(data_retriever_output, dict) and "retrieved_data" in data_retriever_output:
                    retrieved_items_data = data_retriever_output["retrieved_data"]
                elif hasattr(data_retriever_output, 'retrieved_data'): # If it's a DataState Pydantic object
                    retrieved_items_data = data_retriever_output.retrieved_data

                self.sql_query_area.setReadOnly(False)
                self.sql_query_area.clear() # Clear previous queries for this step
                
                if isinstance(retrieved_items_data, list) and retrieved_items_data:
                    queries_found = []
                    for item in retrieved_items_data:
                        if isinstance(item, dict) and "query_attempted" in item and item["query_attempted"] is not None:
                            queries_found.append(str(item["query_attempted"]))
                    
                    if queries_found:
                        self.sql_query_area.append("\n\n---\n\n".join(queries_found))
                    else:
                        self.sql_query_area.append("No SQL query attempts found or executed in this step.")
                elif retrieved_items_data: # If it's not None/empty but not a list
                    self.sql_query_area.append(f"Unexpected format for retrieved_data: {type(retrieved_items_data)}\nContent: {str(retrieved_items_data)[:500]}")
                else: # If it's None or empty
                     self.sql_query_area.append("No data retrieval attempts or 'retrieved_data' was empty for this step.")
                self.sql_query_area.setReadOnly(True)
                self.sql_query_area.verticalScrollBar().setValue(self.sql_query_area.verticalScrollBar().maximum())


    def handle_graph_error(self, error_message):
        """Handles errors reported by the graph worker (Intermediate Steps/Full Log)."""
        self.response_area.setReadOnly(False)
        self.response_area.append(f"\n--- Error During Graph Execution ---\n{error_message}\n")
        self.response_area.setReadOnly(True)
        self.response_area.verticalScrollBar().setValue(self.response_area.verticalScrollBar().maximum())


    # def handle_graph_success(self, final_output_data):
    #     """Handles successful completion signal with final output (Final Answer Pane)."""
    #     self.final_answer_area.setReadOnly(False)
    #     self.final_answer_area.setText(json.dumps(final_output_data.get("answer", final_output_data), indent=2))
    #     self.final_answer_area.setReadOnly(True)

    def handle_execution_finished(self):
        """Handles the signal that graph execution is done."""
        self.execute_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.response_area.setReadOnly(False)
        self.response_area.append("\n--- Graph Execution Finished ---\n")
        self.response_area.setReadOnly(True)
        self.response_area.verticalScrollBar().setValue(self.response_area.verticalScrollBar().maximum())
        if self.graph_worker: 
             self.graph_worker.quit() 
             self.graph_worker.wait() 
             self.graph_worker = None


    def closeEvent(self, event):
        """Handle window close event to stop the worker thread if it's running."""
        if self.graph_worker and self.graph_worker.isRunning():
            print("Window closing, attempting to stop worker thread...")
            self.graph_worker.stop()  # Signal the thread to stop
            self.graph_worker.quit()
            if not self.graph_worker.wait(1000): # Wait for 1 sec
                 print("Worker thread did not stop gracefully. Forcing termination (this might be unsafe).")
                 self.graph_worker.terminate() # Force terminate if it doesn't stop
                 self.graph_worker.wait() # Wait again
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # main_window is now an instance variable, no need for global one for placeholder
    launcher_window = InteractiveLauncherWindow() 
    launcher_window.show()
    sys.exit(app.exec())
