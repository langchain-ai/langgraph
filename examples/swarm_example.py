from typing import TypedDict
from langgraph.graph import StateGraph

# Define the state schema
class SwarmState(TypedDict):
    message: str

# Define worker nodes
def worker1(state: SwarmState) -> SwarmState:
    print("Worker 1 processing:", state)
    return {"message": state["message"] + " → W1"}

def worker2(state: SwarmState) -> SwarmState:
    print("Worker 2 processing:", state)
    return {"message": state["message"] + " → W2"}

# Build the graph
def build_swarm_graph():
    builder = StateGraph(SwarmState)
    builder.add_node("worker1", worker1)
    builder.add_node("worker2", worker2)

    builder.set_entry_point("worker1")
    builder.add_edge("worker1", "worker2")

    return builder.compile()

# Main function
def main():
    graph = build_swarm_graph()
    initial_state = {"message": "Hello from the Swarm!"}
    final_state = graph.invoke(initial_state)
    print("Final Output:", final_state)

if __name__ == "__main__":
    main()
