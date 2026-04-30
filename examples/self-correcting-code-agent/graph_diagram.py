"""
Run this script to render the graph as ASCII or PNG (requires graphviz).
Usage: python graph_diagram.py
"""
from agent import build_graph

graph = build_graph()

# ASCII diagram — always works
print(graph.get_graph().draw_ascii())

# Uncomment for PNG (requires: pip install pygraphviz)
# img = graph.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(img)
# print("Saved graph.png")
