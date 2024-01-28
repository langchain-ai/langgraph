from collections import defaultdict

from langgraph.graph.state import StateGraph


class StateGraphDrawer:
    """
    A helper class to draw a state graph into a PNG file.
    Requires graphviz and pygraphviz to be installed.

    :param fontname: The font to use for the labels
    :param label_overrides: A dictionary of label overrides. The dictionary
        should have the following format:
        {
            "nodes": {
                "node1": "CustomLabel1",
                "node2": "CustomLabel2",
                "__end__": "End Node"
            },
            "conditional_edges": {
                "should_continue": "ConditionLabel",
                "should_continue2": "ConditionLabel2",
            },
            "edges": {
                "continue": "ContinueLabel",
                "end": "EndLabel"
            }
        }

        The keys are the original labels, and the values are the new labels.
        
    Usage:
        drawer = StateGraphDrawer()
        drawer.draw(state_graph, 'graph.png')    
    """

    def __init__(self, fontname="calibri", label_overrides=None):
        self.fontname = fontname
        self.label_overrides = defaultdict(dict) if not label_overrides else label_overrides

    def get_node_label(self, label: str) -> str:
        label = self.label_overrides.get('nodes', {}).get(label, label)
        return f"<<B>{label}</B>>"

    def get_conditional_edge_label(self, label: str) -> str:
        label = self.label_overrides.get('conditional_edges', {}).get(label, label)
        return f"<<I>{label}</I>>"

    def get_edge_label(self, label: str) -> str:
        label = self.label_overrides.get('edges', {}).get(label, label)
        return f"<<U>{label}</U>>"

    def add_node(
        self,
        graphviz_graph,
        node: str,
        label: str = None
    ):
        if not label:
            label = node 

        graphviz_graph.add_node(
            node,
            label=self.get_node_label(label),
            style='filled',
            fillcolor='yellow',
            fontsize=15,
            fontname=self.fontname
        )

    def add_conditional_node(
        self,
        graphviz_graph,
        node: str,
        label: str = None
    ):
        if not label:
            label = node

        graphviz_graph.add_node(
            node,
            label=self.get_conditional_edge_label(label),
            shape='rect',
            fixedsize=True,
            width=0.12 * len(label),
            height=0.4,
            fontsize=12,
            fontname=self.fontname
        )

    def add_edge(
        self,
        graphviz_graph,
        source: str,
        target: str,
        label: str = None
    ):
        graphviz_graph.add_edge(
            source,
            target,
            label=self.get_edge_label(label) if label else '',
            fontsize=12,
            fontname=self.fontname
        )

    def draw(
        self,
        state_graph: StateGraph,
        output_file_path='graph.png'
    ):
        """
        Draws the given state graph into a PNG file.
        Requires graphviz and pygraphviz to be installed.

        :param state_graph: The state graph to draw
        :param output_file_path: The path to the output file
        """
                
        try:
            import pygraphviz as pgv
        except ImportError:
            raise ImportError("pygraphviz is required to draw the state graph")

        # Create a directed graph
        graphviz_graph = pgv.AGraph(directed=True, nodesep=0.9, ranksep=1.0)

        # Add nodes, conditional edges, and edges to the graph
        self.add_nodes(graphviz_graph, state_graph)
        self.add_conditional_edges(graphviz_graph, state_graph)
        self.add_edges(graphviz_graph, state_graph)

        # Update entrypoint and END styles
        self.update_styles(graphviz_graph, state_graph)

        # Save the graph as PNG
        graphviz_graph.draw(
            output_file_path,
            format='png',
            prog='dot'
        )
        graphviz_graph.close()

    def add_nodes(self, graph, state_graph):
        for node, _ in state_graph.nodes.items():
            self.add_node(graph, node, node)
        self.add_node(graph, "__end__")

    def add_conditional_edges(self, graph, state_graph):
        for source, _target in state_graph.branches.items():
            branch = _target[0]
            condition = branch.condition.__name__
            self.add_conditional_node(graph, condition)
            for check_result, target in branch.ends.items():
                self.add_edge(graph, source, condition)
                self.add_edge(graph, condition, target, label=check_result)

    def add_edges(self, graph, state_graph):
        for start, end in state_graph.edges:
            self.add_edge(graph, start, end)

    def update_styles(self, graph, state_graph):
        graph.get_node(state_graph.entry_point).attr.update(fillcolor='lightblue')
        graph.get_node("__end__").attr.update(fillcolor='orange')