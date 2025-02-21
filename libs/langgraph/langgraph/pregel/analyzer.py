import networkx as nx
from collections import defaultdict
from langgraph.graph.graph import START

class Analyzer:
    # create original directed graph
    digraph = nx.DiGraph()
    nodes_children = []
    # store selector nodes and their corresponding active branch lists
    selector_nodes = {}
    # store nodes and their corresponding preconditions
    nodes_precondition = defaultdict(list)
    nodes_precondition_combinations = defaultdict(list)

    def find_connected_nodes(self):
        """
        find all nodes' children from start node
        """
        self.nodes_children = set(nx.descendants(self.digraph, START))

    def add_edge(self, start, end):
        # add all edges to graph,include condition edges and normal edges
        self.digraph.add_edge(start, end)

    def add_selector(self, selector, active_branches):
        """
        add selector node and its active branch list

        :param selector: selector node
        :param active_branches: active branch list
        """
        if not isinstance(active_branches, list):
            active_branches = [active_branches]

        self.selector_nodes[selector] = active_branches

    def _trace_path_with_selectors(self, start, end):
        """
        trace path, consider the influence of selectors

        :param start: start node
        :param end: end node
        :return: possible path list
        """

        def dfs_path(current, path):
            if current == end:
                return [path]

            all_paths = []

            # check if current node is a selector
            is_selector = current in self.selector_nodes
            active_branches = (
                self.selector_nodes.get(current, []) if is_selector else []
            )

            for next_node in list(self.digraph.neighbors(current)):
                # if current node is a selector, only allow access to active branches
                if is_selector and next_node not in active_branches:
                    continue

                # avoid loop
                if next_node not in path:
                    new_path = path + [next_node]
                    all_paths.extend(dfs_path(next_node, new_path))

            return all_paths

        return dfs_path(start, [start])

    def find_valid_paths_to_end(self, start, end):
        """
        find all valid paths from start to end

        :param start: start node
        :param end: end node
        :return: valid path list
        """
        paths = self._trace_path_with_selectors(start, end)
        return paths

    def filter_edges_to_end(self, start_node, end_node):
        """
        filter edges to end node

        :param start: start node
        :param end: end node
        :return: valid edges
        """
        # find all valid paths
        valid_paths = self.find_valid_paths_to_end(start_node, end_node)

        # extract edges from valid paths
        valid_edges = set()
        for path in valid_paths:
            valid_edges.update(list(zip(path[:-1], path[1:])))

        return valid_edges

    def get_node_valid_edges(self, node, valid_edges):
        return {edge[0] for edge in valid_edges if node == edge[1]}
