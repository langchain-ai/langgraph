from langchain_core.runnables import Runnable, RunnableMap, RunnableLambda, RunnablePassthrough
from typing import Callable, Union, Optional, List, Any, Dict
from permchain import Channel, Pregel

#########################################################
#                    NODE CLASSES                       #
#########################################################

class LangGraphNode:
    def __init__(self, key: str):
        self.key = key

    def get_runnable(self) -> Runnable:
        pass


class Actor(LangGraphNode):

    def __init__(self, key: str, runnable: Runnable):
        self.runnable = runnable
        super().__init__(key)

    def get_runnable(self) -> Union[Runnable, Callable]:
        return self.runnable


class End(LangGraphNode):
    def __init__(self):
        super().__init__(key="end")

    def get_runnable(self) -> Union[Runnable, Callable]:
        raise NotImplementedError


class Branch(LangGraphNode):

    def __init__(self, parent_key: str, condition: str):
        self.parent_key = parent_key
        self.condition = condition
        super().__init__(f"{self.parent_key}.{self.condition}")

    def get_runnable(self) -> Union[Runnable, Callable]:
        # Only used for structure, so the runnable should never be called
        raise NotImplementedError
        

class Conditional(LangGraphNode):

    def __init__(self, key: str, conditional_edge_mapping: Dict[str, str], callable: Callable):
        self.callable = callable
        self.branches = []

        for condition, output in conditional_edge_mapping.items():
            self.branches.append(Branch(key, condition))
            
        self.conditional_edge_mapping = conditional_edge_mapping
            
        super().__init__(key)

    def get_runnable(self) -> Union[Runnable, Callable]:
        return self.callable
    


#########################################################
#                    EDGE CLASSES                       #
#########################################################

class LangGraphEdge:
    def __init__(self, start_key: str, end_key: str):
        self.start_key = start_key
        self.end_key = end_key
    
    def flow(self, node_map: Dict[str, LangGraphNode]):
        return (
            Channel.subscribe_to(self.start_key) | 
            node_map[self.start_key].get_runnable() | 
            Channel.write_to(self.end_key)
        )


class BranchEdge(LangGraphEdge):

    def flow(self, node_map: Dict[str, LangGraphNode]):
        # flow should skip over the branch edge
        raise NotImplementedError

        
class ConditionalEdge(LangGraphEdge):
    
    def __init__(self, base_key: str, branch_key: str):
        self.base_key = base_key
        self.branch_key = branch_key
        if not branch_key.startswith(base_key) or not branch_key[len(base_key)] == ".":
            raise ValueError(f"Invalid branch edge from {base_key} to {branch_key}")
            
        super().__init__(base_key, branch_key)

    def _branch(self, data, condition, mapping):
        result = condition(data)
        return Channel.write_to(mapping[result])

    def flow(self, node_map: Dict[str, LangGraphNode]):
        conditional_node = node_map[self.base_key]
        
        return (
            Channel.subscribe_to(self.start_key) | 
            (
                lambda x: self._branch(
                    x, 
                    conditional_node.get_runnable(), 
                    conditional_node.conditional_edge_mapping
                )
            )
        )


class Graph:

    def __init__(self):
        end_node = End()
        self.nodes = {end_node.key: end_node}
        self.edges = []
        
        # self.connections = {}
        # self.branches = {}
        self.entry_point: Optional[str] = None

    def add_node(self, node: Actor):
        if node.key in self.nodes:
            raise ValueError(f"Actor `{node.key}` already present.")
        self.nodes[node.key] = node


    def add_edge(self, start_key: str, end_key: str):
        if start_key not in self.nodes:
            raise ValueError(f"Need to add_node `{start_key}` first")
        if end_key not in self.nodes:
            raise ValueError(f"Need to add_node `{end_key}` first")

        # TODO: support multiple message passing
        if start_key in set(edge.start_key for edge in self.edges):
            raise ValueError(f"Already found path for {start_key}")
            
        self.edges.append(LangGraphEdge(start_key, end_key))

    def add_conditional_edges(
        self,
        start_key: str,
        condition: Callable[Any, str],
        conditional_edge_mapping: Dict[str, str]):

        conditional_node = Conditional(
            f"_conditional_from_{start_key}",
            conditional_edge_mapping,
            condition
        )

        self.add_node(conditional_node)
        self.add_edge(start_key, conditional_node.key)

        for branch in conditional_node.branches:
            self.add_node(branch)
            self.edges.append(ConditionalEdge(conditional_node.key, branch.key))
            self.edges.append(
                BranchEdge(branch.key, conditional_node.conditional_edge_mapping[branch.condition])
            )
        
    
    def set_entry_point(self, key: str):
        if key not in self.nodes:
            raise ValueError(f"Need to add_node `{node.key}` first")
        self.entry_point = key

    def set_finish_point(self, key: str):
        if key not in self.nodes:
            raise ValueError(f"Need to add_node `{node.key}` first")
        self.add_edge(key, "end")

    def compile(self):

        ################################################
        #       STEP 1: VALIDATE GRAPH STRUCTURE       #
        ################################################
        seen_node_keys = set()
        all_node_keys = set(self.nodes.keys())
        
        edge_map = {}
        for edge in self.edges:
            edge_map[edge.start_key] = edge_map.get(edge.start_key, []) + [edge.end_key]

        to_see = [self.entry_point]
        while len(to_see) > 0:
            current = to_see.pop(0)
            if current in seen_node_keys:
                continue

            seen_node_keys.add(current)
            next_nodes = edge_map.get(current, [])
            to_see += next_nodes

            if len(next_nodes) == 0 and current != "end":
                raise ValueError(f"Node {current} is a dead end")

        if seen_node_keys != all_node_keys:
            raise ValueError(f"Found unreachable nodes: {list(all_node_keys - seen_node_keys)}")
            
        
        ################################################
        #             STEP 2: CREATE GRAPH             #
        ################################################

        chains = {
            edge.start_key: edge.flow(self.nodes)
            for edge in self.edges
            # specifically skip over branch edges since they are defined purely for structure
            if not isinstance(edge, BranchEdge)
        }
        
        app = Pregel(
            chains=chains,
            input=self.entry_point,
            output="end"
        )
        return app










