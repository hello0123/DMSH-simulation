import pickle
from typing import Dict, Tuple, Any, Optional
from .edge_types import EdgeType

class Node:
    """
    Represents a node in the satellite network graph.
    """
    def __init__(self, name: str):
        self.name = name
        
    def __str__(self):
        return f"Node({self.name})"

class Edge:
    """
    Represents an edge in the satellite network graph.
    """
    def __init__(self, edge_type: EdgeType, capacity: int, length: int):
        self.type = edge_type
        self.capacity = capacity
        self.length = length
        
    def __str__(self):
        return f"Edge(type={self.type}, capacity={self.capacity}, length={self.length})"

class Graph:
    """
    Represents a graph for the satellite network.
    """
    def __init__(self):
        self.nodes = {}  # Dict[str, Node]
        self.node_capacity = {}  # Dict[str, int]
        self.edges = {}  # Dict[Tuple[str, str, Optional[int]], Edge]
    
    def add_edge(self, u: str, v: str, ue: Optional[int], edge_type: EdgeType, capacity: int, length: int):
        """
        Add an edge between nodes u and v.
        
        Args:
            u: Source node name
            v: Destination node name
            ue: User equipment ID (optional)
            edge_type: Type of the edge
            capacity: Capacity of the edge
            length: Length of the edge
        """
        edge = Edge(edge_type, capacity, length)
        self.edges[(u, v, ue)] = edge
    
    def add_node(self, v: str):
        """
        Add a node to the graph.
        
        Args:
            v: Node name
        """
        n = Node(v)
        self.nodes[v] = n
    
    def set_node_capacity(self, v: str, capacity: int):
        """
        Set the capacity of a node.
        
        Args:
            v: Node name
            capacity: Capacity value
        """
        self.node_capacity[v] = capacity

    def get_nodes(self):
        """
        Get all nodes in the graph.
        
        Returns:
            Dictionary of nodes
        """
        return self.nodes

    def get_edges(self):
        """
        Get all edges in the graph.
        
        Returns:
            Dictionary of edges
        """
        return self.edges
        
    def save_graph(self, filename: str):
        """
        Save the graph to a file using pickle.
        
        Args:
            filename: File path to save the graph
        """
        # Create a dictionary to store all graph data
        graph_data = {
            'nodes': list(self.nodes.keys()),
            'node_capacity': self.node_capacity,
            'edges': {}
        }
        
        # Store edge data
        for (u, v, ue), edge in self.edges.items():
            graph_data['edges'][(u, v, ue)] = {
                'type': edge.type,
                'capacity': edge.capacity,
                'length': edge.length
            }
        
        # Save to file using pickle
        with open(filename, 'wb') as f:
            pickle.dump(graph_data, f)

    @classmethod
    def load_graph(cls, filename: str) -> 'Graph':
        """
        Load a graph from a file.
        
        Args:
            filename: File path to load the graph from
            
        Returns:
            Loaded Graph instance
        """
        # Create a new graph instance
        graph = cls()
        
        # Load data from file
        with open(filename, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Restore nodes
        for node_id in graph_data['nodes']:
            graph.add_node(node_id)
        
        # Restore node capacities
        for node_id, capacity in graph_data['node_capacity'].items():
            graph.set_node_capacity(node_id, capacity)
        
        # Restore edges
        for (u, v, ue), edge_data in graph_data['edges'].items():
            graph.add_edge(u, v, ue, 
                        edge_data['type'],
                        edge_data['capacity'],
                        edge_data['length'])
        
        return graph
    
    