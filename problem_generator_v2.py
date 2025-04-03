from collections import defaultdict
from enum import Enum
import random
import math
import networkx as nx
import pickle
from typing import Dict, Tuple, Any, List, Set

class EdgeType(Enum):
    FORWARD = 1
    BACKWARD = 2
    VERTEX_FOR = 3
    VERTEX_REV = 4

class Node:
    def __init__(self, v:str):
        self.name = v

class Edge:
    def __init__(self, type, capacity, length):
        self.type = type
        self.capacity=capacity
        self.length = length

#Graph.edges[(u,v,ue)] = (type,capacity,length)
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_capacity = {}
        self.edges = {}
        #self.edges: Dict[Tuple(str,str,int),Edge]
    
    #add_edge(u, v, ue, type, capacity, length)
    def add_edge(self, u:str, v:str, ue, type, capacity, length):
        #check key "u" is in edges and add the edge (u,v)
        edge = Edge(type, capacity, length)
        self.edges[(u,v,ue)] = edge
    
    def add_node(self, v:str):
        n = Node(v)
        self.nodes[v] = n
    
    def set_node_capacity(self, v:str, capacity:int):
        self.node_capacity[v] = capacity

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges  

    def save_graph(self, filename: str):
        """
        Save the graph to a file
        Args:
            filename (str): file path to save the graph
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
        Load a graph from a file
        Args:
            filename (str): file path to load the graph from
        Returns:
            Graph: loaded graph instance
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

    def find_min_cut_value(self, source: str, sink: str) -> float:
        """
        Find the value of minimum cut in the graph using Ford-Fulkerson algorithm
        Args:
            source (str): source node
            sink (str): sink node
        Returns:
            float: value of the minimum cut
        """
        def bfs(graph, residual, source, sink):
            visited = {source}
            paths = {source: []}
            if source == sink:
                return paths[source]
                
            q = [source]
            while q:
                u = q.pop(0)
                for (u_, v, ue), edge in graph.edges.items():
                    if u_ != u:
                        continue
                        
                    residual_capacity = edge.capacity - residual.get((u, v, ue), 0)
                    if residual_capacity > 0 and v not in visited:
                        visited.add(v)
                        paths[v] = paths[u] + [(u, v, ue)]
                        if v == sink:
                            return paths[v]
                        q.append(v)
                        
            return None

        # Initialize residual graph
        residual = {}
        
        # Find augmenting paths and update residual graph
        while True:
            path = bfs(self, residual, source, sink)
            if not path:
                break
                
            # Find minimum residual capacity along the path
            min_capacity = float('inf')
            for u, v, ue in path:
                edge = self.edges[(u, v, ue)]
                residual_capacity = edge.capacity - residual.get((u, v, ue), 0)
                min_capacity = min(min_capacity, residual_capacity)
                
            # Update residual capacities
            for u, v, ue in path:
                residual[(u, v, ue)] = residual.get((u, v, ue), 0) + min_capacity
        
        # Find the source side of the cut
        source_side = set()
        q = [source]
        source_side.add(source)
        
        while q:
            u = q.pop(0)
            for (u_, v, ue), edge in self.edges.items():
                if u_ != u:
                    continue
                residual_capacity = edge.capacity - residual.get((u, v, ue), 0)
                if residual_capacity > 0 and v not in source_side:
                    source_side.add(v)
                    q.append(v)
                    
        # Calculate cut value
        cut_value = 0
        for (u, v, ue), edge in self.edges.items():
            if u in source_side and v not in source_side:
                cut_value += edge.capacity
                
        return cut_value


def generate_bin_packing_instance(n_items, bin_capacity, seed):
    """
    Generates a challenging bin packing problem instance.
    Returns items with weights that make optimal packing non-trivial.
    """
    random.seed(seed)
    
    # Generate items with weights following specific patterns
    items = []
    
    # Large items (30-45% of bin capacity)
    n_large = n_items // 4
    for _ in range(n_large):
        items.append(random.randint(int(0.30 * bin_capacity), 
                                  int(0.45 * bin_capacity)))
    
    # Medium items (20-30% of bin capacity)
    n_medium = n_items // 3
    for _ in range(n_medium):
        items.append(random.randint(int(0.20 * bin_capacity), 
                                  int(0.30 * bin_capacity)))
    
    # Small items (5-15% of bin capacity)
    for _ in range(n_items - n_large - n_medium):
        items.append(random.randint(int(0.05 * bin_capacity), 
                                  int(0.15 * bin_capacity)))
    
    # Shuffle items
    random.shuffle(items)
    
    # Calculate lower bound on bins needed
    total_weight = sum(items)
    min_bins = total_weight // bin_capacity + (1 if total_weight % bin_capacity else 0)
    
    return {
        'items': items,
        'bin_capacity': bin_capacity,
        'theoretical_min_bins': min_bins
    }

def generate_network(sim_time:int,sat_capacity:int,min_bins:int,seed:int):
    sat_t = [[] for _ in range(sim_time)]
    sat_t[0] = [s for s in range(min_bins)]
    node_capacity = [sat_capacity for _ in range(min_bins)]
    num_sat = min_bins

    #add big capacity satellites for each time slot
    random.seed(seed)
    p_stay = 0.8
    for t in range(1,sim_time):
        #random leave
        list_leave = []
        for sat in sat_t[t-1]:
            if(len(sat_t[t-1])-len(list_leave)<=2):
                break
            r = random.random()
            if(r>p_stay):
                list_leave.append(sat)
        #sat_t[t] = list(set(sat_t[t-1]).difference(set(list_leave)))
        for s in sat_t[t-1]:
            if(s in list_leave):
                continue
            sat_t[t].append(s)

        #arrive for reserving "min_bins" satellites with capacity "bin_capacity"
        for i in range(len(list_leave)):
            sat_t[t].append(num_sat)
            node_capacity.append(sat_capacity)
            num_sat += 1
    
    '''
    #add small capacity satellites
    num_small_sat = min_bins
    cap_small_sat = int(0.45*sat_capacity)
    for _ in range(num_small_sat):
        sat_t[0].append(num_sat)
        node_capacity.append(cap_small_sat)
        num_sat += 1

    for t in range(1,sim_time):
        #remove half of satellites in sat_t[t-1]
        list_leave = []
        for i in range(int(0.5*num_small_sat)):
            list_leave.append(sat_t[t-1][i])
        sat_t[t] = list(set(sat_t[t-1]).difference(set(list_leave)))

        #add the same number of satellites into sate_t[t]
        for i in range(int(0.5*num_small_sat)):
            sat_t[t].append(num_sat)
            node_capacity.append(0.45*sat_capacity)
            num_sat += 1
    '''
    
    #add small capacity satellites
    num_small_sat = min_bins
    num_leave_sat = int(min_bins/2) if(min_bins%2==0) else int((min_bins-1)/2)
    cap_small_sat = int(0.45*sat_capacity)
    small_sat_t = [[] for _ in range(sim_time)]

    for t in range(sim_time):
        for i in range(num_small_sat):
            new_sat = int(num_sat+i)
            small_sat_t[t].append(new_sat)
            if(new_sat>=len(node_capacity)):
                node_capacity.append(cap_small_sat)
        num_sat += num_leave_sat
    num_sat += (num_small_sat-num_leave_sat)

    for t in range(sim_time):
        sat_t[t] = sat_t[t] + small_sat_t[t]

    return num_sat, sat_t, node_capacity

#transform items: [int,...,int] into {com:demand} sorted by demand (increasing)
def generate_request(items):
    sum_capacity = 0
    commodities = {}
    sorted_items = sorted(items)
    for item, demand in enumerate(sorted_items):
        commodities[item] = demand
        sum_capacity += demand

    return commodities, sum_capacity


def get_coverage_period(sat_t, list_sat, sim_time):
    coverage_period = {}
    for sat in list_sat:
        coverage_period[str(sat)] = []

    for t in range(sim_time):
        for sat in sat_t[t]:
            coverage_period[str(sat)].append(t)
    return coverage_period


def construct_graph(num_sat,sim_time,coverage_period,node_capacity,list_sat,edge_cap):
    G = Graph()

    begin_sat = []
    end_sat = []

    #create vertex, edge
    for s1 in list_sat:
        G.add_node(str(s1))
        G.set_node_capacity(str(s1),node_capacity[str(s1)])
        #check begin_sat
        if(0 in coverage_period[s1]):
            begin_sat.append(s1)
        if(sim_time-1 in coverage_period[s1]):
            end_sat.append(s1)
        #check end_sat
        for s2 in list_sat:
            if(s1==s2):
                continue
            overlap = list(set(coverage_period[s1])&(set(coverage_period[s2])))
            if(len(overlap)>0):
                G.add_edge(str(s1),str(s2),None,EdgeType.FORWARD,edge_cap,1)
                #add_edge(u, v, ue, type, capacity, length)

    #vertex, edge of source + destination
    cap_st = 0
    for cap in node_capacity.values():
        cap_st += cap
    G.add_node('S')
    G.set_node_capacity('S',cap_st)
    G.add_node('T')
    G.set_node_capacity('T',cap_st)
    for sat in begin_sat:
        G.add_edge('S', str(sat), None, EdgeType.FORWARD,edge_cap,1)

    for sat in end_sat:
        G.add_edge(str(sat), 'T', None, EdgeType.FORWARD,edge_cap,1)

    return G


def generate_reduced_sat_t(sat_t, T):
    num_t = math.ceil(len(sat_t)/T)
    reduced_sat_t = [[] for _ in range(num_t)]

    for index_reduced in range(num_t):
        start = index_reduced*T
        end = min((index_reduced+1)*T-1, len(sat_t)-1)
        for index_sat_t in range(start, end+1):
            for sat in sat_t[index_sat_t]:
                if sat not in reduced_sat_t[index_reduced]:
                    reduced_sat_t[index_reduced].append(sat)

    return reduced_sat_t




#divide by time slots
def construct_graph_v2(sat_t, node_capacity, reduce_ratio):
    G = Graph()
    G.add_node('S')
    G.set_node_capacity('S',math.inf)
    G.add_node('T')
    G.set_node_capacity('T',math.inf)

    reduced_sat_t = generate_reduced_sat_t(sat_t,reduce_ratio)

    #list_of_v_in_previous_slot = []
    for t in range(len(reduced_sat_t)):

        #create vertices, edges for each verex in sat_t[t]
        for v in reduced_sat_t[t]:

            #create vertices v_t
            v_name = str(v)+"-"+str(t)
            G.add_node(v_name)
            G.set_node_capacity(v_name,node_capacity)

            #create edge from S
            if t==0:
                G.add_edge('S', v_name, None, EdgeType.FORWARD, math.inf, 0)
                continue
            #create edge to T
            if t==len(reduced_sat_t)-1:
                G.add_edge(v_name, 'T', None, EdgeType.FORWARD, math.inf, 0)

            #create edge between satellite nodes
            for u in reduced_sat_t[t-1]:
                u_name = str(u)+"-"+str(t-1)
                #u==v
                if u==v:
                    length=0
                    G.add_edge(u_name, v_name, None, EdgeType.FORWARD, math.inf, length)
                #u,v both in t-1 or u,v both in t
                elif v in reduced_sat_t[t-1] or u in reduced_sat_t[t]:
                    length = 1
                    G.add_edge(u_name, v_name, None, EdgeType.FORWARD, math.inf, length)

    return G





