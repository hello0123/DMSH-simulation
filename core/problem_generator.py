import random
import math
from typing import Dict, List, Set, Tuple
from .graph import Graph
from .edge_types import EdgeType

def get_coverage_period(sat_t: List[List[int]], list_sat: List[str], sim_time: int) -> Dict[str, List[int]]:
    """
    Determine the coverage period for each satellite.
    
    Args:
        sat_t: List of lists containing visible satellites at each time slot
        list_sat: List of satellite IDs
        sim_time: Total simulation time
        
    Returns:
        Dictionary mapping satellite IDs to their coverage periods
    """
    coverage_period = {}
    for sat in list_sat:
        coverage_period[str(sat)] = []

    for t in range(sim_time):
        for sat in sat_t[t]:
            coverage_period[str(sat)].append(t)
            
    return coverage_period

def construct_graph(num_sat: int, sim_time: int, coverage_period: Dict[str, List[int]], 
                   node_capacity: Dict[str, int], list_sat: List[str], edge_cap: int) -> Graph:
    """
    Construct a graph representing the satellite network.
    
    Args:
        num_sat: Number of satellites
        sim_time: Total simulation time
        coverage_period: Dictionary mapping satellite IDs to their coverage periods
        node_capacity: Dictionary mapping satellite IDs to their capacity
        list_sat: List of satellite IDs
        edge_cap: Edge capacity
        
    Returns:
        Constructed Graph instance
    """
    G = Graph()

    begin_sat = []
    end_sat = []

    # Create vertex, edge
    for s1 in list_sat:
        G.add_node(str(s1))
        G.set_node_capacity(str(s1), node_capacity[str(s1)])
        
        # Check begin_sat
        if 0 in coverage_period[s1]:
            begin_sat.append(s1)
        if sim_time-1 in coverage_period[s1]:
            end_sat.append(s1)
            
        # Check end_sat
        for s2 in list_sat:
            if s1 == s2:
                continue
                
            overlap = list(set(coverage_period[s1]) & set(coverage_period[s2]))
            if len(overlap) > 0:
                G.add_edge(str(s1), str(s2), None, EdgeType.FORWARD, edge_cap, 1)

    # Vertex, edge of source + destination
    cap_st = sum(cap for cap in node_capacity.values())
    G.add_node('S')
    G.set_node_capacity('S', cap_st)
    G.add_node('T')
    G.set_node_capacity('T', cap_st)
    
    for sat in begin_sat:
        G.add_edge('S', str(sat), None, EdgeType.FORWARD, edge_cap, 1)

    for sat in end_sat:
        G.add_edge(str(sat), 'T', None, EdgeType.FORWARD, edge_cap, 1)

    return G

def generate_bin_packing_instance(n_items=50, node_capacity=100, cut=None, seed=42):
    """
    Generates a challenging bin packing problem instance with total weight exactly equal to cut.
    
    Args:
        n_items: Number of items to generate
        node_capacity: Capacity of each node
        cut: Target total weight (if None, will be set to node_capacity * (n_items // 2))
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping item IDs to their weights
    """
    random.seed(seed)
    
    if cut is None:
        cut = node_capacity * (n_items // 2)
    
    # Generate initial items
    items = []
    n_large = n_items // 4
    for _ in range(n_large):
        items.append(random.randint(int(0.30 * node_capacity), 
                                  int(0.45 * node_capacity)))
    
    n_medium = n_items // 3
    for _ in range(n_medium):
        items.append(random.randint(int(0.20 * node_capacity), 
                                  int(0.30 * node_capacity)))
    
    for _ in range(n_items - n_large - n_medium):
        items.append(random.randint(int(0.05 * node_capacity), 
                                  int(0.15 * node_capacity)))
    
    random.shuffle(items)
    
    # Scale items to match cut exactly
    total_weight = sum(items)
    scale_factor = cut / total_weight
    items = [int(item * scale_factor) for item in items]
    
    # Adjust for rounding errors to match cut exactly
    items = [max(1, item) for item in items]  # Ensure no zeros
    current_total = sum(items)
    
    # Distribute remaining difference across items
    diff = cut - current_total
    if diff > 0:
        # Add the remaining weight to random items
        indices = random.sample(range(len(items)), min(abs(diff), len(items)))
        for idx in indices:
            items[idx] += 1
    elif diff < 0:
        # Subtract from largest items to maintain problem difficulty
        sorted_indices = sorted(range(len(items)), key=lambda k: items[k], reverse=True)
        for idx in sorted_indices[:abs(diff)]:
            if items[idx] > 1:  # Ensure no zeros
                items[idx] -= 1

    # Create commodities dictionary
    commodities = {i: demand for i, demand in enumerate(sorted(items))}
    
    return commodities

