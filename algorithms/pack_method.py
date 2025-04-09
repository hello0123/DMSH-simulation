"""
Pack Method algorithm for satellite scheduling.

This module implements the Pack Method algorithm which aims to minimize
handovers by grouping user flows into packs and finding optimal paths.
"""

import math
import copy
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque

from ..core.graph import Graph
from ..core.edge_types import EdgeType
from ..utils.network_flow import (
    create_residual_graph, 
    find_shortest_path, 
    new_update_flow,
    create_residual_graph_for_split
)
from ..utils.io_utils import process_and_output_flow


def subset_sum(candidate_neighbor: Dict[str, int], demand_reversed: int) -> Dict[str, int]:
    """
    Find a subset of candidates whose values sum to the target value.
    If no exact match exists, find a minimal set that exceeds the target.
    
    Args:
        candidate_neighbor: Dictionary mapping candidate names to their values
        demand_reversed: The target value to match
        
    Returns:
        Dictionary of candidate names and their flow amounts that form the solution
    """
    # Extract the candidates and their values
    candidates = list(candidate_neighbor.keys())
    values = [candidate_neighbor[c] for c in candidates]
    
    # Sort candidates by value in descending order for greedy approach later
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    sorted_candidates = [candidates[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Edge cases
    if demand_reversed == 0:
        return {}
    
    if all(v == 0 for v in values) and demand_reversed != 0:
        return {}
    
    # Initialize DP table
    dp = [[-float('inf')] * (int(demand_reversed) + 1) for _ in range(len(candidates) + 1)]
    dp[0][0] = 0  # Base case: empty set sums to 0
    
    # Keep track of choices
    choices = {}
    
    # Fill the DP table
    for i in range(1, len(candidates) + 1):
        for j in range(int(demand_reversed) + 1):
            # Don't include current candidate
            dp[i][j] = dp[i-1][j]
            
            # Try to include current candidate if possible
            value = values[i-1]
            if value <= j and dp[i-1][j-value] != -float('inf'):
                if dp[i-1][j-value] + value > dp[i][j]:
                    dp[i][j] = dp[i-1][j-value] + value
                    choices[(i, j)] = True  # Mark that we included this candidate
                else:
                    choices[(i, j)] = False
            else:
                choices[(i, j)] = False
    
    # Check if we have an exact solution
    if dp[len(candidates)][int(demand_reversed)] == demand_reversed:
        # Reconstruct the solution
        solution = {}
        i, j = len(candidates), int(demand_reversed)
        while i > 0 and j > 0:
            if choices.get((i, j), False):
                solution[candidates[i-1]] = candidate_neighbor[candidates[i-1]]
                j -= values[i-1]
            i -= 1
        return solution
    
    # If no exact solution, find minimal set that exceeds target
    # Using a greedy approach (taking largest values first)
    greedy_solution = {}
    current_sum = 0
    
    for i, candidate in enumerate(sorted_candidates):
        if current_sum >= demand_reversed:
            break
        flow_amount = min(candidate_neighbor[candidate], demand_reversed - current_sum)
        greedy_solution[candidate] = flow_amount
        current_sum += sorted_values[i]
    
    return greedy_solution


def choose_neighbor(list_neighbor: Dict[str, int], demand_reversed: int) -> Dict[str, int]:
    """
    Choose neighbors to satisfy the reversed demand.
    
    Args:
        list_neighbor: Dictionary of neighbors with flow amounts
        demand_reversed: Demand to be satisfied
        
    Returns:
        Dictionary of selected neighbors with flow amounts
    """
    selected_neighbor = {}
    candidate_neighbor = {}
    
    for neighbor, flow_amount in list_neighbor.items():
        if flow_amount == demand_reversed:
            selected_neighbor[neighbor] = flow_amount
            return selected_neighbor
        candidate_neighbor[neighbor] = flow_amount
    
    selected_neighbor = subset_sum(candidate_neighbor, demand_reversed)
    return selected_neighbor


def UE_packing(G: Graph, commodities: Dict[str, int]) -> Tuple[Dict[int, Tuple[int, List[str]]], List[str]]:
    """
    Apply bin packing to user equipment demands.
    
    Args:
        G: Network graph
        commodities: Dictionary mapping UE IDs to their demands
        
    Returns:
        Tuple containing:
        - Dictionary mapping pack index to (capacity, list of UEs)
        - List of UEs that couldn't be packed
    """
    # Get UE_pack by greedy
    remaining_demand = sum(commodities.values())
    flow_dict = {}
    pack_capacity = []
    pack_index = 0
    
    while remaining_demand > 0:
        G_residual = create_residual_graph(G, flow_dict)
        path, demand = find_shortest_path(G_residual, False, None)
        
        if path is None:
            print(f"UE_packing(): Unable to satisfy remaining demand of {remaining_demand}")
            break
            
        new_update_flow(flow_dict, path, pack_index)
        remaining_demand -= demand
        pack_capacity.append(demand)
        pack_index += 1

    # Allocate UE to adjust UE_pack
    remaining_capacity_of_pack = {index: value for index, value in enumerate(pack_capacity)}
    sorted_commodities = {k: v for k, v in sorted(commodities.items(), key=lambda item: item[1], reverse=True)}

    UE_pack = {index: (capacity, []) for index, capacity in enumerate(pack_capacity)}
    remaining_UE = []

    # First fit algorithm
    for com, demand_com in sorted_commodities.items():
        allocated = False
        for index, capacity in remaining_capacity_of_pack.items():
            if capacity >= demand_com:
                allocated = True
                UE_pack[index][1].append(com)
                remaining_capacity_of_pack[index] -= demand_com
                break
        if not allocated:
            remaining_UE.append(com)

    return UE_pack, remaining_UE


def pick_path(path_d: List, remaining_demand: int) -> Tuple[List, int]:
    """
    Pick the shortest path with the capacity as tiebreaker (prefer big capacity).
    
    Args:
        path_d: List of paths with different demands
        remaining_demand: Remaining demand to satisfy
        
    Returns:
        Tuple of (selected path, demand of the path)
    """
    path = None
    demand_path = None
    length = float('inf')
    
    for demand, p in enumerate(path_d):
        if demand == 0:
            continue
        if p is None or p == []:
            continue
        if demand > remaining_demand:
            break
            
        if len(p) < length:
            path = p
            demand_path = demand
            length = len(p)
        elif len(p) == length and demand > demand_path:
            path = p
            demand_path = demand
            length = len(p)

    return path, demand_path


def pack_method(G: Graph, commodities: Dict[int, int], file_flow_path: str, file_flow_HO: str) -> int:
    """
    Apply the Pack Method algorithm for satellite scheduling.
    
    This method groups user flows into packs and finds optimal paths to minimize handovers.
    
    Args:
        G: Network graph
        commodities: Dictionary mapping user IDs to their demands
        file_flow_path: Output file path for the flow paths
        file_flow_HO: Output file path for handover count
        
    Returns:
        Total number of handovers
    """
    # Input: G, commodities
    UE_pack, remaining_UE = UE_packing(G, commodities)

    # Order the pack
    sorted_items = sorted(
        UE_pack.items(),
        key=lambda item: (-item[1][0], -len(item[1][1]))  # Primary: -capacity, Secondary: -list length
    )
    sorted_UE_pack = dict(sorted_items)
    flow_dict = {}  # flow for each pack

    # Find paths based on pack_capacity
    # format: sorted_UE_pack[index_pack] = (demand_pack, [list of UEs in the pack])
    for index_pack, pack in sorted_UE_pack.items():
        G_residual = create_residual_graph(G, flow_dict)
        path, capacity = find_shortest_path(G_residual, True, pack[0])
        
        if path is None:
            print(f"pack_method(): Unable to find path for pack {index_pack} with demand {pack[0]}")
            continue
            
        new_update_flow(flow_dict, path, index_pack)

    # Allocate the paths based on UE_pack
    flow_UE = {}  # flow for each UE
    for index_pack, flow_pack in flow_dict.items():
        for u, dict_u in flow_pack.items():
            for v, flow_amount in dict_u.items():
                for ue in UE_pack[index_pack][1]:
                    if ue not in flow_UE:
                        flow_UE[ue] = {}
                    if u not in flow_UE[ue]:
                        flow_UE[ue][u] = {}
                    flow_UE[ue][u][v] = commodities[ue]

    # Find flow for remaining_UE
    if remaining_UE:
        for ue in remaining_UE:
            print(f"pack_method(): Finding paths for UE {ue}")
            G_residual = create_residual_graph(G, flow_UE)
            path, capacity = find_shortest_path(G_residual, True, commodities[ue])
            
            if path:
                new_update_flow(flow_UE, path, ue)
            else:
                target_demand = commodities[ue]
                served_demand = 0

                flow_tmp = copy.deepcopy(flow_UE)
                virtual_ue_index = len(commodities.keys())
                frac_flow_for_ue = {}  # for create_residual_graph_for_split()
                num_path = 0
                
                while served_demand < target_demand:
                    path_d = [[] for _ in range(target_demand - served_demand + 1)]
                    has_path = False

                    for d in range(1, target_demand - served_demand + 1):
                        path_d[d], capacity = find_shortest_path(G_residual, True, d)
                        if path_d[d] is not None:
                            has_path = True
                    
                    if not has_path:
                        print(f"pack_method(): No fractional flow for UE {ue}")
                        break
                    
                    path, demand_path = pick_path(path_d, target_demand - served_demand)
                    served_demand += demand_path

                    num_path += 1

                    # Update fractional flow
                    new_update_flow(flow_tmp, path, virtual_ue_index)
                    new_update_flow(frac_flow_for_ue, path, num_path)
                    G_residual = create_residual_graph_for_split(G, flow_tmp, frac_flow_for_ue)     

                # Combine flow_UE with frac_flow_for_ue
                flow_UE[ue] = {}
                for path in frac_flow_for_ue.values():
                    for u, dict_u in path.items():
                        if u not in flow_UE[ue]:
                            flow_UE[ue][u] = {}
                        
                        for v, amount in dict_u.items():
                            if v not in flow_UE[ue][u]:
                                flow_UE[ue][u][v] = 0
                            flow_UE[ue][u][v] += amount                   

    # Sort and output the results
    sorted_flow_UE = dict(sorted(flow_UE.items()))
    total_handovers = process_and_output_flow(sorted_flow_UE, file_flow_path, file_flow_HO)
    print(f"pack.pack_method(): write result to {file_flow_path} and {file_flow_HO}")

    return total_handovers