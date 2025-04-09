"""
Satellite network simulation module.

This module contains simulation functions for testing different algorithms
for satellite handover optimization in dynamic satellite networks.
"""

"""
Satellite network simulation module.

This module contains simulation functions for testing different algorithms
for satellite handover optimization in dynamic satellite networks.
"""

"""
Satellite network simulation module.

This module contains simulation functions for testing different algorithms
for satellite handover optimization in dynamic satellite networks.
"""

import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Any, Set

# Import core modules
from satellite_scheduling.core.graph import Graph
from satellite_scheduling.core.edge_types import EdgeType
import satellite_scheduling.core.problem_generator as gen_problem

# Import utility functions directly (not through utils/__init__.py)
from satellite_scheduling.utils.io_utils import save_2d_list, load_2d_list

# Import algorithm functions directly
from satellite_scheduling.algorithms.mvt import MVT
from satellite_scheduling.algorithms.mac import MAC
from satellite_scheduling.algorithms.msh import new_MSH
from satellite_scheduling.algorithms.pack_method import pack_method


def set_uniform_node_capacity(sat_t: List[List[int]], node_capacity: int) -> Tuple[Dict[str, int], List[str]]:
    """
    Set uniform capacity for all satellites in the network.
    
    Args:
        sat_t: List of lists where sat_t[t] represents satellites visible at time t
        node_capacity: Capacity value to assign to each satellite
        
    Returns:
        Tuple containing dictionary mapping satellite name to capacity and list of all satellites
    """
    list_sat = []

    for time_slot in sat_t:
        for sat in time_slot:
            if str(sat) not in list_sat:
                list_sat.append(str(sat))

    dict_node_capacity = {}
    for sat in list_sat:
        dict_node_capacity[sat] = node_capacity

    return dict_node_capacity, list_sat


def get_cut(sat_t: List[List[int]], sim_time: int, node_capacity: int) -> int:
    """
    Calculate the minimum cut in the network over time.
    
    Args:
        sat_t: List of lists where sat_t[t] represents satellites visible at time t
        sim_time: Total simulation time
        node_capacity: Capacity of each satellite
        
    Returns:
        Value of the minimum cut
    """
    min_cut = math.inf

    for t in range(sim_time-1):
        sat_current = set(sat_t[t])
        sat_next = set(sat_t[t+1])
        intersection = sat_current.intersection(sat_next)
        cut = len(intersection) * node_capacity
        if cut < min_cut:
            min_cut = cut

    return min_cut


def generate_bin_packing_instance(n_items: int = 50, 
                                 node_capacity: int = 100, 
                                 cut: Optional[int] = None, 
                                 seed: int = 42) -> Dict[int, int]:
    """
    Generate a challenging bin packing problem instance with total weight equal to cut.
    
    Args:
        n_items: Number of items to generate
        node_capacity: Capacity of each node
        cut: Target total weight (if None, uses n_items // 2 * node_capacity)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping item IDs to their weights/demands
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


def sat_t_to_string(sat_t: List[List[int]]) -> List[List[str]]:
    """
    Convert satellite IDs from integers to strings.
    
    Args:
        sat_t: List of lists where sat_t[t] represents satellites visible at time t as integers
        
    Returns:
        List of lists where satellites are represented as strings
    """
    str_sat_t = [[] for _ in range(len(sat_t))]
    for t in range(len(sat_t)):
        for index in range(len(sat_t[t])):
            str_sat_t[t].append(str(sat_t[t][index]))
    return str_sat_t


def ST_DAG(sim_time: int, node_capacity: int, num_ue: int, num_st: int, 
           ST_con_file: str, ST_cul_file: str) -> Tuple[Graph, List[List[int]]]:
    """
    Create a Directed Acyclic Graph (DAG) from satellite timeline data.
    
    Args:
        sim_time: Total simulation time
        node_capacity: Capacity of each satellite node
        num_ue: Number of user equipments
        num_st: Number of satellites
        ST_con_file: Path to satellite connectivity file
        ST_cul_file: Path to satellite culmination file
        
    Returns:
        Tuple containing Graph object and satellite timeline
    """
    G = Graph()
    df = pd.read_csv(ST_con_file).T
    cul = pd.read_csv(ST_cul_file).T
    capacity = node_capacity * num_st
    t_sat = [[] for _ in range(sim_time)]

    for t in range(sim_time):
        for sat in range(num_st):
            if df[t].iloc[sat] == 1:
                t_sat[t].append(sat)

    for i in range(num_st):
        for j in range(num_st):
            if i == j:
                pass           
            else:
                for k in range(sim_time):
                    if df[k].iloc[i] == 0:
                        pass
                    elif df[k].iloc[i] == 1 and df[k].iloc[j] == 0:
                        pass
                    elif df[k].iloc[i] == 1 and df[k].iloc[j] == 1 and cul[k].iloc[i] > cul[k].iloc[j]:
                        v_i = str(i)
                        v_j = str(j)
                        G.add_node(v_i)
                        G.set_node_capacity(v_i, node_capacity)
                        G.add_node(v_j)
                        G.set_node_capacity(v_j, node_capacity)
                        G.add_edge(v_i, v_j, None, EdgeType.FORWARD, capacity, 1)
                        break
    
    return G, t_sat


def simulation_1(t_sat_file: str = "t_sat.txt") -> None:
    """
    Basic simulation to test graph construction and minimum cut calculation.
    
    Args:
        t_sat_file: Path to satellite timeline file
    """
    sim_time = 3600
    node_capacity = 40
    num_st = 298
    seed = 50

    # Load satellite visibility timeline
    sat_t = load_2d_list(t_sat_file)

    # Construct graph G from sat_t
    dict_node_capacity, list_sat = set_uniform_node_capacity(sat_t, node_capacity)
    coverage_period = gen_problem.get_coverage_period(sat_t, list_sat, sim_time)
    edge_cap = node_capacity * num_st + 1
    G = gen_problem.construct_graph(num_st, sim_time, coverage_period, dict_node_capacity, list_sat, edge_cap)

    # Generate UE requests
    cut = get_cut(sat_t, sim_time, node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=node_capacity, cut=cut, seed=seed)

    print(f"Simulation 1: Graph constructed with {len(G.nodes)} nodes and {len(G.edges)} edges")
    print(f"Minimum cut value: {cut}")
    print(f"Total demand: {sum(commodities.values())}")


def simulation_node_capacity(data_dir: str = "result/node_capacity", 
                            t_sat_file: str = "t_sat.txt") -> None:
    """
    Run simulation comparing algorithms with varying node capacity.
    
    Args:
        data_dir: Directory to store simulation results
        t_sat_file: Path to satellite visibility data
    """
    # Ensure directories exist
    os.makedirs(f"{data_dir}/path", exist_ok=True)
    os.makedirs(f"{data_dir}/HO", exist_ok=True)
    
    # Load satellite visibility data
    t_sat = load_2d_list(t_sat_file)
    str_sat_t = sat_t_to_string(t_sat)

    # Set simulation parameters
    sim_time = 3600
    num_st = 298
    seed = 10
    initial_node_capacity = 40

    # Generate commodities (node_capacity)
    cut = get_cut(t_sat, sim_time, initial_node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=initial_node_capacity, cut=cut, seed=seed)

    # Generate set_node_capacity
    num_cap = 10
    list_node_capacity = [initial_node_capacity + i*5 for i in range(num_cap)]

    # For each node capacity
    for node_capacity in list_node_capacity:
        print(f"Running simulation with node_capacity = {node_capacity}")
        
        # Construct graph G
        dict_node_capacity, list_sat = set_uniform_node_capacity(t_sat, node_capacity)
        coverage_period = gen_problem.get_coverage_period(t_sat, list_sat, sim_time)
        edge_cap = node_capacity * num_st + 1
        G = gen_problem.construct_graph(num_st, sim_time, coverage_period, dict_node_capacity, list_sat, edge_cap)

        # Run MVT algorithm
        file_MVT_path = f"{data_dir}/path/path_MVT_{node_capacity}.txt"
        file_MVT_HO = f"{data_dir}/HO/HO_MVT_{node_capacity}.txt"
        MVT(str_sat_t, dict_node_capacity, commodities, file_MVT_path, file_MVT_HO)

        # Run MAC algorithm
        file_MAC_path = f"{data_dir}/path/path_MAC_{node_capacity}.txt"
        file_MAC_HO = f"{data_dir}/HO/HO_MAC_{node_capacity}.txt"
        MAC(str_sat_t, dict_node_capacity, commodities, file_MAC_path, file_MAC_HO)

        # Run pack_method algorithm
        file_pack_path = f"{data_dir}/path/path_pack_{node_capacity}.txt"
        file_pack_HO = f"{data_dir}/HO/HO_pack_{node_capacity}.txt"
        pack_method(G, commodities, file_pack_path, file_pack_HO)
        
        # Run MSH algorithm
        file_MSH_path = f"{data_dir}/path/path_MSH_{node_capacity}.txt"
        file_MSH_HO = f"{data_dir}/HO/HO_MSH_{node_capacity}.txt"
        new_MSH(G, commodities, file_MSH_path, file_MSH_HO)
        
    print("Node capacity simulation completed.")


def simulation_ue_request(data_dir: str = "result/UE_request", 
                         t_sat_file: str = "t_sat.txt") -> None:
    """
    Run simulation comparing algorithms with varying UE request amounts.
    
    Args:
        data_dir: Directory to store simulation results
        t_sat_file: Path to satellite visibility data
    """
    # Ensure directories exist
    os.makedirs(f"{data_dir}/path", exist_ok=True)
    os.makedirs(f"{data_dir}/HO", exist_ok=True)
    
    # Load satellite visibility data
    sat_t = load_2d_list(t_sat_file)
    str_sat_t = sat_t_to_string(sat_t)

    # Set simulation parameters
    sim_time = 3600
    num_st = 298
    seed = 10
    node_capacity = 40

    # Construct graph G
    dict_node_capacity, list_sat = set_uniform_node_capacity(sat_t, node_capacity)
    coverage_period = gen_problem.get_coverage_period(sat_t, list_sat, sim_time)
    edge_cap = node_capacity * num_st + 1
    G = gen_problem.construct_graph(num_st, sim_time, coverage_period, dict_node_capacity, list_sat, edge_cap)

    # Generate cut_list
    initial_cut = get_cut(sat_t, sim_time, node_capacity)
    num_instance = 10
    cut_list = [initial_cut - i*10 for i in range(num_instance)]

    # For each UE request amount
    for cut in cut_list:
        print(f"Running simulation with cut = {cut}")
        
        # Generate commodities for different cut
        commodities = generate_bin_packing_instance(node_capacity=node_capacity, cut=cut, seed=seed)

        # Run MVT algorithm
        file_MVT_path = f"{data_dir}/path/path_MVT_{cut}.txt"
        file_MVT_HO = f"{data_dir}/HO/HO_MVT_{cut}.txt"
        MVT(str_sat_t, dict_node_capacity, commodities, file_MVT_path, file_MVT_HO)

        # Run MAC algorithm
        file_MAC_path = f"{data_dir}/path/path_MAC_{cut}.txt"
        file_MAC_HO = f"{data_dir}/HO/HO_MAC_{cut}.txt"
        MAC(str_sat_t, dict_node_capacity, commodities, file_MAC_path, file_MAC_HO)

        # Run pack_method algorithm
        file_pack_path = f"{data_dir}/path/path_pack_{cut}.txt"
        file_pack_HO = f"{data_dir}/HO/HO_pack_{cut}.txt"
        pack_method(G, commodities, file_pack_path, file_pack_HO)
        
        # Run MSH algorithm
        file_MSH_path = f"{data_dir}/path/path_MSH_{cut}.txt"
        file_MSH_HO = f"{data_dir}/HO/HO_MSH_{cut}.txt"
        new_MSH(G, commodities, file_MSH_path, file_MSH_HO)
        
    print("UE request simulation completed.")


def simulation_4(t_sat_file: str = "t_sat.txt") -> None:
    """
    Test flow_method simulation.
    
    Args:
        t_sat_file: Path to satellite timeline file
    """
    sim_time = 3600
    num_st = 298
    seed = 10
    node_capacity = 40

    # Load sat_t
    sat_t = load_2d_list(t_sat_file)

    # Construct graph G by sat_t
    dict_node_capacity, list_sat = set_uniform_node_capacity(sat_t, node_capacity)
    coverage_period = gen_problem.get_coverage_period(sat_t, list_sat, sim_time)
    
    edge_cap = node_capacity * num_st + 1
    G = gen_problem.construct_graph(num_st, sim_time, coverage_period, dict_node_capacity, list_sat, edge_cap)

    # Generate UE requests
    cut = get_cut(sat_t, sim_time, node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=node_capacity, cut=cut, seed=seed)

    total_demand = sum(commodities.values())
    print("simulation_4():")
    print(f"cut = {cut}")
    print(f"total_demand = {total_demand}")

    # Run flow algorithm if it exists
    try:
        import flow
        file_flow_path = "path_flow.txt"
        file_flow_HO = "HO_flow.txt"
        flow.flow_method(G, commodities, file_flow_path, file_flow_HO)
    except ImportError:
        print("Flow module not found. Skipping flow_method.")


def simulation_6(t_sat_file: str = "t_sat.txt") -> None:
    """
    Test pack_method implementation.
    
    Args:
        t_sat_file: Path to satellite timeline file
    """
    sat_t = load_2d_list(t_sat_file)

    sim_time = 3600
    node_capacity = 50
    num_st = 298
    seed = 10

    # Construct graph G by sat_t
    dict_node_capacity, list_sat = set_uniform_node_capacity(sat_t, node_capacity)
    coverage_period = gen_problem.get_coverage_period(sat_t, list_sat, sim_time)
    edge_cap = node_capacity * num_st + 1
    G = gen_problem.construct_graph(num_st, sim_time, coverage_period, dict_node_capacity, list_sat, edge_cap)

    cut = get_cut(sat_t, sim_time, node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=node_capacity, cut=cut, seed=seed)

    pack_method.pack_method(G, commodities, "path_oldpack.txt", "HO_oldpack.txt")


def simulation_8(t_sat_file: str = "t_sat.txt") -> None:
    """
    Test MVT implementation.
    
    Args:
        t_sat_file: Path to satellite timeline file
    """
    t_sat = load_2d_list(t_sat_file)
    str_sat_t = sat_t_to_string(t_sat)

    # Set simulation parameters
    sim_time = 3600
    num_st = 298
    seed = 10
    initial_node_capacity = 40

    # Generate commodities (node_capacity)
    cut = get_cut(t_sat, sim_time, initial_node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=initial_node_capacity, cut=cut, seed=seed)

    dict_node_capacity, list_sat = set_uniform_node_capacity(t_sat, initial_node_capacity)

    file_MAC_path = f"path_MAC.txt"
    file_MAC_HO = f"HO_MAC.txt"
    mvt.MVT(str_sat_t, dict_node_capacity, commodities, file_MAC_path, file_MAC_HO)


def simulation_9(t_sat_file: str = "t_sat.txt") -> None:
    """
    Test gen_problem.construct_graph_v2() + pack_method().
    
    Args:
        t_sat_file: Path to satellite timeline file
    """
    t_sat = load_2d_list(t_sat_file)

    sim_time = 3600
    node_capacity = 50
    seed = 10

    reduce_ratio = 30
    G = gen_problem.construct_graph_v2(t_sat, node_capacity, reduce_ratio)

    cut = get_cut(t_sat, sim_time, node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=node_capacity, cut=cut, seed=seed)

    total_demand = sum(d for d in commodities.values())
    print(f"total_demand = {total_demand}")

    pack_method.pack_method(G, commodities, "path_pack.txt", "HO_pack.txt")


def simulation_10(t_sat_file: str = "t_sat.txt") -> None:
    """
    Test MSH algorithm.
    
    Args:
        t_sat_file: Path to satellite timeline file
    """
    t_sat = load_2d_list(t_sat_file)
    str_sat_t = sat_t_to_string(t_sat)

    sim_time = 3600
    num_st = 298
    seed = 10
    initial_node_capacity = 40
    
    cut = get_cut(t_sat, sim_time, initial_node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=initial_node_capacity, cut=cut, seed=seed)

    dict_node_capacity, list_sat = set_uniform_node_capacity(t_sat, initial_node_capacity)
    coverage_period = gen_problem.get_coverage_period(t_sat, list_sat, sim_time)
    edge_cap = initial_node_capacity * num_st + 1
    G = gen_problem.construct_graph(num_st, sim_time, coverage_period, dict_node_capacity, list_sat, edge_cap)

    file_MSH_path = f"test_MSH_{initial_node_capacity}.txt"
    file_MSH_HO = f"test_MSH_{initial_node_capacity}.txt"
    msh.new_MSH(G, commodities, file_MSH_path, file_MSH_HO)


def plot_results(directory: str, name: str, output_file: Optional[str] = None) -> None:
    """
    Plot the comparison of handover counts from different algorithms.
    
    Args:
        directory: Directory containing handover count results
        name: Name for the x-axis (e.g., "node capacity" or "UE request")
        output_file: File to save the plot to (if None, uses default name)
    """
    pack_path = f"{directory}/HO_pack_*.txt"
    mvt_path = f"{directory}/HO_MVT_*.txt"
    mac_path = f"{directory}/HO_MAC_*.txt"
    msh_path = f"{directory}/HO_MSH_*.txt"
    
    # Get all files matching the patterns
    pack_files = sorted(glob(pack_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mvt_files = sorted(glob(mvt_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mac_files = sorted(glob(mac_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    msh_files = sorted(glob(msh_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    list_x = []
    list_flow_data = []
    list_mvt_data = []
    list_mac_data = []
    list_msh_data = []

    for pack_file, mvt_file, mac_file, msh_file in zip(pack_files, mvt_files, mac_files, msh_files):
        # Extract integer from filename (taking the number before .txt)
        number = int(pack_file.split('_')[-1].replace('.txt', ''))
        list_x.append(number)
        list_flow_data.append(np.loadtxt(pack_file))
        list_mvt_data.append(np.loadtxt(mvt_file))
        list_mac_data.append(np.loadtxt(mac_file))
        list_msh_data.append(np.loadtxt(msh_file))

    # Convert lists to numpy arrays
    x = np.array(list_x)
    pack_data_array = np.array(list_flow_data)
    mvt_data_array = np.array(list_mvt_data)
    mac_data_array = np.array(list_mac_data)
    msh_data_array = np.array(list_msh_data)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot the lines
    plt.plot(x, pack_data_array, 'b-', label='Pack Method', linewidth=2)
    plt.plot(x, mvt_data_array, 'r--', label='MVT', linewidth=2)
    plt.plot(x, mac_data_array, 'g:', label='MAC', linewidth=2)
    plt.plot(x, msh_data_array, 'm-', label='MSH', linewidth=2)

    # Customize the plot
    plt.title('Performance Comparison', fontsize=14, pad=15)
    plt.xlabel(f"{name}", fontsize=12)
    plt.ylabel('Number of handovers', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Add x-ticks for each file
    plt.xticks(x)

    # Set x-axis limits based on list_x values
    plt.xlim(min(list_x), max(list_x))

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    if output_file is None:
        output_file = f"HO v.s. {name}.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print data arrays for verification
    print("\nPack Data Array:")
    print(pack_data_array)
    print("\nMVT Data Array:")
    print(mvt_data_array)
    print("\nMAC Data Array:")
    print(mac_data_array)
    print("\nMSH Data Array:")
    print(msh_data_array)

    # Clear the plot from memory
    plt.close()


def run_all_simulations(data_dir: str = "result", t_sat_file: str = "t_sat.txt") -> None:
    """
    Run all simulations and generate plots.
    
    Args:
        data_dir: Base directory for all results
        t_sat_file: Path to satellite visibility data
    """
    # Run simulations
    node_capacity_dir = f"{data_dir}/node_capacity"
    ue_request_dir = f"{data_dir}/UE_request"
    
    simulation_node_capacity(data_dir=node_capacity_dir, t_sat_file=t_sat_file)
    simulation_ue_request(data_dir=ue_request_dir, t_sat_file=t_sat_file)
    
    # Generate plots
    plot_results(f"{node_capacity_dir}/HO", "node capacity")
    plot_results(f"{ue_request_dir}/HO", "UE request")
    
    print("All simulations and visualizations completed.")


if __name__ == "__main__":
    # Example usage
    run_all_simulations()