import pandas as pd
import math
import random
import networkx as nx
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np

import problem_generator_v2 as gen_problem
import MVT
import MAC
import MSH
import pack_method as pack

from typing import Dict, List
from glob import glob
from pathlib import Path


def ST_DAG(sim_time:int, node_capacity:int,num_ue:int, num_st:int, ST_con_file, ST_cul_file):
    G = gen_problem.Graph()
    df=pd.read_csv(ST_con_file).T
    cul=pd.read_csv(ST_cul_file).T
    capacity = node_capacity*num_st
    t_sat = [[] for _ in range(sim_time)]

    for t in range(sim_time):
        for sat in range(num_st):
            if(df[t].iloc[sat]==1):
                t_sat[t].append(sat)

    for i in range(num_st):
        for j in range(num_st):
            if i == j:
                pass           
            else:
                for k in range(sim_time):
                    if df[k].iloc[i]==0:
                        pass
                    elif df[k].iloc[i]==1 and df[k].iloc[j]==0:
                        pass
                    elif  df[k].iloc[i]==1 and df[k].iloc[j]==1 and cul[k].iloc[i] > cul[k].iloc[j] :
                        #G.add_nodes_from([(i,  {"ID": i, 'repeat':0,'node_c':node_capacity, 'copy':False})])
                        #G.add_nodes_from([(j,  {"ID": j, 'repeat':0,'node_c':node_capacity, 'copy':False})])                                
                        #G.add_weighted_edges_from([(j, i, 1)])
                        v_i = str(i)
                        v_j = str(j)
                        G.add_node(v_i)
                        G.set_node_capacity(v_i,node_capacity)
                        G.add_node(v_j)
                        G.set_node_capacity(v_j,node_capacity)
                        G.add_edge(v_i,v_j,None,gen_problem.EdgeType.FORWARD,capacity,1)
                        break
    
    return G, t_sat


def load_graph(filename: str):
    """
    Load a graph from a file that was saved using save_graph
    """
    
    graph = gen_problem.Graph()
    
    # Use pickle to load the graph
    with open(filename, 'rb') as f:
        nx_graph = pickle.load(f)
    
    for node_id, node_data in nx_graph.nodes(data=True):
        graph.add_node(str(node_id))
        if node_data.get('capacity') is not None:
            graph.set_node_capacity(str(node_id), node_data['capacity'])
    
    for u, v, edge_data in nx_graph.edges(data=True):
        graph.add_edge(
            str(u),
            str(v),
            edge_data.get('id'),
            edge_data.get('type'),
            edge_data.get('capacity'),
            edge_data.get('length')
        )
    
    return graph


def save_2d_list(t_sat: list, filename: str):
    """
    Save a 2D list to a file
    Args:
        t_sat: 2D list to save
        filename: name of the file to save to
    """
    with open(filename, 'w') as f:
        # Write number of rows and columns at the start
        rows = len(t_sat)
        cols = len(t_sat[0]) if rows > 0 else 0
        f.write(f"{rows} {cols}\n")
        
        # Write the data
        for row in t_sat:
            # Convert numbers to strings and join with spaces
            line = ' '.join(str(x) for x in row)
            f.write(line + '\n')


def load_2d_list(filename: str) -> list:
    """
    Load a 2D list from a file
    Args:
        filename: name of the file to load from
    Returns:
        2D list loaded from file
    """
    with open(filename, 'r') as f:
        # Read dimensions from first line
        rows, cols = map(int, f.readline().split())
        
        # Initialize empty 2D list
        t_sat = []
        
        # Read each line and convert to numbers
        for _ in range(rows):
            row = list(map(int, f.readline().split()))
            t_sat.append(row)
            
    return t_sat


def get_cut(sat_t,sim_time,node_capacity):
    min_cut = math.inf

    for t in range(sim_time-1):
        sat_current = set(sat_t[t])
        sat_next = set(sat_t[t+1])
        intersection = sat_current.intersection(sat_next)
        cut = len(intersection)*node_capacity
        if(cut<min_cut):
            min_cut = cut

    return min_cut


def generate_bin_packing_instance(n_items=50, node_capacity=100, cut=None, seed=42):
    """
    Generates a challenging bin packing problem instance with total weight exactly equal to cut.
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


def set_uniform_node_capacity(sat_t: List[List[int]], node_capacity: int):
    list_sat = []

    for list in sat_t:
        for sat in list:
            if(str(sat) not in list_sat):
                list_sat.append(str(sat))

    dict_node_capacity = {}
    for sat in list_sat:
        dict_node_capacity[sat] = node_capacity

    return dict_node_capacity, list_sat


def simulation_1():
    sim_time = 3600
    node_capacity = 40
    num_st = 298
    seed = 50

    #G, t_sat = ST_DAG(sim_time, node_capacity, num_ue, num_st, 
    #           ST_con_file= 'ST_298_space_6360_time/ST_connect',
    #           ST_cul_file ='ST_298_space_6360_time/ST_cul_time',
    #           )

    #G.save_graph("mygraph.gpickle")
    #save_2d_list(t_sat, "t_sat.txt")
    #G = load_graph("mygraph.gpickle")

    #load sat_t
    sat_t = load_2d_list("t_sat.txt")

    #construct graph G by sat_t
    dict_node_capacity = {}
    dict_node_capacity, list_sat = set_uniform_node_capacity(sat_t, node_capacity)
    coverage_period = gen_problem.get_coverage_period(sat_t,list_sat,sim_time)
    edge_cap = node_capacity*num_st+1
    G = gen_problem.construct_graph(num_st,sim_time,coverage_period,dict_node_capacity,list_sat,edge_cap)

    #generate UE requests
    cut = get_cut(sat_t, sim_time, node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=node_capacity,cut=cut, seed=seed)

    '''
    def sat_t_to_string(sat_t):
        for t in range(len(sat_t)):
            for index in range(len(sat_t[t])):
                sat_t[t][index] = str(sat_t[t][index])
    sat_t_to_string(sat_t)

    assignment = MVT.MVT(sat_t,dict_node_capacity,commodities, 40)
    '''

    #OPT.OPT(G,commodities)
    #heuristic.heuristic_v2(G,commodities,bin_capacity,K)

    return


def sat_t_to_string(sat_t):
    str_sat_t = [[] for _ in range(len(sat_t))]
    for t in range(len(sat_t)):
        for index in range(len(sat_t[t])):
            str_sat_t[t].append(str(sat_t[t][index]))
    return str_sat_t


#num_HO v.s. node capacity
def simulation_3():
    t_sat = load_2d_list("t_sat.txt")
    str_sat_t = sat_t_to_string(t_sat)

    #set simulation parameter
    sim_time = 3600
    num_st = 298
    seed = 10
    initial_node_capacity = 40
    #initial_node_capacity = 45
    #reduce_ratio = 30

    #generate commodities (node_capacity)
    cut = get_cut(t_sat, sim_time, initial_node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=initial_node_capacity,cut=cut, seed=seed)

    #generate set_node_capacity
    num_cap = 10
    list_node_capacity = [initial_node_capacity+i*5 for i in range(num_cap)]
    #list_node_capacity = [initial_node_capacity]

    '''
    #for each node capacity
    for node_capacity in list_node_capacity:
        #construct graph G
        dict_node_capacity = {}
        dict_node_capacity, list_sat = set_uniform_node_capacity(t_sat, node_capacity)
        coverage_period = gen_problem.get_coverage_period(t_sat,list_sat,sim_time)
        edge_cap = node_capacity*num_st+1
        G = gen_problem.construct_graph(num_st,sim_time,coverage_period,dict_node_capacity,list_sat,edge_cap)

        file_MVT_path = f"path_MVT_{node_capacity}.txt"
        file_MVT_HO = f"HO_MVT_{node_capaci ty}.txt"
        MVT.MVT(str_sat_t, dict_node_capacity,commodities,file_MVT_path,file_MVT_HO)

        file_MAC_path = f"path_MAC_{node_capacity}.txt"
        file_MAC_HO = f"HO_MAC_{node_capacity}.txt"
        MAC.MAC(str_sat_t, dict_node_capacity, commodities, file_MAC_path, file_MAC_HO)

        file_pack_path = f"path_pack_{node_capacity}.txt"
        file_pack_HO = f"HO_pack_{node_capacity}.txt"
        pack.pack_method(G,commodities,file_pack_path,file_pack_HO)

        file_MSH_path = f"path_MSH_{node_capacity}.txt"
        file_MSH_HO = f"HO_MSH_{node_capacity}.txt"
        #MSH.MSH(G, commodities, file_MSH_path, file_MSH_HO)
        MSH.new_MSH(G,commodities,file_MSH_path,file_MSH_HO)
    '''

    #for each node capacity
    for node_capacity in list_node_capacity:
        #construct graph G
        dict_node_capacity = {}
        dict_node_capacity, list_sat = set_uniform_node_capacity(t_sat, node_capacity)
        coverage_period = gen_problem.get_coverage_period(t_sat,list_sat,sim_time)
        edge_cap = node_capacity*num_st+1
        G = gen_problem.construct_graph(num_st,sim_time,coverage_period,dict_node_capacity,list_sat,edge_cap)

        file_MVT_path = f"result/node_capacity/path/path_MVT_{node_capacity}.txt"
        file_MVT_HO = f"result/node_capacity/HO/HO_MVT_{node_capacity}.txt"
        MVT.MVT(str_sat_t, dict_node_capacity,commodities,file_MVT_path,file_MVT_HO)

        file_MAC_path = f"result/node_capacity/path/path_MAC_{node_capacity}.txt"
        file_MAC_HO = f"result/node_capacity/HO/HO_MAC_{node_capacity}.txt"
        MAC.MAC(str_sat_t, dict_node_capacity, commodities, file_MAC_path, file_MAC_HO)

        file_pack_path = f"result/node_capacity/path/path_pack_{node_capacity}.txt"
        file_pack_HO = f"result/node_capacity/HO/HO_pack_{node_capacity}.txt"
        pack.pack_method(G,commodities,file_pack_path,file_pack_HO)

        file_MSH_path = f"result/node_capacity/path/path_MSH_{node_capacity}.txt"
        file_MSH_HO = f"result/node_capacity/HO/HO_MSH_{node_capacity}.txt"
        #MSH.MSH(G, commodities, file_MSH_path, file_MSH_HO)
        MSH.new_MSH(G,commodities,file_MSH_path,file_MSH_HO)


#test flow_method
def simulation_4():
    sim_time = 3600
    num_st = 298
    seed = 10
    node_capacity = 40

    #load sat_t
    sat_t = load_2d_list("t_sat.txt")

    #construct graph G by sat_t
    dict_node_capacity = {}
    dict_node_capacity, list_sat = set_uniform_node_capacity(sat_t, node_capacity)
    coverage_period = gen_problem.get_coverage_period(sat_t,list_sat,sim_time)
    
    edge_cap = node_capacity*num_st+1
    G = gen_problem.construct_graph(num_st,sim_time,coverage_period,dict_node_capacity,list_sat,edge_cap)

    #generate UE requests
    cut = get_cut(sat_t, sim_time, node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=node_capacity,cut=cut, seed=seed)

    total_demand = sum(commodities.values())
    print("simulation_4():")
    print(f"cut = {cut}")
    print(f"total_demand = {total_demand}")

    #flow
    file_flow_path = "path_flow.txt"
    file_flow_HO = "HO_flow.txt"
    flow.flow_method(G,commodities,file_flow_path,file_flow_HO)

    return


#num_HO v.s UE request
def simulation_5():
    sat_t = load_2d_list("t_sat.txt")
    str_sat_t = sat_t_to_string(sat_t)

    #set simulation parameter
    sim_time = 3600
    num_st = 298
    seed = 10
    node_capacity = 40
    #reduce_ratio = 30

    #construct graph G
    dict_node_capacity = {}
    dict_node_capacity, list_sat = set_uniform_node_capacity(sat_t, node_capacity)
    coverage_period = gen_problem.get_coverage_period(sat_t,list_sat,sim_time)
    edge_cap = node_capacity*num_st+1
    G = gen_problem.construct_graph(num_st,sim_time,coverage_period,dict_node_capacity,list_sat,edge_cap)

    #generate cut_list
    initial_cut = get_cut(sat_t, sim_time, node_capacity)
    num_instance = 10
    cut_list = [initial_cut-i*10 for i in range(num_instance)]

    #for each node capacity
    for cut in cut_list:
        #generate commodities for different cut
        commodities = generate_bin_packing_instance(node_capacity=node_capacity,cut=cut, seed=seed)

        file_MVT_path = f"result/UE_request/path/path_MVT_{cut}.txt"
        file_MVT_HO = f"result/UE_request/HO/HO_MVT_{cut}.txt"
        MVT.MVT(str_sat_t, dict_node_capacity, commodities, file_MVT_path, file_MVT_HO)

        file_MAC_path = f"result/UE_request/path/path_MAC_{cut}.txt"
        file_MAC_HO = f"result/UE_request/HO/HO_MAC_{cut}.txt"
        MAC.MAC(str_sat_t, dict_node_capacity, commodities, file_MAC_path, file_MAC_HO)

        file_pack_path = f"result/UE_request/path/path_pack_{cut}.txt"
        file_pack_HO = f"result/UE_request/HO/HO_pack_{cut}.txt"
        pack.pack_method(G,commodities,file_pack_path,file_pack_HO)
        
        file_MSH_path = f"result/UE_request/path/path_MSH_{cut}.txt"
        file_MSH_HO = f"result/UE_request/HO/HO_MSH_{cut}.txt"
        #MSH.MSH(G, commodities, file_MSH_path, file_MSH_HO)
        MSH.new_MSH(G,commodities,file_MSH_path,file_MSH_HO)
    return


#test pack_method
def simulation_6():
    sat_t = load_2d_list("t_sat.txt")

    sim_time = 3600
    node_capacity = 50
    num_st = 298
    seed = 10

    #construct graph G by sat_t
    dict_node_capacity = {}
    dict_node_capacity, list_sat = set_uniform_node_capacity(sat_t, node_capacity)
    coverage_period = gen_problem.get_coverage_period(sat_t,list_sat,sim_time)
    edge_cap = node_capacity*num_st+1
    G = gen_problem.construct_graph(num_st,sim_time,coverage_period,dict_node_capacity,list_sat,edge_cap)

    cut = get_cut(sat_t, sim_time, node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=node_capacity,cut=cut, seed=seed)

    pack.pack_method(G,commodities,"path_oldpack.txt","HO_oldpack.txt")

    return


#test MVT
def simulation_8():
    t_sat = load_2d_list("t_sat.txt")
    str_sat_t = sat_t_to_string(t_sat)

    #set simulation parameter
    sim_time = 3600
    num_st = 298
    seed = 10
    initial_node_capacity = 40

    #generate commodities (node_capacity)
    cut = get_cut(t_sat, sim_time, initial_node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=initial_node_capacity,cut=cut, seed=seed)

    dict_node_capacity = {}
    dict_node_capacity, list_sat = set_uniform_node_capacity(t_sat, initial_node_capacity)

    file_MAC_path = f"path_MAC.txt"
    file_MAC_HO = f"HO_MAC.txt"
    MVT.MVT(str_sat_t, dict_node_capacity, commodities, file_MAC_path, file_MAC_HO)

#test gen_problem.construct_graph_v2() + pack_method()
def simulation_9():
    t_sat = load_2d_list("t_sat.txt")

    sim_time = 3600
    node_capacity = 50
    #num_st = 298
    seed = 10

    reduce_ratio = 30
    G = gen_problem.construct_graph_v2(t_sat, node_capacity, reduce_ratio)

    cut = get_cut(t_sat, sim_time, node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=node_capacity,cut=cut, seed=seed)

    total_demand = sum(d for d in commodities.values())
    print(f"total_demand = {total_demand}")

    pack.pack_method(G,commodities,"path_pack.txt","HO_pack.txt")

    return


def simulation_10():
    t_sat = load_2d_list("t_sat.txt")
    str_sat_t = sat_t_to_string(t_sat)

    sim_time = 3600
    num_st = 298
    seed = 10
    initial_node_capacity = 40
    
    cut = get_cut(t_sat, sim_time, initial_node_capacity)
    commodities = generate_bin_packing_instance(node_capacity=initial_node_capacity,cut=cut, seed=seed)

    dict_node_capacity = {}
    dict_node_capacity, list_sat = set_uniform_node_capacity(t_sat, initial_node_capacity)
    coverage_period = gen_problem.get_coverage_period(t_sat,list_sat,sim_time)
    edge_cap = initial_node_capacity*num_st+1
    G = gen_problem.construct_graph(num_st,sim_time,coverage_period,dict_node_capacity,list_sat,edge_cap)

    file_MSH_path = f"test_MSH_{initial_node_capacity}.txt"
    file_MSH_HO = f"test_MSH_{initial_node_capacity}.txt"
    MSH.new_MSH(G,commodities,file_MSH_path,file_MSH_HO)

    return


def plot(directory: str, name: str):
    pack_path = f"{directory}/HO_pack_*.txt"
    mvt_path = f"{directory}/HO_MVT_*.txt"
    mac_path = f"{directory}/HO_MAC_*.txt"
    msh_path = f"{directory}/HO_MSH_*.txt"
    # Get all files matching the patterns
    pack_files = sorted(glob(pack_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mvt_files = sorted(glob(mvt_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mac_files = sorted(glob(mac_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    msh_files = sorted(glob(msh_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print(pack_files)
    print(mvt_files)

    list_x = []
    list_flow_data = []
    list_mvt_data = []
    list_mac_data = []
    list_msh_data = []

    for pack_file, mvt_file, mac_file, msh_file in zip(pack_files, mvt_files, mac_files, msh_files):
        # Extract integer from filename (taking the number before .txt)
        number = int(pack_file.split('_')[-1].replace('.txt', ''))
        list_x.append(number)
        print(pack_file)
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
    plot_name = f"HO v.s. {name}.pdf"
    plt.savefig(plot_name, bbox_inches='tight')
    print(f"save the plot to {plot_name}")

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


if __name__ == "__main__":
    #simulation_1()
    #simulation_2(3)
    simulation_3() #num_HO v.s. node capacity
    #simulation_4()
    simulation_5() #num_HO v.s. UE request
    #simulation_6()
    #simulation_7()
    plot("result/node_capacity/HO", "node capacity")
    plot("result/UE_request/HO", "UE request")
    #simulation_8()
    #simulation_9()
    #simulation_10()
    


