import problem_generator_v2 as gen_problem


import math
import json
import copy


from typing import Dict, List, Set, Tuple, Optional
from heapq import heappush, heappop
from collections import defaultdict, deque

#"double" vertex splitting
#G_residual[u][v]=(capacity,length,edge_type,reversed_ue)
def create_residual_graph(graph: gen_problem.Graph, flow_dict: Dict[str, Dict[str, Dict[str, int]]]) -> gen_problem.Graph:
    #G_residual[u][v]=(capacity,length,type,reversed_ue)
    G_residual = {}

    #initialize residual_capacity for creating VERTEX_FOR edges
    #residual_capacity[n] = reisdual capacity of vertex n
    residual_capacity = {}
    for n in graph.nodes:
        if n=='S' or n=='T':
            continue
        residual_capacity[n]=graph.node_capacity[n]

    #initialize vertex_ue for creating VERTEX_REV edges
    #vertex_ue[u] = [list of UE whose flow pass through vertex u]
    vertex_ue = {}
    for n in graph.nodes:
        if n=='S' or n=='T':
            continue
        vertex_ue[n]=[]

    #read flow_dict
    #count residual_capacity
    #get vertex_ue for vertex splitting
    for ue, dict_ue in flow_dict.items():
        for u, dict_u in dict_ue.items():
            for v, flow_amount in dict_u.items():
                u_name=u.split('_')[0]
                v_name=v.split('_')[0]
                if u_name==v_name and u_name!='S' and u_name!='T':
                    residual_capacity[u_name]-=flow_amount
                    if ue not in vertex_ue[u_name]:
                        vertex_ue[u_name].append(ue)

    #add VERTEX_FOR, VERTEX_REV edge
    for n in graph.nodes:
        if n=='S' or n=='T':
            G_residual[n]={}
            continue
        #VERTEX_FOR
        n_in=f"{n}_in"
        n_out=f"{n}_out"
        G_residual[n_in]={}
        G_residual[n_in][n_out]=(residual_capacity[n],0,gen_problem.EdgeType.VERTEX_FOR,None)
        G_residual[n_out]={}
        #VERTEX_REV
        for ue in vertex_ue[n]:
            if flow_dict[ue][n_in][n_out]!=0:
                n_in_ue=f"{n}_in_{ue}"
                n_out_ue=f"{n}_out_{ue}"
                G_residual[n_in_ue]={}
                G_residual[n_out_ue]={}
                length = 0
                G_residual[n_out_ue][n_in_ue]=(flow_dict[ue][n_in][n_out],length,gen_problem.EdgeType.VERTEX_REV,ue)

    #add BACKWARD edge
    for ue, dict_ue in flow_dict.items():
        for u, dict_u in dict_ue.items():
            for v, flow_amount in dict_u.items():
                u_name=u.split('_')[0]
                v_name=v.split('_')[0]
                if u_name==v_name:
                    continue
                u_node=f"{u_name}_out" if u_name!='S' else 'S'
                u_node_ue=f"{u_name}_out_{ue}" if u_name!='S' else 'S'
                v_node_ue=f"{v_name}_in_{ue}" if v!='T' else 'T'
                v_node=f"{v_name}_in" if v!='T' else 'T'
                length = -1
                G_residual[v_node_ue][u_node]=(flow_amount,length,gen_problem.EdgeType.BACKWARD,ue)
                G_residual[v_node_ue][u_node_ue]=(flow_amount,length,gen_problem.EdgeType.BACKWARD,ue)
                G_residual[v_node][u_node]=(flow_amount,length,gen_problem.EdgeType.BACKWARD,ue)


    #add FORWARD edge
    for e in graph.edges:
        u=f"{e[0]}_out" if e[0]!='S' else 'S'
        v=f"{e[1]}_in" if e[1]!='T' else 'T'
        length = 1
        G_residual[u][v]=(math.inf,length,gen_problem.EdgeType.FORWARD,None)
        if e[0]=='S':
            continue
        for ue in vertex_ue[e[0]]:
            u_out_ue=f"{e[0]}_out_{ue}" if e[0]!='S' else 'S'
            G_residual[u_out_ue][v]=(math.inf,length,gen_problem.EdgeType.FORWARD,None)

    return G_residual

#based on bellman-ford algorithm
#path format: (node, edge type, reversed_ue, demand)
def find_shortest_path(G_residual: Dict[str,Dict[str,Tuple[int,int]]], bool_demand: bool, demand: int) -> Tuple[List[str],int]:
    distance={node: math.inf for node in G_residual.keys()}
    parent={node: None for node in G_residual.keys()}

    V = len(G_residual.keys())
    distance['S']=0
    for i in range(V-1):    
        for u, dict_u in G_residual.items():
            if distance[u]==math.inf:
                    continue
            for v, edge in dict_u.items():
                #edge = (flow_amount, length)
                if bool_demand and edge[0]<demand:
                    continue
                
                #edge[0]!=0 when bool_demand=False
                if distance[u]+edge[1]<distance[v] and edge[0]!=0:
                    distance[v]=distance[u]+edge[1]
                    parent[v]=u
    
    if distance['T']==math.inf:
        #print("find_shortest_path(): No path exists")
        return None, None

    #path
    path_vertex=[]
    current='T'
    while current!='S':
        path_vertex.append(current)
        current=parent[current]
    path_vertex.append('S')
    path_vertex.reverse()

    #get capacity of path
    capacity = math.inf
    v_previous = None
    for v in path_vertex:
        if v==path_vertex[0]:
            v_previous=v
            continue
        edge_capacity = G_residual[v_previous][v][0]
        capacity = edge_capacity if edge_capacity<capacity else capacity
        v_previous=v

    path = []
    v_previous = None
    for v in path_vertex:
        if v==path_vertex[0]:
            path.append((v,None,None,capacity))
            v_previous=v
            continue
        #G_residual[u][v]=(flow_amount,lengt,edge_type,reversed_ue)
        edgetype=G_residual[v_previous][v][2]
        reversed_ue=G_residual[v_previous][v][3]
        path_item=(v,edgetype,reversed_ue,capacity)
        path.append(path_item)
        v_previous=v

    return path, capacity


#flow.update_commodity_flows()
#path format: (node, edge type, reversed_ue, demand)
def update_flow(flow_dict, path, ue_target, demand):
    #print("update_flow():")
    #print(path)
    flow_dict[ue_target] = {}

    ue_current = ue_target
    vertex_previous = None
    flow_amount=demand
    for e in path:
        if e==path[0]:
            #print(f"update_flow(): e={e}")
            vertex_previous = e[0]
            continue
        
        #add e into commodity_flows[ue_current][vertex_previous]
        if e[1]==gen_problem.EdgeType.FORWARD or e[1]==gen_problem.EdgeType.VERTEX_FOR:
            if vertex_previous not in flow_dict[ue_current].keys():
                flow_dict[ue_current][vertex_previous] = {}
            flow_dict[ue_current][vertex_previous][e[0]]=flow_amount
            vertex_previous = e[0]

        #remove the victim flow segment
        #add the victim flow segment into commodity_flows[ue_current]
        #ue_current = reversed_ue
        else:
            queue = deque()
            visited = set()
            vertex_current = e[0]
            reversed_ue = e[2]
            flow_path = e[3]
            victim_flow_segment = {}
            victim_flow_segment[vertex_previous]={}
            victim_flow_segment[vertex_previous][vertex_current] = flow_path
            
            queue.append(vertex_previous)
            visited.add(vertex_current)

            
            #get victim_flow_segment by BFS
            #remove the victim_flow_segment from flow_dict[reversed_ue]
            while queue:
                vertex_pop = queue.popleft()
                if vertex_pop=='T':
                    continue
                victim_flow_segment[vertex_pop] = {}

                #save the victim flow segment
                #remove the victim flow segment from commodity_flows
                for neighbor, flow_amount in flow_dict[reversed_ue][vertex_pop].items():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        victim_flow_segment[vertex_pop][neighbor]=flow_amount
                        flow_dict[reversed_ue][vertex_pop][neighbor]-=flow_path

                #delete empty items of flow_dict[reversed_ue][vertex_pop] here
                flow_dict[reversed_ue][vertex_pop] = {key:value for key,value in flow_dict[reversed_ue][vertex_pop].items() if value!=0}
                if not flow_dict[reversed_ue][vertex_pop]:
                    del flow_dict[reversed_ue][vertex_pop]

            #add victim_flow_segment to flow_dict[ue_current]
            for u, dict_u in victim_flow_segment.items():
                for v, flow_amount in dict_u.items():
                    if u not in flow_dict[ue_current]:
                        flow_dict[ue_current][u]={}
                    if v not in flow_dict[ue_current][u]:
                        flow_dict[ue_current][u][v]=flow_amount
                    else:
                        flow_dict[ue_current][u][v]+=flow_amount

            ue_current = reversed_ue
            #remove the reversed edge from flow_dict[ue_current]
            #the reversed edge: [vertex_previous][vertex_current]
            flow_dict[ue_current][vertex_current][vertex_previous]-=flow_amount
            if flow_dict[ue_current][vertex_current][vertex_previous]==0:
                del flow_dict[ue_current][vertex_current][vertex_previous]

            vertex_previous = vertex_current

    return flow_dict


def subset_sum(candidate_neighbor, demand_reversed):
    """
    Find a subset of candidates whose values sum to the target value.
    If no exact match exists, find a minimal set that exceeds the target.
    
    Args:
        candidate_neighbor (dict): Dictionary mapping candidate names to their values
        demand_reversed (float/int): The target value to match
        
    Returns:
        list: List of candidate names that form the solution
    """
    # Extract the candidates and their values
    candidates = list(candidate_neighbor.keys())
    values = [candidate_neighbor[c] for c in candidates]
    
    # Sort candidates by value in descending order for greedy approach later
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    sorted_candidates = [candidates[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Try to find exact subset sum using dynamic programming
    n = len(candidates)
    
    # If the target is 0, return empty set
    if demand_reversed == 0:
        return []
    
    # If all values are 0 and target is not 0, no solution exists
    if all(v == 0 for v in values) and demand_reversed != 0:
        # Return minimal subset that exceeds (which is impossible with all zeros)
        return []
    
    # Initialize DP table
    # dp[i][j] = the subset of first i candidates that sums closest to j without exceeding it
    dp = [[-float('inf')] * (int(demand_reversed) + 1) for _ in range(n + 1)]
    dp[0][0] = 0  # Base case: empty set sums to 0
    
    # Keep track of choices
    choices = {}
    
    # Fill the DP table
    for i in range(1, n + 1):
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
    if dp[n][int(demand_reversed)] == demand_reversed:
        # Reconstruct the solution
        #solution = []
        solution = {}
        i, j = n, int(demand_reversed)
        while i > 0 and j > 0:
            if choices.get((i, j), False):
                #solution.append(candidates[i-1])
                solution[candidates[i-1]]=candidate_neighbor[candidates[i-1]]
                j -= values[i-1]
            i -= 1
        return solution
    
    # If no exact solution, find minimal set that exceeds target
    # Using a greedy approach (taking largest values first)
    #greedy_solution = []
    greedy_solution={}
    current_sum = 0
    
    for i, candidate in enumerate(sorted_candidates):
        if current_sum >= demand_reversed:
            break
        #greedy_solution.append(candidate)
        flow_amount = min(candidate_neighbor[candidate],demand_reversed-current_sum)
        greedy_solution[candidate]=flow_amount
        current_sum += sorted_values[i]
    
    return greedy_solution


def choose_neighbor(list_neighbor,demand_reversed):
    selected_neighbor={}
    candidate_neighbor={}
    for neighbor, flow_amount in list_neighbor.items():
        if flow_amount==demand_reversed:
            selected_neighbor[neighbor]=flow_amount
            return selected_neighbor
        candidate_neighbor[neighbor]=flow_amount
    
    selected_neighbor= subset_sum(candidate_neighbor,demand_reversed)

    return selected_neighbor


def get_victim_flow_by_BFS(flow_reversed, vertex_current, demand_reversed):
    victim_segment = {}
    victim_segment[vertex_current]={}

    queue=deque()
    queue.append((vertex_current,demand_reversed))
    while queue:
        visited=set()

        tuple_vertex_pop=queue.popleft()
        vertex_pop=tuple_vertex_pop[0]
        flow_pop=tuple_vertex_pop[1]

        if vertex_pop=='T':
            continue
        
        if vertex_pop not in victim_segment:
            victim_segment[vertex_pop]={}

        list_neighbor = {}
        flag=False
        selected_neighbor={}
        for neighbor, flow_amount in flow_reversed[vertex_pop].items():
            #print(f"{neighbor}, {flow_amount}")
            if neighbor not in visited:
                visited.add(neighbor)
                list_neighbor[neighbor]=flow_amount
            if flow_amount==demand_reversed:
                flag=True
                selected_neighbor[neighbor]=flow_amount
                break
                
        if not flag:
            selected_neighbor = choose_neighbor(list_neighbor,flow_pop)

        for neighbor, flow_amount in selected_neighbor.items():
            victim_segment[vertex_pop][neighbor]=flow_amount
            tuple_neighbor=(neighbor,flow_amount)
            queue.append(tuple_neighbor)

    return victim_segment


def add_victim_segment(flow,ue_current,victim_segment):
    #if ue_current not in flow:
    #    flow[ue_current]={}
    for u, dict_u in victim_segment.items():
        for v, flow_amount in dict_u.items():
            if u not in flow[ue_current]:
                flow[ue_current][u]={}
                flow[ue_current][u][v]=flow_amount
            elif v not in flow[ue_current]:
                flow[ue_current][u][v]=flow_amount
            else:
                flow[ue_current][u][v]+=flow_amount

    return flow


def add_reverse_edge(path,index_path,victim_segment):
    num_reversed_edge=0

    while True:
        if path[index_path+num_reversed_edge][1]==gen_problem.EdgeType.BACKWARD or path[index_path+num_reversed_edge][1]==gen_problem.EdgeType.VERTEX_REV:
            u=path[index_path+num_reversed_edge][0]
            v=path[index_path+num_reversed_edge-1][0]
            if u not in victim_segment:
                victim_segment[u]={}
                if v not in victim_segment[u]:
                    victim_segment[u][v]=1
                else:
                    victim_segment[u][v]+=1
            num_reversed_edge+=1
        else:
            break

    return victim_segment, num_reversed_edge


def remove_victim_segment(flow,ue_reversed,victim_segment):
    for u, dict_u in victim_segment.items():
        for v, flow_amount in dict_u.items():
            flow[ue_reversed][u][v]-=flow_amount
            if flow[ue_reversed][u][v]==0:
                del flow[ue_reversed][u][v]
                if not flow[ue_reversed][u]:
                    del flow[ue_reversed][u]

    return flow


#path format: (node, edge type, reversed_ue, demand)
def new_update_flow(flow, path, target_ue):
    flow[target_ue]={}
    
    index_path=0
    vertex_previous = None
    ue_current = target_ue
    while True:
        e = path[index_path]

        if e==path[0]:
            index_path+=1
            vertex_previous=e[0]
            continue

        if e[1]==gen_problem.EdgeType.FORWARD or e[1]==gen_problem.EdgeType.VERTEX_FOR:
            #add forward edge
            if vertex_previous not in flow[ue_current].keys():
                flow[ue_current][vertex_previous] = {}
            flow[ue_current][vertex_previous][e[0]]=e[3]
            #break condition
            if e[0]=='T':
                break
            #update loop var
            vertex_previous = e[0]
            index_path+=1
        
        else:
            ue_reversed = e[2]
            demand_reversed = e[3]
            victim_segment=get_victim_flow_by_BFS(flow[ue_reversed],vertex_previous,demand_reversed)
            flow=add_victim_segment(flow,ue_current,victim_segment)
            victim_segment, num_reverse_edge = add_reverse_edge(path,index_path,victim_segment)
            flow=remove_victim_segment(flow,ue_reversed,victim_segment)
            ue_current = ue_reversed
            index_path+=num_reverse_edge
            vertex_previous=path[index_path-1][0]


#output graph into .json file
def save_graph(graph, filename):
    """
    Save graph to a file, handling infinity values.
    
    Args:
        graph (dict): Graph represented as nested dictionary with tuple values
        filename (str): Name of file to save to
    """
    # Create a copy of the graph to modify for saving
    save_data = {}
    for node, edges in graph.items():
        save_data[node] = {}
        for dest, (capacity, length, edge_type, reversed_ue) in edges.items():
            # Convert inf to string 'inf' for serialization
            saved_capacity = 'inf' if capacity == math.inf else capacity
            save_data[node][dest] = (saved_capacity, length)
    
    # Write to file
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=4)
    
    print("save_graph(): done")


def write_flow_dict_to_file(flow_dict, filename):
    """
    Write a nested dictionary representing network flows to a file.
    
    The nested dictionary has the structure flow_dict[u][v] = f,
    where f is the flow amount from node u to node v.
    
    Args:
        flow_dict (dict): A nested dictionary where flow_dict[u][v] = f
                          represents flow amount f from node u to node v
        filename (str): Path to the output file
    
    Returns:
        None
    """
    with open(filename, 'w') as file:
        # Write a header
        file.write("# Flow Network\n")
        file.write("# Format: source_node destination_node flow_amount\n\n")
        
        # Iterate through the nested dictionary
        for source_node in sorted(flow_dict.keys()):
            for dest_node in sorted(flow_dict[source_node].keys()):
                flow_amount = flow_dict[source_node][dest_node]
                
                # Only write non-zero flows
                if flow_amount != 0:
                    file.write(f"{source_node} {dest_node} {flow_amount}\n")


def UE_packing(G: gen_problem.Graph, commodities: Dict[str,int]):
    #get UE_pack by greedy
    remaining_demand = sum(commodities.values())
    flow_dict = {}
    pack_capacity=[]
    pack_index = 0
    while remaining_demand>0:
        #print(f"UE_packing(): find the {pack_index+1}-th pack")
        G_residual=create_residual_graph(G, flow_dict)
        #print("create_residual_graph() finished")
        path, demand = find_shortest_path(G_residual,False,None)
        #print("find_shortest_path() finished")
        if path==None:
            print(f"remaining_demand={remaining_demand}")
            save_graph(G_residual,"G_residual.json")
            print("ue_packing(): save graph to G_residual.json. Load it with graph_utils.load_graph()")
            write_flow_dict_to_file(flow_dict, "flow_dict.txt")
            break
        #flow_dict = update_flow(flow_dict, path, pack_index, demand)
        new_update_flow(flow_dict,path,pack_index)
        remaining_demand -= demand
        pack_capacity.append(demand)
        pack_index += 1

    #print("pack_capacity")
    #print(pack_capacity)

    UE_pack = []

    #allocate UE to adjust UE_pack
    remaining_capacity_of_pack = {index: value for index, value in enumerate(pack_capacity)}
    sorted_commodities = {k: v for k, v in sorted(commodities.items(), key=lambda item: item[1], reverse=True)}

    UE_pack = {index: (capacity, []) for index, capacity in enumerate(pack_capacity)}
    remaining_UE = []

    #first fit
    for com, demand_com in sorted_commodities.items():
        allocated=False
        for index, capacity in remaining_capacity_of_pack.items():
            if(capacity>=demand_com):
                allocated=True
                UE_pack[index][1].append(com)
                remaining_capacity_of_pack[index] -= demand_com
                break
        if not allocated:
            remaining_UE.append(com)


    return UE_pack, remaining_UE


#pick the shortest path with the capacity as tiebreaker (prefer big capacity)
def pick_path(path_d,remaining_demand):
    path = None
    demand_path = None
    length = float('inf')
    for demand, p in enumerate(path_d):
        if(demand==0):
            continue
        if(p==None or p==[]):
            continue
        if(demand > remaining_demand):
            break
        if(len(p)<length):
            path = p
            demand_path = demand
            length = len(p)
        elif(len(p)==length and demand>demand_path):
            path = p
            demand_path = demand
            length = len(p)

    return path, demand_path


def create_residual_graph_for_split(G,flow_UE,frac_flow_for_ue):
    #G_residual[u][v]=(capacity,length,type,reversed_ue)
    G_residual = {}

    #initialize residual_capacity for creating VERTEX_FOR edges
    #residual_capacity[n] = reisdual capacity of vertex n
    residual_capacity = {}
    for n in G.nodes:
        if n=='S' or n=='T':
            continue
        residual_capacity[n]=G.node_capacity[n]

    #initialize vertex_ue for creating VERTEX_REV edges
    #vertex_ue[u] = [list of UE whose flow pass through vertex u]
    vertex_ue = {}
    for n in G.nodes:
        if n=='S' or n=='T':
            continue
        vertex_ue[n]=[]

    #read flow_UE
    #count residual_capacity
    #get vertex_ue for vertex splitting
    for ue, dict_ue in flow_UE.items():
        for u, dict_u in dict_ue.items():
            for v, flow_amount in dict_u.items():
                u_name=u.split('_')[0]
                v_name=v.split('_')[0]
                if u_name==v_name and u_name!='S' and u_name!='T':
                    residual_capacity[u_name]-=flow_amount
                    if ue not in vertex_ue[u_name]:
                        vertex_ue[u_name].append(ue)

    #add VERTEX_FOR, VERTEX_REV edge
    for n in G.nodes:
        if n=='S' or n=='T':
            G_residual[n]={}
            continue
        #VERTEX_FOR
        n_in=f"{n}_in"
        n_out=f"{n}_out"
        length = 0
        G_residual[n_in]={}
        G_residual[n_in][n_out]=(residual_capacity[n],length,gen_problem.EdgeType.VERTEX_FOR,None)
        G_residual[n_out]={}
        #VERTEX_REV
        for ue in vertex_ue[n]:
            if flow_UE[ue][n_in][n_out]!=0:
                n_in_ue=f"{n}_in_{ue}"
                n_out_ue=f"{n}_out_{ue}"
                G_residual[n_in_ue]={}
                G_residual[n_out_ue]={}
                length = 0
                G_residual[n_out_ue][n_in_ue]=(flow_UE[ue][n_in][n_out],length,gen_problem.EdgeType.VERTEX_REV,ue)

    #add BACKWARD edge
    for ue, dict_ue in flow_UE.items():
        for u, dict_u in dict_ue.items():
            for v, flow_amount in dict_u.items():
                u_name=u.split('_')[0]
                v_name=v.split('_')[0]
                if u_name==v_name:
                    continue
                u_node=f"{u_name}_out" if u_name!='S' else 'S'
                u_node_ue=f"{u_name}_out_{ue}" if u_name!='S' else 'S'
                v_node_ue=f"{v_name}_in_{ue}" if v!='T' else 'T'
                v_node=f"{v_name}_in" if v!='T' else 'T'
                length = -1
                G_residual[v_node_ue][u_node]=(flow_amount,length,gen_problem.EdgeType.BACKWARD,ue)
                G_residual[v_node_ue][u_node_ue]=(flow_amount,length,gen_problem.EdgeType.BACKWARD,ue)
                G_residual[v_node][u_node]=(flow_amount,length,gen_problem.EdgeType.BACKWARD,ue)

    def find_edge(frac_flow_for_ue, u, v):
        for index_path, p in frac_flow_for_ue.items():
            for u, dict_u in p.items():
                for v in dict_u.keys():
                    if(e[0]==u and e[1]==v):
                        return True
        return False

    #add FORWARD edge
    for e in G.edges:
        u=f"{e[0]}_out" if e[0]!='S' else 'S'
        v=f"{e[1]}_in" if e[1]!='T' else 'T'
        is_flow_edge = find_edge(frac_flow_for_ue,u,v)
        length = 1 if not is_flow_edge else 0
        G_residual[u][v]=(math.inf,length,gen_problem.EdgeType.FORWARD,None)
        if e[0]=='S':
            continue
        for ue in vertex_ue[e[0]]:
            u_out_ue=f"{e[0]}_out_{ue}" if e[0]!='S' else 'S'
            G_residual[u_out_ue][v]=(math.inf,length,gen_problem.EdgeType.FORWARD,None)


    return G_residual


def process_and_output_flow(flow, file_flow_path, file_flow_HO):
    """
    Process flow data and output statistics and paths to specified files.
    
    Parameters:
    - flow: Dictionary where keys are user IDs and values are adjacency lists (dictionaries)
    - file_flow_path: Path to the file for storing path statistics
    - file_flow_HO: Path to the file for storing handover count
    """
    # Initialize counters and storage
    handovers_per_user = {}
    total_handovers = 0
    
    # Process each user's flow - just count handovers directly from the graph structure
    for user_id, adjacency_list in flow.items():
        handover_pairs = set()
        
        # Look for handover patterns - any out->in transitions
        for vertex, targets in adjacency_list.items():
            if "_out" in vertex:
                source_id = vertex.split("_")[0]  # Get the ID of the source
                
                for target in targets:
                    if "_in" in target:
                        target_id = target.split("_")[0]  # Get the ID of the target
                        
                        # If source and target IDs are different, it's a handover
                        if source_id != target_id:
                            handover_pairs.add((source_id, target_id))
        
        # Store the handover count for this user
        handover_count = len(handover_pairs)
        handovers_per_user[user_id] = handover_count
        total_handovers += handover_count
        
        # For debug: print the handover pairs
        #print(f"User {user_id} handovers: {handover_pairs}")
    
    # Now let's reconstruct paths for output
    processed_paths = {}
    for user_id, adjacency_list in flow.items():
        # This is still simplified but tries to show a representative path
        path = ['S']
        processed = set()
        
        # Get all unique nodes (excluding S and T, in/out pairs combined)
        all_nodes = set()
        for v in adjacency_list:
            if v == 'S' or v == 'T':
                continue
            
            if "_in" in v or "_out" in v:
                base_name = v.split("_")[0]
                all_nodes.add(base_name)
        
        # Add nodes to the path in order if possible
        for node in sorted(all_nodes):
            path.append(node)
        
        path.append('T')
        processed_paths[user_id] = path
    
    # Write path information to file_flow_path
    with open(file_flow_path, 'w') as f:
        # Write overall statistics
        f.write("Overall Statistics:\n")
        f.write("-----------------\n")
        for user_id, count in handovers_per_user.items():
            f.write(f"User {user_id}: {count} handovers\n")
        f.write("\n")
        
        # Write paths
        f.write("User Paths:\n")
        f.write("----------\n")
        for user_id, path in processed_paths.items():
            f.write(f"User {user_id}: {' -> '.join(path)}\n")
    
    # Write total handovers to file_flow_HO
    with open(file_flow_HO, 'w') as f:
        f.write(f"{total_handovers}")
    
    print(f"MSH.process_and_output_flow(): output the result to {file_flow_path} and {file_flow_HO}")

    return total_handovers


def pack_method(G: gen_problem.Graph, commodities: Dict[int, int], file_flow_path: str, file_flow_HO: str):
    #input: G, commodities

    UE_pack, remaining_UE = UE_packing(G,commodities)

    #order the pack
    sorted_items = sorted(
        UE_pack.items(),
        key=lambda item: (-item[1][0], -len(item[1][1]))  # Primary: -capacity, Secondary: -list length
    )
    sorted_UE_pack = dict(sorted_items)
    flow_dict={} #flow for each pack

    #find paths based on pack_capacity
    #format: sorted_UE_pack[index_pack] = (demand_pack, [list of UEs in the pack])
    for index_pack, pack in sorted_UE_pack.items():
        #print(f"find path for pack {pack}")
        G_residual = create_residual_graph(G,flow_dict)
        path, capacity = find_shortest_path(G_residual,True,pack[0])
        #flow_dict = update_flow(flow_dict,path,index_pack,pack[0])
        new_update_flow(flow_dict,path,index_pack)
        flag=False
        for p, flow_pack in flow_dict.items():
            if 'T' in flow_pack.keys():
                flag=True
        if flag:
            print(f"path has reverse edges when index_pack = {index_pack}")


    #allocate the paths based on UE_pack
    flow_UE={} #flow for each UE
    for index_pack, flow_pack in flow_dict.items():
        for u, dict_u in flow_pack.items():
            for v, flow_amount in dict_u.items():
                for ue in UE_pack[index_pack][1]:
                    if ue not in flow_UE:
                        flow_UE[ue]={}
                    if u not in flow_UE[ue]:
                        flow_UE[ue][u]={}
                    flow_UE[ue][u][v]=commodities[ue]

    #find flow for remaining_UE
    if remaining_UE!=[]:
        for ue in remaining_UE:
            print(f"find paths for UE {ue}")
            G_residual = create_residual_graph(G,flow_UE)
            path, capacity = find_shortest_path(G_residual,True,commodities[ue])
            
            if path:
                #flow_UE = update_flow(flow_UE,path,ue,commodities[ue])
                new_update_flow(flow_UE,path,ue)
            elif capacity==-1:
                return
            else:
                target_demand = commodities[ue]
                served_demand = 0

                flow_tmp = copy.deepcopy(flow_UE)
                virtual_ue_index = len(commodities.keys())
                frac_flow_for_ue = {} #for create_residual_graph_for_split()
                num_path = 0
                while(served_demand < target_demand):
                    #print(f"{num_path+1}-th iteration")
                    path_d = [[] for _ in range(target_demand-served_demand+1)]
                    has_path = False

                    for d in range(1,target_demand-served_demand+1):
                        path_d[d], capacity = find_shortest_path(G_residual,True,d)
                        if capacity==-1:
                            return
                        if path_d[d]!=None:
                            has_path=True
                    
                    if not has_path:
                        print(f"pack_method: no fractional flow for UE {ue}")
                        save_graph(G_residual,"G_residual.json")
                        #for ue, dict_ue in flow_UE.items():
                        #    print(f"flow for ue {ue}")
                        #    print(flow_UE[ue])
                        return
                    
                    [path, demand_path] = pick_path(path_d, target_demand-served_demand)
                    served_demand += demand_path

                    num_path+=1

                    #should use workaround to update fractional flow...
                    #flow_tmp = update_flow(flow_tmp,path,virtual_ue_index,demand_path)
                    new_update_flow(flow_tmp,path,virtual_ue_index)
                    #frac_flow_for_ue = update_flow(frac_flow_for_ue,path,num_path,demand_path)
                    new_update_flow(frac_flow_for_ue,path,num_path)
                    G_residual = create_residual_graph_for_split(G,flow_tmp,frac_flow_for_ue)     

                    

                #print("pack_method(): frac_flow_for_ue")
                #print(frac_flow_for_ue)

                #combine flow_UE with frac_flow_for_ue
                print(frac_flow_for_ue)
                
                flow_UE[ue]={}
                for path in frac_flow_for_ue.values():
                    for u, dict_u in path.items():
                        if u not in flow_UE[ue]:
                            flow_UE[ue][u]={}
                        
                        for v, amount in dict_u.items():
                            if v not in flow_UE[ue][u]:
                                flow_UE[ue][u][v]=0
                            flow_UE[ue][u][v]+=amount                   

    sorted_flow_UE = dict(sorted(flow_UE.items()))
    process_and_output_flow(sorted_flow_UE,file_flow_path,file_flow_HO)


#test create_residual_graph(), find_shortest_path() for update_flow()
def test_1():
    #create G
    vertex_capacity=40
    G=gen_problem.Graph()
    G.add_node('S')
    G.set_node_capacity('S',vertex_capacity*12)
    G.add_node('T')
    G.set_node_capacity('T',vertex_capacity*12)
    for sat in range(1,13):
        G.add_node(str(sat))
        G.set_node_capacity(str(sat),vertex_capacity)
    G.add_edge('S','1',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('S','5',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('S','9',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('2','3',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('3','4',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('4','T',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('5','6',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('6','7',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('8','T',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('9','10',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('10','11',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('12','T',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('1','2', None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('2','7', None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('7','8', None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('5','6', None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('6','11', None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('11','12', None,gen_problem.EdgeType.FORWARD,math.inf,1)

    flow_dict={}
    flow_dict[1]={}
    flow_dict[1]['S']={}
    flow_dict[1]['S']['1_in']=40
    flow_dict[1]['1_in']={}
    flow_dict[1]['1_in']['1_out']=40
    flow_dict[1]['1_out']={}
    flow_dict[1]['1_out']['2_in']=40
    flow_dict[1]['2_in']={}
    flow_dict[1]['2_in']['2_out']=40
    flow_dict[1]['2_out']={}
    flow_dict[1]['2_out']['7_in']=40
    flow_dict[1]['7_in']={}
    flow_dict[1]['7_in']['7_out']=40
    flow_dict[1]['7_out']={}
    flow_dict[1]['7_out']['8_in']=40
    flow_dict[1]['8_in']={}
    flow_dict[1]['8_in']['8_out']=40
    flow_dict[1]['8_out']={}
    flow_dict[1]['8_out']['T']=40
    flow_dict[2]={}
    flow_dict[2]['S']={}
    flow_dict[2]['S']['5_in']=40
    flow_dict[2]['5_in']={}
    flow_dict[2]['5_in']['5_out']=40
    flow_dict[2]['5_out']={}
    flow_dict[2]['5_out']['6_in']=40
    flow_dict[2]['6_in']={}
    flow_dict[2]['6_in']['6_out']=40
    flow_dict[2]['6_out']={}
    flow_dict[2]['6_out']['11_in']=40
    flow_dict[2]['11_in']={}
    flow_dict[2]['11_in']['11_out']=40
    flow_dict[2]['11_out']={}
    flow_dict[2]['11_out']['12_in']=40
    flow_dict[2]['12_in']={}
    flow_dict[2]['12_in']['12_out']=40
    flow_dict[2]['12_out']={}
    flow_dict[2]['12_out']['T']=40

    G_residual = create_residual_graph(G,flow_dict)
    path = find_shortest_path(G_residual,False,None)

    save_graph(G_residual, "G_residual.json")
    print(path)

    return


#test choose_neighbor()
def test_2():
    list_neighbor = {}
    list_neighbor['0']=1
    list_neighbor['1']=2
    list_neighbor['2']=2
    list_neighbor['3']=2
    demand_reversed = 3
    selected_neighbor = choose_neighbor(list_neighbor,demand_reversed)
    print(selected_neighbor)


#test get_victim_flow_by_BFS()
def test_3():
    flow_reversed={}
    flow_reversed['S']={'1':12}
    flow_reversed['1']={'2':2,'4':2,'6':4,'8':4}
    flow_reversed['2']={'3':2}
    flow_reversed['3']={'T':2}
    flow_reversed['4']={'5':2}
    flow_reversed['5']={'T':2}
    flow_reversed['6']={'7':4}
    flow_reversed['7']={'T':4}
    flow_reversed['8']={'9':4}
    flow_reversed['9']={'T':4}

    vertex_current='1'
    demand_reversed=5
    victim_segment=get_victim_flow_by_BFS(flow_reversed, vertex_current, demand_reversed)
    print(victim_segment)


def test():
    flow={}
    flow['u1']={}
    flow['u1']['S']={'1':6}
    flow['u1']['1']={'2':6}
    flow['u1']['2']={'3':3,'5':2,'7':1}
    flow['u1']['3']={'4':3}
    flow['u1']['4']={'T':3}
    flow['u1']['5']={'6':2}
    flow['u1']['6']={'T':2}
    flow['u1']['7']={'8':1}
    flow['u1']['8']={'T':1}
    '''
    flow['u2']={}
    flow['u2']['S']={'1':2}
    flow['u2']['1']={'2':2}
    flow['u2']['2']={'5':2}
    flow['u2']['5']={'6':2}
    flow['u2']['6']={'T':2}
    flow['u3']={}
    flow['u3']['S']={'1':1}
    flow['u3']['1']={'2':1}
    flow['u3']['2']={'7':1}
    flow['u3']['7']={'8':1}
    flow['u3']['8']={'T':1}
    '''

    #path format: (node, edge type, reversed_ue, demand)
    path=[]
    path.append(('S',None,None,1))
    path.append(('9',gen_problem.EdgeType.FORWARD,None,4))
    path.append(('2',gen_problem.EdgeType.FORWARD,None,4))
    path.append(('1',gen_problem.EdgeType.BACKWARD,'u1',4))
    path.append(('10',gen_problem.EdgeType.FORWARD,None,4))
    path.append(('11',gen_problem.EdgeType.FORWARD,None,4))
    path.append(('12',gen_problem.EdgeType.FORWARD,None,4))
    path.append(('13',gen_problem.EdgeType.FORWARD,None,4))
    path.append(('T',gen_problem.EdgeType.FORWARD,None,4))
    target_ue='u2'
    new_update_flow(flow, path, target_ue)
    for ID_ue, flow_ue in flow.items():
        print(f"flow for UE {ID_ue}")
        print(flow_ue)
    return


if __name__ == "__main__":
    test()
















