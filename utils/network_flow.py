import math
from typing import Dict, List, Tuple, Set, Optional
from collections import deque, defaultdict
from ..core.graph import Graph
from ..core.edge_types import EdgeType


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


def create_residual_graph(G, flow):
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
    for ue, dict_ue in flow.items():
        for u, dict_u in dict_ue.items():
            for v, flow_amount in dict_u.items():
                u_name=u.split('_')[0]
                v_name=v.split('_')[0]
                if u_name==v_name and u_name!='S' and u_name!='T':
                    residual_capacity[u_name]-=flow_amount
                    if ue not in vertex_ue[u_name]:
                        vertex_ue[u_name].append(ue)

    #print("create_residual_graph(): residual_capacity")
    #for vertex, cap in residual_capacity.items():
    #    print(f"residual_capacity[{vertex}]={cap}")

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
        G_residual[n_in][n_out]=(residual_capacity[n],length,EdgeType.VERTEX_FOR,None)
        G_residual[n_out]={}
        #VERTEX_REV
        for ue in vertex_ue[n]:
            if flow[ue][n_in][n_out]!=0:
                n_in_ue=f"{n}_in_{ue}"
                n_out_ue=f"{n}_out_{ue}"
                G_residual[n_in_ue]={}
                G_residual[n_out_ue]={}
                length = 0
                G_residual[n_out_ue][n_in_ue]=(flow[ue][n_in][n_out],length,EdgeType.VERTEX_REV,ue)

    #add BACKWARD edge
    for ue, dict_ue in flow.items():
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
                G_residual[v_node_ue][u_node]=(flow_amount,length,EdgeType.BACKWARD,ue)
                G_residual[v_node_ue][u_node_ue]=(flow_amount,length,EdgeType.BACKWARD,ue)
                G_residual[v_node][u_node]=(flow_amount,length,EdgeType.BACKWARD,ue)

    #add FORWARD edge
    for e in G.edges:
        u=f"{e[0]}_out" if e[0]!='S' else 'S'
        v=f"{e[1]}_in" if e[1]!='T' else 'T'
        length = 1
        G_residual[u][v]=(math.inf,length,EdgeType.FORWARD,None)
        if e[0]=='S':
            continue
        for ue in vertex_ue[e[0]]:
            u_out_ue=f"{e[0]}_out_{ue}" if e[0]!='S' else 'S'
            G_residual[u_out_ue][v]=(math.inf,length,EdgeType.FORWARD,None)

    return G_residual


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
        G_residual[n_in][n_out]=(residual_capacity[n],length,EdgeType.VERTEX_FOR,None)
        G_residual[n_out]={}
        #VERTEX_REV
        for ue in vertex_ue[n]:
            if flow_UE[ue][n_in][n_out]!=0:
                n_in_ue=f"{n}_in_{ue}"
                n_out_ue=f"{n}_out_{ue}"
                G_residual[n_in_ue]={}
                G_residual[n_out_ue]={}
                length = 0
                G_residual[n_out_ue][n_in_ue]=(flow_UE[ue][n_in][n_out],length,EdgeType.VERTEX_REV,ue)

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
                G_residual[v_node_ue][u_node]=(flow_amount,length,EdgeType.BACKWARD,ue)
                G_residual[v_node_ue][u_node_ue]=(flow_amount,length,EdgeType.BACKWARD,ue)
                G_residual[v_node][u_node]=(flow_amount,length,EdgeType.BACKWARD,ue)

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
        G_residual[u][v]=(math.inf,length,EdgeType.FORWARD,None)
        if e[0]=='S':
            continue
        for ue in vertex_ue[e[0]]:
            u_out_ue=f"{e[0]}_out_{ue}" if e[0]!='S' else 'S'
            G_residual[u_out_ue][v]=(math.inf,length,EdgeType.FORWARD,None)


    return G_residual


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


def add_victim_segment(flow,ue_current,victim_segment):
    #if ue_current not in flow:
    #    flow[ue_current]={}
    #print(f"add_victim_segment(): victim_segment={victim_segment}")
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
        if path[index_path+num_reversed_edge][1]==EdgeType.BACKWARD or path[index_path+num_reversed_edge][1]==EdgeType.VERTEX_REV:
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

        if e[1]==EdgeType.FORWARD or e[1]==EdgeType.VERTEX_FOR:
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

