import problem_generator_v2 as gen_problem
import pack_method as pack

import math

from typing import Dict, List, Set, Tuple, Optional
from heapq import heappush, heappop
from collections import defaultdict, deque


def create_residual_graph(G:gen_problem.Graph, flow):
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
        G_residual[n_in][n_out]=(residual_capacity[n],length,gen_problem.EdgeType.VERTEX_FOR,None)
        G_residual[n_out]={}
        #VERTEX_REV
        for ue in vertex_ue[n]:
            if flow[ue][n_in][n_out]!=0:
                n_in_ue=f"{n}_in_{ue}"
                n_out_ue=f"{n}_out_{ue}"
                G_residual[n_in_ue]={}
                G_residual[n_out_ue]={}
                length = 0
                G_residual[n_out_ue][n_in_ue]=(flow[ue][n_in][n_out],length,gen_problem.EdgeType.VERTEX_REV,ue)

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
                G_residual[v_node_ue][u_node]=(flow_amount,length,gen_problem.EdgeType.BACKWARD,ue)
                G_residual[v_node_ue][u_node_ue]=(flow_amount,length,gen_problem.EdgeType.BACKWARD,ue)
                G_residual[v_node][u_node]=(flow_amount,length,gen_problem.EdgeType.BACKWARD,ue)

    #add FORWARD edge
    for e in G.edges:
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


#G_residual[u][v]=(capacity,length,edgetype,reversed_ue)
def old_check_old_path(G_residual, old_path):
    if old_path==None:
        return False
    
    flow_path=old_path[0][3]

    index_old_path=1
    while True:
        vertex_current=old_path[index_old_path][0]
        vertex_previous=old_path[index_old_path-1][0]
        if G_residual[vertex_previous][vertex_current][0]<flow_path:
            return False

        if vertex_current=='T':
            break
        
        index_old_path+=1

    return True


#G_residual[u][v]=(capacity,length,edgetype,reversed_ue)
#old_path cannot be the next path -> False
#old_path can be the next path -> True
def check_old_path(G:gen_problem.Graph,unit_flow,old_path):
    if old_path==None:
        return False

    #collect vertices in the old_path
    vertex_in_old_path=[]
    for e in old_path:
        vertex_in_old_path.append(e[0])

    residual_capacity = {v: G.node_capacity[v] for v in vertex_in_old_path}
    #check residual capacity for vertex_in_old_path

    return True


#based on bellman-ford algorithm
#path format: (node, edge type, reversed_ue, demand)
def find_shortest_path(G_residual: Dict[str,Dict[str,Tuple[int,int]]], bool_demand: bool, demand: int) -> Tuple[List[str],int]:
    distance={node: math.inf for node in G_residual.keys()}
    parent={node: None for node in G_residual.keys()}

    #get distance, parent
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

    return path


def get_victim_flow_by_BFS(flow_reversed, vertex_current):
    victim_segment = {}
    victim_segment[vertex_current]={}

    queue=deque()
    queue.append(vertex_current)
    visited=set()

    while queue:
        vertex_pop=queue.popleft()
        if vertex_pop=='T' or vertex_pop not in flow_reversed:
            continue
        if vertex_pop not in victim_segment:
            victim_segment[vertex_pop]={}

        for neighbor, flow_amount in flow_reversed[vertex_pop].items():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        victim_segment[vertex_pop][neighbor]=flow_amount

    return victim_segment, flow_amount


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
def update_flow(flow, path, target_ue):
    flow[target_ue] = {}

    ue_current = target_ue
    vertex_previous = None
    flow_amount=1
    index_path = 0
    #break condition: e[0]=='T' in forward edge block
    while True:
        e=path[index_path]

        #e is the first edge
        if e==path[0]:
            #print(f"update_flow(): e={e}")
            vertex_previous = e[0]
            index_path+=1
            continue
        
        #e is a forward edge
        #add e into commodity_flows[ue_current][vertex_previous]
        if path[index_path][1]==gen_problem.EdgeType.FORWARD or path[index_path][1]==gen_problem.EdgeType.VERTEX_FOR:
            #add forward edge
            if vertex_previous not in flow[ue_current].keys():
                flow[ue_current][vertex_previous] = {}
            flow[ue_current][vertex_previous][e[0]]=flow_amount
            #break condition
            if e[0]=='T':
                break
            #update loop var
            vertex_previous = e[0]
            index_path+=1
            

        #e is a reverse edge
        #assume the amount of flow through all paths are 1
        else:
            ue_reversed = e[2]
            victim_segment=get_victim_flow_by_BFS(flow[ue_reversed],vertex_previous)
            flow=add_victim_segment(flow,ue_current,victim_segment)
            victim_segment, num_reverse_edge = add_reverse_edge(path,index_path,victim_segment)
            flow=remove_victim_segment(flow,ue_reversed,victim_segment)
            ue_current = ue_reversed
            index_path+=num_reverse_edge
            vertex_previous=path[index_path-1][0]
            #print(flow)
            #return

        #if index_path==len(path)-1:
        #    break


def add_path_to_flow(flow, unit_flow):
    for u, dict_u in unit_flow.items():
        if u not in flow:
            flow[u]={}
        for v, flow_amount in dict_u.items():
            if v not in flow[u]:
                flow[u][v]=flow_amount
            else:
                flow[u][v]+=flow_amount


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


def new_find_shortest_path(G_residual: Dict[str,Dict[str,Tuple[int,int]]]) -> Tuple[List[Tuple],int]:
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
                #edge[0]!=0
                if distance[u]+edge[1]<distance[v] and edge[0]!=0:
                    distance[v]=distance[u]+edge[1]
                    parent[v]=u
    
    if distance['T']==math.inf:
        #print("find_shortest_path(): No path exists")
        return None, None

    #get path_vertex
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

    #capacity=capacity if capacity<remaining_demand else remaining_demand

    #output path
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


#modify by pack_method()
def MSH(G: gen_problem.Graph, commodities: Dict[int, int], file_flow_path: str, file_flow_HO: str):
    total_demand = sum(d for d in commodities.values())
    unit_flow = {}

    print(f"MSH(): total demand = {total_demand}")

    old_path=None
    for i in range(total_demand):
        print(f"MSH(): i = {i}")
        #check old path
        #check_result=check_old_path(G,unit_flow,old_path)
        G_residual=create_residual_graph(G, unit_flow)
        '''
        #check old path
        check_result=old_check_old_path(G_residual,old_path)
        if not check_result:
            path=find_shortest_path(G_residual,unit_flow,1)
            old_path=path
        '''
        path=find_shortest_path(G_residual,unit_flow,1)
        ID_path=i
        update_flow(unit_flow,path,ID_path)
    
    flow = {}
    count_path = 0
    for ID_ue, demand in commodities.items():
        flow[ID_ue]={}
        for i in range(demand):
            add_path_to_flow(flow[ID_ue],unit_flow[count_path])
            count_path+=1

    for ID_ue, dict_ue in flow.items():
        print(f"path for ue {ID_ue}")
        print(dict_ue)
    
    process_and_output_flow(flow, file_flow_path, file_flow_HO)


def new_remove_victim_segment(flow,size_flow,ue_reversed,flow_reversed,flow_path,victim_segment):
    #print("new_remove_victim_segment()")
    #print(f"flow_path={flow_path}")
    #print(f"flow_reversed={flow_reversed}")
    if flow_path!=flow_reversed:
        #print("new_remove_victim_segment(): flow_victim!=flow_reversed")
        #print(f"flow_victim={flow_path}, flow_reversed={flow_reversed}")
        for u, dict_u in flow[ue_reversed].items():
            for v in dict_u.keys():
                flow[ue_reversed][u][v]-=flow_reversed

        new_ue=str(size_flow)
        flow[new_ue]={}
        for u, dict_u in flow[ue_reversed].items():
            if u not in flow[new_ue]:
                flow[new_ue][u]={}
            if u not in victim_segment.keys():
                for v in dict_u.keys():
                    flow[new_ue][u][v]=flow_reversed
        return flow

    for u, dict_u in victim_segment.items():
        for v in dict_u.keys():
            del flow[ue_reversed][u][v]
            if not flow[ue_reversed][u]:
                del flow[ue_reversed][u]

    return flow


#path format: (node, edge type, reversed_ue, demand)
def new_update_flow(flow,path,size_flow):
    flow_path=path[0][3]
    new_size_flow=size_flow

    ue_current=str(size_flow)
    if ue_current not in flow:
        flow[ue_current]={}
    index_path=0
    vertex_previous=None
    while True:
        e=path[index_path]

        if e==path[0]:
            index_path+=1
            vertex_previous=e[0]
            continue
        
        #e is a forward edge
        #add e into commodity_flows[ue_current][vertex_previous]
        if path[index_path][1]==gen_problem.EdgeType.FORWARD or path[index_path][1]==gen_problem.EdgeType.VERTEX_FOR:
            #add forward edge
            if vertex_previous not in flow[ue_current].keys():
                flow[ue_current][vertex_previous] = {}
            flow[ue_current][vertex_previous][e[0]]=flow_path
            #break condition
            if e[0]=='T':
                break
            #update loop var
            vertex_previous = e[0]
            index_path+=1

        else:
            ue_reversed = e[2]
            victim_segment, flow_reversed=get_victim_flow_by_BFS(flow[ue_reversed],vertex_previous)
            flow=add_victim_segment(flow,ue_current,victim_segment)
            #if size_flow==6:
            #    for ue, f in flow.items():
            #        print(f"flow for ue {ue}")
            #        print(f)
            #    print("----------------------------------------------")
            victim_segment, num_reverse_edge = add_reverse_edge(path,index_path,victim_segment)
            flow=new_remove_victim_segment(flow,size_flow,ue_reversed,flow_reversed,flow_path,victim_segment)
            #if size_flow==6:
            #    for ue, f in flow.items():
            #        print(f"flow for ue {ue}")
            #        print(f)
            #    print("----------------------------------------------")
            ue_current = ue_reversed
            index_path+=num_reverse_edge
            vertex_previous=path[index_path-1][0]

            if flow_reversed != flow_path:
                new_size_flow+=1

    return new_size_flow


def assign_flow_to_UE(flow_UE,flow,amount):
    for u, dict_u in flow.items():
        if u not in flow_UE:
            flow_UE[u]={}
        for v in dict_u.keys():
            if v not in flow_UE[u]:
                flow_UE[u][v]=amount
            else:
                flow_UE[u][v]+=amount


def new_MSH(G: gen_problem.Graph, commodities: Dict[int, int], file_flow_path: str, file_flow_HO: str):
    remaining_demand=sum(d for d in commodities.values())
    flow={}

    size_flow=0
    while True:
        G_residual=create_residual_graph(G,flow)

        path, demand=new_find_shortest_path(G_residual)
        #if path:
        #    print(f"path found with demand={demand}, when size_flow={size_flow} and remaining_demand={remaining_demand}")
        #else:
        #    print(f"path not found when size_flow={size_flow}")
        size_flow=new_update_flow(flow,path,size_flow)
        #update_flow(unit_flow,path,str(size_flow))

        remaining_demand-=demand
        size_flow+=1

        #for ue, f in flow.items():
        #    print(f"flow for ue {ue}")
        #    print(f)
        #print("----------------------------------------------------------")

        if remaining_demand<=0:
            break

    #for ue, f in flow.items():
    #    print(f"flow for ue {ue}")
    #    print(f)


    #allocate unit_flow to UE
    flow_UE={}
    
    remaining_capacity={}
    for ID_flow, f in flow.items():
        cap=sum(v for v in f['S'].values())
        remaining_capacity[ID_flow]=cap

    #list_frac = []

    index_flow=0
    for ID_com, demand_com in commodities.items():
        flow_UE[str(ID_com)]={}
        served_demand=0
        while True:
            amount=min(demand_com-served_demand,remaining_capacity[str(index_flow)])
            assign_flow_to_UE(flow_UE[str(ID_com)],flow[str(index_flow)],amount)
            remaining_capacity[str(index_flow)]-=amount
            served_demand+=amount
            if remaining_capacity[str(index_flow)]==0:
                index_flow+=1
            
            if served_demand==demand_com:
                break
            #else:
                #list_frac.append(ID_com)
                #print(f"fractional flow for UE {ID_com}")

    #for com in list_frac:
    #    print(f"flow for UE {str(com)}")
    #    print(flow_UE[str(com)])

    process_and_output_flow(flow_UE, file_flow_path, file_flow_HO)


#test update_flow()
def test_1():
    #flow_reversed = {'S':{'0':1}, '0':{'1':1}, '1':{'2':1}, '2':{'T':1}}
    flow = {}
    flow['0'] = {}
    flow['0'] = {'S':{'1':1}, '1':{'2':1}, '2':{'3':1}, '3':{'T':1}}
    flow['1'] = {}
    flow['1'] = {'S':{'4':1}, '4':{'5':1}, '5':{'6':1}, '6':{'T':1}}
    #flow['2'] = {}
    #flow['2'] = {'S':{'7':1}, '7':{'8':1}, '8':{'6':1}}
    #path format: (node, edge type, reversed_ue, demand)
    path=[]
    path.append(('S',None,None,1))
    path.append(('7',gen_problem.EdgeType.FORWARD,None,1))
    path.append(('8',gen_problem.EdgeType.FORWARD,None,1))
    path.append(('6',gen_problem.EdgeType.FORWARD,None,1))
    path.append(('5',gen_problem.EdgeType.BACKWARD,'1',1))
    path.append(('4',gen_problem.EdgeType.BACKWARD,'1',1))
    path.append(('3',gen_problem.EdgeType.FORWARD,None,1))
    path.append(('2',gen_problem.EdgeType.BACKWARD,'0',1))
    path.append(('1',gen_problem.EdgeType.BACKWARD,'0',1))
    path.append(('9',gen_problem.EdgeType.FORWARD,None,1))
    path.append(('10',gen_problem.EdgeType.FORWARD,None,1))
    path.append(('T',gen_problem.EdgeType.FORWARD,None,1))
    print(path)
    return
    '''
    vertex_previous = '6'
    vertex_current = '5'
    ue_reversed='1'
    victim_segment=get_victim_flow_by_BFS(flow[ue_reversed],vertex_previous)
    ue_current='2'
    flow=add_victim_segment(flow,ue_current,victim_segment)
    index_path=4 #index of vertex_current
    victim_segment, num_reverse_edge = add_reverse_edge(path,index_path,victim_segment)
    flow=remove_victim_segment(flow,ue_reversed,victim_segment)
    '''
    
    update_flow(flow,path,'2')


#test MSH
def test_2():
    G=gen_problem.Graph()
    G.add_node('S')
    G.set_node_capacity('S',math.inf)
    G.add_node('1')
    G.set_node_capacity('1',2)
    G.add_node('2')
    G.set_node_capacity('2',2)
    G.add_node('3')
    G.set_node_capacity('3',3)
    G.add_node('4')
    G.set_node_capacity('4',1)
    G.add_node('5')
    G.set_node_capacity('5',1)
    G.add_node('6')
    G.set_node_capacity('6',2)
    G.add_node('7')
    G.set_node_capacity('7',2)
    G.add_node('8')
    G.set_node_capacity('8',2)
    G.add_node('T')
    G.set_node_capacity('T',math.inf)
    #add_edge(u, v, ue, type, capacity, length)
    G.add_edge('S','1',None,gen_problem.EdgeType.FORWARD,math.inf,0)
    G.add_edge('S','4',None,gen_problem.EdgeType.FORWARD,math.inf,0)
    G.add_edge('1','2',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('1','6',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('2','3',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('4','5',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('5','2',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('6','7',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('7','8',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('3','T',None,gen_problem.EdgeType.FORWARD,math.inf,0)
    G.add_edge('8','T',None,gen_problem.EdgeType.FORWARD,math.inf,0)
    #commodities={'0':1,'1':1,'2':1}
    commodities={'0':2,'1':1}
    file_flow_path=None
    file_flow_HO=None
    MSH(G,commodities,file_flow_path,file_flow_HO)
    return
    

#test new_MSH()
def test_3():

    G=gen_problem.Graph()
    G.add_node('S')
    G.set_node_capacity('S',math.inf)
    G.add_node('1')
    G.set_node_capacity('1',2)
    G.add_node('2')
    G.set_node_capacity('2',2)
    G.add_node('3')
    G.set_node_capacity('3',3)
    G.add_node('4')
    G.set_node_capacity('4',1)
    G.add_node('5')
    G.set_node_capacity('5',1)
    G.add_node('6')
    G.set_node_capacity('6',2)
    G.add_node('7')
    G.set_node_capacity('7',2)
    G.add_node('8')
    G.set_node_capacity('8',2)
    G.add_node('T')
    G.set_node_capacity('T',math.inf)
    #add_edge(u, v, ue, type, capacity, length)
    G.add_edge('S','1',None,gen_problem.EdgeType.FORWARD,math.inf,0)
    G.add_edge('S','4',None,gen_problem.EdgeType.FORWARD,math.inf,0)
    G.add_edge('1','2',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('1','6',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('2','3',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('4','5',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('5','2',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('6','7',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('7','8',None,gen_problem.EdgeType.FORWARD,math.inf,1)
    G.add_edge('3','T',None,gen_problem.EdgeType.FORWARD,math.inf,0)
    G.add_edge('8','T',None,gen_problem.EdgeType.FORWARD,math.inf,0)
    #commodities={'0':1,'1':1,'2':1}
    commodities={'0':2,'1':1}

    file_flow_path=None
    file_flow_HO=None
    new_MSH(G,commodities,file_flow_path,file_flow_HO)

    return


#test process_and_output_flow()
def test_4():
    flow={}
    flow['user1']={}
    flow['user1']['S']={'1':1,'2':1}
    flow['user1']['1']={'3':1}
    flow['user1']['2']={'3':1}
    flow['user1']['3']={'T':2}

    process_and_output_flow(flow,"test_path.txt","test_HO.txt")

    return


if __name__ == "__main__":
    #test_1()
    #test_2()
    #test_3()
    test_4()





