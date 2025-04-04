import math
from typing import Dict, List, Tuple, Set, Optional
from collections import deque, defaultdict
from ..core.graph import Graph
from ..core.edge_types import EdgeType

def get_victim_flow_by_BFS(flow_reversed, vertex_current, demand_reversed=None):
    """
    Get the victim flow segment using breadth-first search.
    
    Args:
        flow_reversed: The flow to analyze
        vertex_current: Current vertex
        demand_reversed: Optional demand to consider
        
    Returns:
        Victim flow segment and flow amount
    """
    victim_segment = {}
    victim_segment[vertex_current] = {}

    queue = deque()
    queue.append(vertex_current)
    visited = set()

    while queue:
        vertex_pop = queue.popleft()
        if vertex_pop == 'T' or vertex_pop not in flow_reversed:
            continue
            
        if vertex_pop not in victim_segment:
            victim_segment[vertex_pop] = {}

        for neighbor, flow_amount in flow_reversed[vertex_pop].items():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                victim_segment[vertex_pop][neighbor] = flow_amount

    # Determine flow amount
    if vertex_current in flow_reversed and flow_reversed[vertex_current]:
        sample_flow = next(iter(flow_reversed[vertex_current].values()))
        return victim_segment, sample_flow
    return victim_segment, 0


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


