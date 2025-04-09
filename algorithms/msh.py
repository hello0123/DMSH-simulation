# satellite_scheduling/algorithms/msh.py

import math
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional

from ..core.graph import Graph
from ..core.edge_types import EdgeType
from ..utils.network_flow import (
    create_residual_graph, 
    find_shortest_path,
    add_victim_segment,
    add_reverse_edge,
)

class MSHAlgorithm:
    """
    Minimum Satellite Handovers (MSH) Algorithm
    
    This algorithm attempts to minimize the number of satellite handovers
    when routing commodities through a network of satellites.
    """
    
    def __init__(self):
        self.name = "Minimum Satellite Handovers (MSH)"
    
    #staticmethod
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

    @staticmethod
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

    @staticmethod
    def update_flow(flow, path, size_flow):
        """
        Update flow with new path.
        
        Args:
            flow: Current flow
            path: New path
            size_flow: Current size of flow
            
        Returns:
            Updated size_flow
        """
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
            if path[index_path][1]==EdgeType.FORWARD or path[index_path][1]==EdgeType.VERTEX_FOR:
                if vertex_previous not in flow[ue_current].keys():
                    flow[ue_current][vertex_previous] = {}
                flow[ue_current][vertex_previous][e[0]]=flow_path

                if e[0]=='T':
                    break

                vertex_previous = e[0]
                index_path+=1

            else:
                ue_reversed = e[2]
                victim_segment, flow_reversed=MSHAlgorithm.get_victim_flow_by_BFS(flow[ue_reversed],vertex_previous)
                flow=add_victim_segment(flow,ue_current,victim_segment)
                victim_segment, num_reverse_edge = add_reverse_edge(path,index_path,victim_segment)
                flow=MSHAlgorithm.new_remove_victim_segment(flow,size_flow,ue_reversed,flow_reversed,flow_path,victim_segment)
                ue_current = ue_reversed
                index_path+=num_reverse_edge
                vertex_previous=path[index_path-1][0]

                if flow_reversed != flow_path:
                    new_size_flow+=1

        return new_size_flow

    @staticmethod
    def assign_flow_to_UE(flow_UE, flow, amount):
        """
        Assign flow to UE with specified amount.
        
        Args:
            flow_UE: Flow for UE
            flow: Source flow
            amount: Amount to assign
            
        Returns:
            None (modifies flow_UE in-place)
        """
        for u, dict_u in flow.items():
            if u not in flow_UE:
                flow_UE[u] = {}
            for v, flow_amount in dict_u.items():
                if v not in flow_UE[u]:
                    flow_UE[u][v] = amount
                else:
                    flow_UE[u][v] += amount

    def process_and_output_flow(self, flow, file_flow_path, file_flow_HO):
        """
        Process flow data and output to files.
        
        Args:
            flow: Flow to process
            file_flow_path: Path output file
            file_flow_HO: Handover output file
            
        Returns:
            Total handovers count
        """
        # Initialize counters and storage
        handovers_per_user = {}
        total_handovers = 0
        
        # Process each user's flow
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
        
        # Reconstruct paths for output
        processed_paths = {}
        for user_id, adjacency_list in flow.items():
            # This is a simplified path representation
            path = ['S']
            
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
        try:
            with open(file_flow_path, 'w') as f:
                # Write overall statistics
                f.write("Handover Summary\n")
                f.write("================\n\n")
                f.write(f"Total number of handovers: {total_handovers}\n\n")
                f.write("Handovers per commodity:\n")
                for i, (user_id, count) in enumerate(sorted(handovers_per_user.items())):
                    f.write(f"Commodity {i}: {count} handovers\n")
                
                f.write("\nSatellite Connection Sequences\n")
                f.write("============================\n\n")
                
                # Write paths
                for i, (user_id, path) in enumerate(sorted(processed_paths.items())):
                    f.write(f"Commodity {i}:\n")
                    f.write("Time  Satellites (Flow Amount)\n")
                    f.write("----------------------------\n")
                    
                    # Simplified path representation - just show the final path
                    satellites = [node for node in path if node not in ('S', 'T')]
                    if satellites:
                        f.write(f"t=0    {', '.join([f'{sat}(1)' for sat in satellites])}\n")
                    
                    f.write("\nSequence with flows:\n")
                    if satellites:
                        f.write(f"{' -> '.join([f'{sat}(1)' for sat in satellites])}\n\n")
                    else:
                        f.write("\n\n")
        except Exception as e:
            print(f"Error writing to {file_flow_path}: {str(e)}")
        
        # Write total handovers to file_flow_HO
        try:
            with open(file_flow_HO, 'w') as f:
                f.write(f"{total_handovers}")
        except Exception as e:
            print(f"Error writing to {file_flow_HO}: {str(e)}")
        
        print(f"MSH: Processed flow written to {file_flow_path} and {file_flow_HO}")
        return total_handovers
        
    def write_handover_output(self, handover_count, file_path):
        """Write handover count to file."""
        with open(file_path, 'w') as f:
            f.write(f"{handover_count}")
        print(f"Wrote total handover count ({handover_count}) to {file_path}")

    def run(self, G, commodities, file_flow_path, file_flow_HO):
        """Run the MSH algorithm."""
        remaining_demand = sum(d for d in commodities.values())
        flow = {}

        size_flow = 0
        while remaining_demand > 0:
            G_residual = create_residual_graph(G, flow)
            path, demand = find_shortest_path(G_residual, False, None)    
            size_flow = MSHAlgorithm.update_flow(flow, path, size_flow)

            remaining_demand -= demand
            size_flow += 1

            if remaining_demand <= 0:
                break

        # Allocate unit_flow to UE
        flow_UE = {}
        
        remaining_capacity = {}
        for ID_flow, f in flow.items():
            if 'S' in f:
                cap = sum(flow_val for flow_val in f['S'].values())
            else:
                # If no direct connection from S, use a default value
                cap = 0
            remaining_capacity[ID_flow] = cap

        index_flow = 0
        for ID_com, demand_com in sorted(commodities.items()):
            flow_UE[str(ID_com)] = {}
            served_demand = 0
            
            while served_demand < demand_com:
                # Make sure we don't go beyond available flows
                while (str(index_flow) not in remaining_capacity or 
                       remaining_capacity[str(index_flow)] <= 0) and index_flow < size_flow:
                    index_flow += 1
                    
                if index_flow >= size_flow:
                    print(f"MSH: Not enough capacity to serve commodity {ID_com}")
                    break
                    
                amount = min(demand_com - served_demand, remaining_capacity[str(index_flow)])
                
                if str(index_flow) in flow:
                    self.assign_flow_to_UE(flow_UE[str(ID_com)], flow[str(index_flow)], amount)
                    remaining_capacity[str(index_flow)] -= amount
                    served_demand += amount
                    
                    if remaining_capacity[str(index_flow)] <= 0:
                        index_flow += 1
                else:
                    # Skip non-existent flows
                    index_flow += 1
                
                if served_demand >= demand_com:
                    break

        return self.process_and_output_flow(flow_UE, file_flow_path, file_flow_HO)


def MSH(G, commodities, file_flow_path, file_flow_HO):
    """
    Legacy wrapper function for MSH algorithm.
    
    Args:
        G: Graph
        commodities: Dictionary of commodities
        file_flow_path: Path output file
        file_flow_HO: Handover output file
        
    Returns:
        Total handovers count
    """
    # This function exists for backward compatibility
    return new_MSH(G, commodities, file_flow_path, file_flow_HO)


def new_MSH(G, commodities, file_flow_path, file_flow_HO):
    """
    Run the MSH algorithm.
    
    Args:
        G: Graph
        commodities: Dictionary of commodities
        file_flow_path: Path output file
        file_flow_HO: Handover output file
        
    Returns:
        Total handovers count
    """
    algorithm = MSHAlgorithm()
    return algorithm.run(G, commodities, file_flow_path, file_flow_HO)