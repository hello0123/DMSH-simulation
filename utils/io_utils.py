import pickle
import json
from typing import List, Dict, Any

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

def write_flow_dict_to_file(flow_dict: Dict[str, Dict[str, Any]], filename: str):
    """
    Write a nested dictionary representing network flows to a file.
    
    Args:
        flow_dict: A nested dictionary with flow data
        filename: Path to the output file
    """
    with open(filename, 'w') as file:
        file.write("# Flow Network\n")
        file.write("# Format: source_node destination_node flow_amount\n\n")
        
        for source_node in sorted(flow_dict.keys()):
            for dest_node in sorted(flow_dict[source_node].keys()):
                flow_amount = flow_dict[source_node][dest_node]
                
                if flow_amount != 0:
                    file.write(f"{source_node} {dest_node} {flow_amount}\n")
                    
    print(f"Flow dictionary saved to {filename}")

def sat_t_to_string(sat_t: List[List[int]]) -> List[List[str]]:
    """
    Convert integer satellite IDs to strings.
    
    Args:
        sat_t: List of lists containing integer satellite IDs
        
    Returns:
        List of lists with string satellite IDs
    """
    str_sat_t = [[] for _ in range(len(sat_t))]
    for t in range(len(sat_t)):
        for index in range(len(sat_t[t])):
            str_sat_t[t].append(str(sat_t[t][index]))
    return str_sat_t

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
    
    #print(f"MSH.process_and_output_flow(): output the result to {file_flow_path} and {file_flow_HO}")

    return total_handovers

