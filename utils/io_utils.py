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

