from typing import List, Dict, Set, Tuple
from collections import defaultdict

def calculate_visible_time(sat_t: List[List[str]], current_t: int, satellite: str) -> int:
    """
    Calculate how long a satellite remains visible from current time slot.
    
    Args:
        sat_t: List of lists containing visible satellites at each time slot
        current_t: Current time slot
        satellite: Satellite to check visibility for
        
    Returns:
        Number of consecutive time slots the satellite remains visible
    """
    visible_time = 0
    for t in range(current_t, len(sat_t)):
        if satellite in sat_t[t]:
            visible_time += 1
        else:
            break
    return visible_time


def count_handovers(assignments: Dict[int, Dict[str, str]]) -> Dict[str, int]:
    """
    Count the number of handovers for each commodity during the simulation period.
    A handover occurs when a commodity switches from one satellite to another.
    
    Args:
        assignments: Dictionary mapping time slots to assignments {t: {commodity_id: satellite_name}}
        
    Returns:
        Dictionary mapping commodity IDs to their total number of handovers
    """
    # Initialize handover counters and last known connections
    handover_counts = defaultdict(int)
    last_connections = {}  # {commodity_id: (satellite, last_time_slot)}
    
    # Get sorted time slots and find all unique commodities
    time_slots = sorted(assignments.keys())
    all_commodities = set()
    for assignments_in_slot in assignments.values():
        all_commodities.update(assignments_in_slot.keys())
    
    # Process each time slot
    for t in time_slots:
        current_assignments = assignments[t]
        
        # Check each commodity
        for commodity in all_commodities:
            current_satellite = current_assignments.get(commodity, None)
            
            if commodity in last_connections:
                last_satellite, last_time = last_connections[commodity]
                
                # Case 1: Was connected before, now disconnected
                if current_satellite is None and last_satellite is not None:
                    pass  # Not counting disconnection as handover
                
                # Case 2: Was connected before, now connected to different satellite
                elif current_satellite is not None and current_satellite != last_satellite:
                    handover_counts[commodity] += 1
                    last_connections[commodity] = (current_satellite, t)
                
                # Case 3: Still connected to same satellite or still disconnected
                else:
                    if current_satellite is not None:
                        last_connections[commodity] = (current_satellite, t)
            
            else:  # First time seeing this commodity
                if current_satellite is not None:
                    last_connections[commodity] = (current_satellite, t)
    
    # Convert defaultdict to regular dict for return
    return dict(handover_counts)


def write_mvt_results_to_file(assignments: Dict[int, Dict[str, Dict[str, int]]], file_path: str, file_HO: str):
    """
    Write MVT results to a file, supporting multiple satellite connections per commodity.
    """
    def get_satellite_sequence(assignments, commodity_id):
        """Get sequence of satellite connections with start times and flow amounts."""
        sequence = []
        current_sats = {}
        time_slots = sorted(assignments.keys())
        
        for t in time_slots:
            if commodity_id in assignments[t]:
                new_sats = assignments[t][commodity_id]
                
                # Check for changes in satellites or flow amounts
                if new_sats != current_sats:
                    sequence.append((new_sats, t))
                    current_sats = new_sats.copy()
                    
        return sequence

    status = True
    total_handovers = 0
    try:
        # Calculate handovers
        handover_counts = defaultdict(int)
        all_commodities = set()
        
        for assignments_in_slot in assignments.values():
            all_commodities.update(assignments_in_slot.keys())
            
        for commodity in all_commodities:
            sequence = get_satellite_sequence(assignments, commodity)
            # Count changes in satellite combinations as handovers
            handover_counts[commodity] = len(sequence) - 1 if sequence else 0
            
        total_handovers = sum(handover_counts.values())
        
        # Write results
        with open(file_path, 'w') as f:
            f.write("Handover Summary\n")
            f.write("================\n\n")
            f.write(f"Total number of handovers: {total_handovers}\n\n")
            f.write("Handovers per commodity:\n")
            for commodity in sorted(handover_counts.keys()):
                f.write(f"Commodity {commodity}: {handover_counts[commodity]} handovers\n")
            
            f.write("\nSatellite Connection Sequences\n")
            f.write("============================\n")
            
            for commodity in sorted(all_commodities):
                f.write(f"\nCommodity {commodity}:\n")
                sequence = get_satellite_sequence(assignments, commodity)
                
                if not sequence:
                    f.write("No connections\n")
                    continue
                
                f.write("Time  Satellites (Flow Amount)\n")
                f.write("----------------------------\n")
                for sats_dict, time in sequence:
                    sat_flows = [f"{sat}({flow})" for sat, flow in sats_dict.items()]
                    f.write(f"t={time:<4} {', '.join(sat_flows)}\n")
                
                # Write compact sequence
                f.write("\nSequence with flows:\n")
                sequences = []
                for sats_dict, _ in sequence:
                    sat_flows = [f"{sat}({flow})" for sat, flow in sats_dict.items()]
                    sequences.append(" + ".join(sat_flows))
                f.write(" -> ".join(sequences))
                f.write("\n")
    except Exception as e:
        status = False
        print(f"Error writing to file {file_path}: {str(e)}")

    try:
        with open(file_HO, 'w') as f:
            f.write(f"{total_handovers}")
    except Exception as e:
        status = False
        print(f"Error writing to file {file_HO}: {str(e)}")

    if status:
        print(f"mvt.write_mvt_results_to_file(): write result to {file_path} and {file_HO}")


class MVT:
    """
    Modified Visible Time (MVT) algorithm for satellite scheduling.
    This algorithm selects satellites with the longest visible time.
    """
    
    def __init__(self):
        """Initialize the MVT algorithm."""
        pass
    
    def __call__(self, sat_t: List[List[str]], node_capacity: Dict[str, int], 
            commodities: Dict[str, int], file_MVT_path: str, file_MVT_HO: str) -> Dict[int, Dict[str, str]]:
        """
        Run the MVT algorithm with the provided parameters.
        This method allows the class to be called as a function.
        """
        return self.run(sat_t, node_capacity, commodities, file_MVT_path, file_MVT_HO)
        
    def run(self, sat_t: List[List[str]], node_capacity: Dict[str, int], 
            commodities: Dict[str, int], file_MVT_path: str, file_MVT_HO: str) -> Dict[int, Dict[str, str]]:
        """
        Run the MVT algorithm with the provided parameters.
        
        Args:
            sat_t: List of lists containing visible satellites at each time slot
            node_capacity: Dictionary mapping satellite name to its capacity
            commodities: Dictionary mapping user name to its demand
            file_MVT_path: Output file path for connection paths
            file_MVT_HO: Output file path for handover count
            
        Returns:
            Dictionary mapping time slots to assignments
        """
        num_time_slots = len(sat_t)
        assignments = {t: {} for t in range(num_time_slots)}
        active_connections = defaultdict(list)  # {commodity_id: [(satellite, end_time, flow_amount)]}
        satellite_reserved_capacity = defaultdict(int)
        
        # Sort commodities by demand
        sorted_commodities = sorted(commodities.items(), key=lambda x: x[1], reverse=True)
        
        # Track remaining demand for each commodity
        remaining_demands = {comm_id: demand for comm_id, demand in commodities.items()}
        
        for t in range(num_time_slots):
            # Update active connections and release expired reservations
            for comm_id in list(active_connections.keys()):
                active_connections[comm_id] = [
                    (sat, end_t, flow) for sat, end_t, flow in active_connections[comm_id]
                    if end_t > t
                ]
                # Update reserved capacities
                for sat, _, flow in active_connections[comm_id]:
                    satellite_reserved_capacity[sat] = sum(
                        f for s, e, f in active_connections[comm_id] 
                        if s == sat and e > t
                    )
                
                if not active_connections[comm_id]:
                    del active_connections[comm_id]
            
            # Collect unconnected and partially connected commodities
            commodities_to_process = []
            for comm_id, total_demand in sorted_commodities:
                current_flow = sum(flow for _, _, flow in active_connections[comm_id])
                if current_flow < total_demand:
                    remaining_demands[comm_id] = total_demand - current_flow
                    commodities_to_process.append((comm_id, remaining_demands[comm_id]))
            
            # Process each commodity
            for comm_id, remaining_demand in commodities_to_process:
                # First, find satellites that can handle full demand
                full_service_satellites = []
                for sat in sat_t[t]:
                    satellite = str(sat)
                    remaining_capacity = node_capacity[satellite] - satellite_reserved_capacity[satellite]
                    if remaining_capacity >= remaining_demand:
                        visibility = calculate_visible_time(sat_t, t, satellite)
                        full_service_satellites.append((satellite, visibility, remaining_capacity))
                
                if full_service_satellites:
                    # If satellites can handle full demand, pick the one with longest visibility
                    full_service_satellites.sort(key=lambda x: x[1], reverse=True)
                    best_satellite, visibility, _ = full_service_satellites[0]
                    
                    # Make assignment
                    end_time = t + visibility
                    active_connections[comm_id].append((best_satellite, end_time, remaining_demand))
                    satellite_reserved_capacity[best_satellite] += remaining_demand
                    
                else:
                    # Otherwise, iteratively assign to satellites with longest visibility
                    remaining_to_assign = remaining_demand
                    used_satellites = set()
                    
                    while remaining_to_assign > 0:
                        # Find available satellites
                        available_satellites = []
                        for sat in sat_t[t]:
                            satellite = str(sat)
                            if satellite not in used_satellites:
                                remaining_capacity = node_capacity[satellite] - satellite_reserved_capacity[satellite]
                                if remaining_capacity > 0:
                                    visibility = calculate_visible_time(sat_t, t, satellite)
                                    available_satellites.append((satellite, visibility, remaining_capacity))
                        
                        if not available_satellites:
                            break
                        
                        # Pick satellite with longest visibility
                        available_satellites.sort(key=lambda x: x[1], reverse=True)
                        best_satellite, visibility, capacity = available_satellites[0]
                        
                        # Assign flow
                        flow_amount = min(remaining_to_assign, capacity)
                        end_time = t + visibility
                        active_connections[comm_id].append((best_satellite, end_time, flow_amount))
                        satellite_reserved_capacity[best_satellite] += flow_amount
                        remaining_to_assign -= flow_amount
                        used_satellites.add(best_satellite)
            
            # Record assignments for this time slot
            assignments[t] = defaultdict(dict)
            for comm_id, connections in active_connections.items():
                for satellite, _, flow in connections:
                    assignments[t][comm_id][satellite] = flow
        
        # Write results to file
        write_mvt_results_to_file(assignments, file_MVT_path, file_MVT_HO)
        
        return assignments


# Instantiate the MVT class for direct import
MVT = MVT()

