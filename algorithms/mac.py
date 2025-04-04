from typing import List, Dict, Set, Tuple
from collections import defaultdict

def write_path_output(user_paths, handovers_per_user, file_path):
    """
    Writes the path output to the specified file, matching the format in the example.
    Only outputs connections when they change.
    """
    with open(file_path, 'w') as f:
        f.write("Handover Summary\n")
        f.write("================\n\n")
        
        # Calculate total handovers
        total_handovers = sum(handovers_per_user.values())
        f.write(f"Total number of handovers: {total_handovers}\n\n")
        
        # Handovers per commodity
        f.write("Handovers per commodity:\n")
        commodity_count = len(user_paths)
        for i in range(commodity_count):
            user = list(sorted(user_paths.keys()))[i] if i < len(user_paths) else None
            handovers = handovers_per_user.get(user, 0) if user else 0
            f.write(f"Commodity {i}: {handovers} handovers\n")
        
        f.write("\nSatellite Connection Sequences\n")
        f.write("============================\n\n")
        
        # Print each user's path
        for i, (user, path) in enumerate(sorted(user_paths.items())):
            f.write(f"Commodity {i}:\n")
            f.write("Time  Satellites (Flow Amount)\n")
            f.write("----------------------------\n")
            
            # Track connections for the sequence summary
            sequence_connections = []
            prev_connections = None
            
            for t, connections in enumerate(path):
                # Skip if no connections at this time
                if not connections:
                    continue
                
                # Sort connections for consistent comparison
                sorted_connections = sorted(connections)
                
                # Check if connections have changed from previous time slot
                if prev_connections != sorted_connections:
                    sat_strs = [f"{sat}({flow})" for sat, flow in sorted_connections]
                    f.write(f"t={t}    {', '.join(sat_strs)}\n")
                    
                    # Store for sequence summary
                    sequence_connections.append((t, sorted_connections))
                    
                    # Update previous connections
                    prev_connections = sorted_connections
            
            # Print the sequence summary
            f.write("\nSequence with flows:\n")
            
            if len(sequence_connections) <= 1:
                # If only one time slot has connections, just list them
                if sequence_connections:
                    sat_strs = [f"{sat}({flow})" for sat, flow in sequence_connections[0][1]]
                    f.write(f"{', '.join(sat_strs)}\n\n")
                else:
                    f.write("\n\n")
            else:
                # Create a readable sequence representation with arrows between time slots
                time_blocks = []
                for _, connections in sequence_connections:
                    sat_strs = [f"{sat}({flow})" for sat, flow in connections]
                    time_blocks.append(' + '.join(sat_strs))
                
                sequence_str = " -> ".join(time_blocks)
                f.write(f"{sequence_str}\n\n")


def write_handover_output(handover_count, file_path):
    """
    Writes the total handover count to the specified file.
    """
    with open(file_path, 'w') as f:
        f.write(f"{handover_count}\n")


class MAC:
    """
    Minimum Assignment Changes (MAC) algorithm for satellite scheduling.
    This algorithm tries to maintain existing connections when possible to minimize handovers.
    """
    
    def __init__(self):
        """Initialize the MAC algorithm."""
        pass
        
    def __call__(self, sat_t: List[List[str]], node_capacity: Dict[str, int], 
               commodities: Dict[str, int], file_MAC_path: str, file_MAC_HO: str):
        """
        Run the MAC algorithm with the provided parameters.
        This method allows the class to be called as a function.
        """
        return self.run(sat_t, node_capacity, commodities, file_MAC_path, file_MAC_HO)
    
    def run(self, sat_t: List[List[str]], node_capacity: Dict[str, int], 
               commodities: Dict[str, int], file_MAC_path: str, file_MAC_HO: str):
        """
        Determines the connection between satellites and users and calculates handovers.
        
        Args:
            sat_t: List of lists where sat_t[t] represents satellites visible at time t
            node_capacity: Dictionary mapping satellite name to its capacity
            commodities: Dictionary mapping user name to its demand
            file_MAC_path: Output file path for connection paths
            file_MAC_HO: Output file path for handover count
            
        Returns:
            Tuple of (handover_count, user_paths)
        """
        # Get total number of time slots
        num_time_slots = len(sat_t)
        
        # Initialize data structures
        # For each user (commodity), track its flow through satellites at each time
        user_paths = {user: [[] for _ in range(num_time_slots)] for user in commodities}
        
        # Verify if all user demands can be satisfied
        capacity_sufficient = True
        for t in range(num_time_slots):
            visible_sats = sat_t[t]
            total_capacity = sum(node_capacity.get(sat, 0) for sat in visible_sats)
            total_demand = sum(commodities.values())
            
            if total_capacity < total_demand:
                capacity_sufficient = False
                print(f"Warning: At time {t}, total satellite capacity ({total_capacity}) " +
                      f"is less than total user demand ({total_demand}).")
        
        if not capacity_sufficient:
            print("Not all demands may be fully served due to insufficient capacity.")
        
        # Process first time slot (t=0) - initial assignment for all users
        t = 0
        visible_sats = sat_t[t]
        remaining_capacity = {sat: node_capacity.get(sat, 0) for sat in visible_sats}
        
        # Sort users by their demand (prioritize smaller demands to maximize number of satisfied users)
        sorted_users = sorted(commodities.items(), key=lambda x: x[1])
        
        # Initial assignment for all users
        for user, demand in sorted_users:
            remaining_demand = demand
            
            # Sort visible satellites by remaining capacity (descending)
            sorted_sats = sorted(visible_sats, key=lambda s: remaining_capacity.get(s, 0), reverse=True)
            
            # Assign satellites to user based on remaining capacity
            for sat in sorted_sats:
                if remaining_capacity.get(sat, 0) <= 0 or remaining_demand <= 0:
                    continue
                    
                # Determine flow amount (min of satellite capacity and remaining demand)
                flow_amount = min(remaining_capacity[sat], remaining_demand)
                
                # Update remaining capacity and demand
                remaining_capacity[sat] -= flow_amount
                remaining_demand -= flow_amount
                
                # Record the connection
                user_paths[user][t].append((sat, flow_amount))
                
                # If demand is fully satisfied, break
                if remaining_demand <= 0:
                    break
        
        # For subsequent time slots, try to maintain existing connections when possible
        for t in range(1, num_time_slots):
            # Get available satellites at current time
            current_visible_sats = set(sat_t[t])
            prev_visible_sats = set(sat_t[t-1])
            
            # Create a copy of node_capacity to track remaining capacity
            remaining_capacity = {sat: node_capacity.get(sat, 0) for sat in current_visible_sats}
            
            # Track which users need reassignment (those with satellites that are no longer visible)
            users_needing_reassignment = {}
            
            # First, try to maintain existing connections when satellites remain visible
            for user, demand in commodities.items():
                prev_connections = user_paths[user][t-1]
                
                # Check which of the user's previous satellites are still visible
                still_visible_connections = []
                lost_capacity = 0
                
                for sat, flow in prev_connections:
                    if sat in current_visible_sats:
                        # Satellite is still visible, maintain connection if possible
                        # Check if we can maintain the same flow amount
                        maintainable_flow = min(flow, remaining_capacity[sat])
                        
                        if maintainable_flow > 0:
                            # Maintain this connection
                            still_visible_connections.append((sat, maintainable_flow))
                            remaining_capacity[sat] -= maintainable_flow
                            
                            # If we couldn't maintain the full flow, track the lost capacity
                            if maintainable_flow < flow:
                                lost_capacity += (flow - maintainable_flow)
                    else:
                        # Satellite is no longer visible, this capacity needs reassignment
                        lost_capacity += flow
                
                # Carry forward the connections that are still viable
                user_paths[user][t].extend(still_visible_connections)
                
                # If the user has lost capacity, they need reassignment
                if lost_capacity > 0:
                    users_needing_reassignment[user] = lost_capacity
            
            # Sort users needing reassignment by their lost capacity (smaller first to maximize satisfied users)
            sorted_users_for_reassignment = sorted(users_needing_reassignment.items(), key=lambda x: x[1])
            
            # Reassign satellites for users who need it
            for user, lost_capacity in sorted_users_for_reassignment:
                # Skip if no capacity needed
                if lost_capacity <= 0:
                    continue
                    
                # Sort remaining visible satellites by capacity
                sorted_sats = sorted(current_visible_sats, key=lambda s: remaining_capacity.get(s, 0), reverse=True)
                
                for sat in sorted_sats:
                    if remaining_capacity.get(sat, 0) <= 0 or lost_capacity <= 0:
                        continue
                    
                    # Check if this user is already connected to this satellite
                    already_connected = False
                    for existing_sat, _ in user_paths[user][t]:
                        if existing_sat == sat:
                            already_connected = True
                            break
                    
                    # Skip if already connected (we'll handle flow updates separately)
                    if already_connected:
                        continue
                    
                    # Assign new satellite
                    flow_amount = min(remaining_capacity[sat], lost_capacity)
                    remaining_capacity[sat] -= flow_amount
                    lost_capacity -= flow_amount
                    
                    # Add the new connection
                    user_paths[user][t].append((sat, flow_amount))
                    
                    # If all lost capacity is reassigned, break
                    if lost_capacity <= 0:
                        break
                
                # If we still have unmet demand, try to increase flow on existing connections
                if lost_capacity > 0:
                    for sat in current_visible_sats:
                        # Skip if no remaining capacity or if user is not connected to this satellite
                        sat_connection = None
                        for i, (existing_sat, flow) in enumerate(user_paths[user][t]):
                            if existing_sat == sat:
                                sat_connection = (i, flow)
                                break
                        
                        if sat_connection is None or remaining_capacity.get(sat, 0) <= 0 or lost_capacity <= 0:
                            continue
                        
                        # Increase flow on existing connection
                        idx, current_flow = sat_connection
                        additional_flow = min(remaining_capacity[sat], lost_capacity)
                        user_paths[user][t][idx] = (sat, current_flow + additional_flow)
                        
                        remaining_capacity[sat] -= additional_flow
                        lost_capacity -= additional_flow
                        
                        # If all lost capacity is reassigned, break
                        if lost_capacity <= 0:
                            break
        
        # Count handovers
        handover_count = 0
        handovers_per_user = {}
        
        for user, path in user_paths.items():
            user_handovers = 0
            prev_sats = set()  # Set of satellites the user was connected to in previous time slot
            
            for t in range(num_time_slots):
                # Get current satellites
                current_sats = {sat for sat, _ in path[t]}
                
                if t > 0 and prev_sats:  # Skip the first time slot
                    # Count new connections (handovers)
                    for sat in current_sats:
                        if sat not in prev_sats:
                            user_handovers += 1
                
                prev_sats = current_sats
            
            handovers_per_user[user] = user_handovers
            handover_count += user_handovers
        
        # Write outputs
        write_path_output(user_paths, handovers_per_user, file_MAC_path)
        write_handover_output(handover_count, file_MAC_HO)
        
        return handover_count, user_paths


# Instantiate the MAC class for direct import
MAC = MAC()

