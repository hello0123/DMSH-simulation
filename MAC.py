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
        print(f"write_path_output(): write path to {file_path}")


def write_handover_output(handover_count, file_path):
    """
    Writes the total handover count to the specified file.
    """
    with open(file_path, 'w') as f:
        f.write(f"{handover_count}\n")
    print(f"write_handover_output(): write HO to {file_path}")


def MAC(sat_t, node_capacity, commodities, file_MAC_path, file_MAC_HO):
    """
    Determines the connection between satellites and users and calculates handovers.
    
    Parameters:
    - sat_t: List of lists where sat_t[t] represents satellites visible at time t
    - node_capacity: Dictionary mapping satellite name to its capacity
    - commodities: Dictionary mapping user name to its demand
    - file_MAC_path: Output file path for connection paths
    - file_MAC_HO: Output file path for handover count
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


def test_MAC():
    """
    Test function for the MAC implementation.
    """
    # Test case 1: Simple case to test connection continuity
    print("\nTest Case 1: Testing connection continuity")
    sat_t = [
        [101, 102, 103],         # Time 0: Satellites 101, 102, 103 visible
        [101, 102, 104],         # Time 1: Satellites 101, 102, 104 visible (103 disappears, 104 appears)
        [101, 104, 105]          # Time 2: Satellites 101, 104, 105 visible (102 disappears, 105 appears)
    ]
    
    # Sufficient capacity for all demands
    node_capacity = {
        101: 8,
        102: 8,
        103: 8,
        104: 8,
        105: 8
    }
    
    # User demands
    commodities = {
        "User1": 5,
        "User2": 7
    }
    
    file_MAC_path = "test_path_output.txt"
    file_MAC_HO = "test_handover_output.txt"
    
    handover_count, user_paths = MAC(sat_t, node_capacity, commodities, file_MAC_path, file_MAC_HO)
    
    print(f"Total handovers: {handover_count}")
    print("User paths:")
    for user, path in user_paths.items():
        print(f"{user}:")
        for t, connections in enumerate(path):
            if connections:
                print(f"  t={t}: {connections}")
    
    # Verify connection continuity
    print("\nVerifying connection continuity:")
    for user, path in user_paths.items():
        for t in range(1, len(path)):
            current_sats = {sat for sat, _ in path[t]}
            prev_sats = {sat for sat, _ in path[t-1]}
            
            # Identify maintained connections
            maintained = current_sats.intersection(prev_sats)
            new_connections = current_sats - prev_sats
            
            print(f"{user} at t={t}:")
            print(f"  Maintained connections: {maintained}")
            print(f"  New connections: {new_connections}")
    
    # Test case 2: Testing connection reassignment when satellites disappear
    print("\nTest Case 2: Testing connection reassignment")
    sat_t2 = [
        [201, 202, 203],         # Time 0: All satellites visible
        [201, 203],              # Time 1: Satellite 202 disappears
        [201]                    # Time 2: Satellite 203 disappears
    ]
    
    node_capacity2 = {
        201: 15,
        202: 10,
        203: 10
    }
    
    # Both users initially connect to different satellites
    commodities2 = {
        "UserA": 10,  # Will initially connect to 201
        "UserB": 8    # Will initially connect to 202, forcing reassignment at t=1
    }
    
    file_MAC_path2 = "test_path_output2.txt"
    file_MAC_HO2 = "test_handover_output2.txt"
    
    handover_count2, user_paths2 = MAC(sat_t2, node_capacity2, commodities2, file_MAC_path2, file_MAC_HO2)
    
    print(f"Total handovers: {handover_count2}")
    print("User paths:")
    for user, path in user_paths2.items():
        print(f"{user}:")
        for t, connections in enumerate(path):
            if connections:
                print(f"  t={t}: {connections}")
    
    # Verify that UserB gets reassigned when satellite 202 disappears
    userB_connections = user_paths2["UserB"]
    t0_sats = {sat for sat, _ in userB_connections[0]}
    t1_sats = {sat for sat, _ in userB_connections[1]}
    
    print(f"\nUserB satellites at t=0: {t0_sats}")
    print(f"UserB satellites at t=1: {t1_sats}")
    
    if 202 in t0_sats and 202 not in t1_sats:
        print("Verified: UserB was correctly reassigned when satellite 202 disappeared")
    
    # Count the expected number of handovers manually
    expected_handovers = 0
    for user, path in user_paths2.items():
        for t in range(1, len(path)):
            current_sats = {sat for sat, _ in path[t]}
            prev_sats = {sat for sat, _ in path[t-1]}
            
            for sat in current_sats:
                if sat not in prev_sats:
                    expected_handovers += 1
    
    print(f"Expected handovers: {expected_handovers}, Actual handovers: {handover_count2}")
    
    return handover_count, user_paths
    
    # Test case 2: Challenging case but still with sufficient total capacity
    print("\nTest Case 2: Challenging case with sufficient capacity")
    sat_t2 = [
        [201, 202],         # Time 0: Satellites 201, 202 visible
        [202, 203],         # Time 1: Satellites 202, 203 visible
        [201, 203]          # Time 2: Satellites 201, 203 visible
    ]
    
    # Increase capacity to just enough to meet demand
    node_capacity2 = {
        201: 6,
        202: 5,
        203: 5
    }
    
    # Total capacity at each time = 11, total demand = 10
    commodities2 = {
        "User1": 3,
        "User2": 7
    }
    
    file_MAC_path2 = "test_path_output2.txt"
    file_MAC_HO2 = "test_handover_output2.txt"
    
    handover_count2, user_paths2 = MAC(sat_t2, node_capacity2, commodities2, file_MAC_path2, file_MAC_HO2)
    
    print(f"Total handovers: {handover_count2}")
    print("User paths:")
    for user, path in user_paths2.items():
        print(f"{user}: {path}")
    
    # Check if demands are being met
    demand_satisfied = True
    for user, paths in user_paths2.items():
        for t, connections in enumerate(paths):
            total_flow = sum(flow for _, flow in connections)
            print(f"Time {t}, {user} flow: {total_flow}/{commodities2[user]}")
            if total_flow < commodities2[user]:
                demand_satisfied = False
                print(f"Warning: Demand not fully satisfied for {user} at time {t}")
    
    if demand_satisfied:
        print("All demands fully satisfied in Test Case 2!")
    
    if demand_satisfied:
        print("All demands fully satisfied in Test Case 2!")
    else:
        print("Some demands were not fully satisfied in Test Case 2!")
    
    # Test case 3: Complex case similar to the example file
    print("\nTest Case 3: Complex case similar to the example file")
    sat_t3 = [
        [30, 66, 180, 191, 202],  # Time 0
        [180, 191, 202, 212, 223], # Time 1
        [212, 223, 233, 244, 255]  # Time 2
    ]
    
    # Set sufficient capacities to ensure all demands can be met
    node_capacity3 = {
        30: 15, 66: 15, 180: 20, 191: 20, 202: 20,
        212: 20, 223: 20, 233: 20, 244: 20, 255: 20
    }
    
    # Define demands for multiple users
    commodities3 = {
        "User31": 15,
        "User32": 25,
        "User33": 10,
        "User34": 30
    }
    
    file_MAC_path3 = "test_path_output3.txt"
    file_MAC_HO3 = "test_handover_output3.txt"
    
    handover_count3, user_paths3 = MAC(sat_t3, node_capacity3, commodities3, file_MAC_path3, file_MAC_HO3)
    
    print(f"Total handovers: {handover_count3}")
    
    # Verify demand satisfaction
    print("Verifying all demands are fully satisfied:")
    all_satisfied = True
    for user, paths in user_paths3.items():
        for t, connections in enumerate(paths):
            total_flow = sum(flow for _, flow in connections)
            if total_flow < commodities3[user]:
                all_satisfied = False
                print(f"Warning: Time {t}, {user} flow: {total_flow}/{commodities3[user]} - NOT FULLY SATISFIED")
            else:
                print(f"Time {t}, {user} flow: {total_flow}/{commodities3[user]} - SATISFIED")
    
    if all_satisfied:
        print("All demands fully satisfied in Test Case 3!")
    else:
        print("Some demands were not fully satisfied in Test Case 3!")
    
    # Test case 4: Insufficient capacity scenario
    print("\nTest Case 4: Scenario with insufficient capacity")
    sat_t4 = [
        [301, 302],         # Time 0: Satellites 301, 302 visible
        [302, 303],         # Time 1: Satellites 302, 303 visible
        [301, 303]          # Time 2: Satellites 301, 303 visible
    ]
    
    # Deliberately set insufficient capacity
    node_capacity4 = {
        301: 3,
        302: 2,
        303: 3
    }
    
    # Total capacity at each time = 5-6, but total demand = 9
    commodities4 = {
        "UserA": 2,
        "UserB": 3,
        "UserC": 4
    }
    
    file_MAC_path4 = "test_path_output4.txt"
    file_MAC_HO4 = "test_handover_output4.txt"
    
    handover_count4, user_paths4 = MAC(sat_t4, node_capacity4, commodities4, file_MAC_path4, file_MAC_HO4)
    
    print(f"Total handovers: {handover_count4}")
    
    # Check how the demands were handled
    print("Checking demand satisfaction with insufficient capacity:")
    for user, paths in user_paths4.items():
        for t, connections in enumerate(paths):
            total_flow = sum(flow for _, flow in connections)
            satisfaction_percent = (total_flow / commodities4[user]) * 100
            print(f"Time {t}, {user} flow: {total_flow}/{commodities4[user]} ({satisfaction_percent:.1f}%)")
            
    # Check if smaller demands were prioritized
    small_users_satisfied = True
    for t in range(len(sat_t4)):
        # Get users with smallest demands
        smallest_users = sorted(commodities4.items(), key=lambda x: x[1])[:2]  # First two users
        
        for user, demand in smallest_users:
            total_flow = sum(flow for _, flow in user_paths4[user][t])
            if total_flow < demand:
                small_users_satisfied = False
                print(f"Small demand user {user} not fully satisfied at time {t}")
    
    if small_users_satisfied:
        print("Users with smaller demands were properly prioritized!")
    else:
        print("Priority for smaller demand users was not maintained.")
    
    return handover_count, user_paths


# Run the test if this file is executed directly
if __name__ == "__main__":
    test_MAC()



