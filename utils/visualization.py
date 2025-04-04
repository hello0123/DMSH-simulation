import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import os
from glob import glob

def plot_algorithm_comparison(directory: str, name: str, output_file: str = None):
    """
    Plot the comparison of different algorithms based on their handover counts.
    
    Args:
        directory: Directory containing handover result files
        name: Name for x-axis label
        output_file: Optional output file path for saving the plot
    """
    # Define file patterns
    pack_path = f"{directory}/HO_pack_*.txt"
    mvt_path = f"{directory}/HO_MVT_*.txt"
    mac_path = f"{directory}/HO_MAC_*.txt"
    msh_path = f"{directory}/HO_MSH_*.txt"
    
    # Get all files matching the patterns
    pack_files = sorted(glob(pack_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mvt_files = sorted(glob(mvt_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mac_files = sorted(glob(mac_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    msh_files = sorted(glob(msh_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    list_x = []
    list_flow_data = []
    list_mvt_data = []
    list_mac_data = []
    list_msh_data = []

    for pack_file, mvt_file, mac_file, msh_file in zip(pack_files, mvt_files, mac_files, msh_files):
        # Extract integer from filename (taking the number before .txt)
        number = int(pack_file.split('_')[-1].replace('.txt', ''))
        list_x.append(number)
        
        try:
            list_flow_data.append(np.loadtxt(pack_file))
            list_mvt_data.append(np.loadtxt(mvt_file))
            list_mac_data.append(np.loadtxt(mac_file))
            list_msh_data.append(np.loadtxt(msh_file))
        except Exception as e:
            print(f"Error loading data from file: {e}")

    # Convert lists to numpy arrays
    x = np.array(list_x)
    pack_data_array = np.array(list_flow_data)
    mvt_data_array = np.array(list_mvt_data)
    mac_data_array = np.array(list_mac_data)
    msh_data_array = np.array(list_msh_data)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot the lines
    plt.plot(x, pack_data_array, 'b-', label='Pack Method', linewidth=2)
    plt.plot(x, mvt_data_array, 'r--', label='MVT', linewidth=2)
    plt.plot(x, mac_data_array, 'g:', label='MAC', linewidth=2)
    plt.plot(x, msh_data_array, 'm-', label='MSH', linewidth=2)

    # Customize the plot
    plt.title('Performance Comparison', fontsize=14, pad=15)
    plt.xlabel(f"{name}", fontsize=12)
    plt.ylabel('Number of handovers', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Add x-ticks for each file
    plt.xticks(x)

    # Set x-axis limits based on list_x values
    plt.xlim(min(list_x), max(list_x))

    # Adjust layout
    plt.tight_layout()

    # Save the plot if output file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

    # Print data arrays for verification
    print("\nPack Data Array:")
    print(pack_data_array)
    print("\nMVT Data Array:")
    print(mvt_data_array)
    print("\nMAC Data Array:")
    print(mac_data_array)
    print("\nMSH Data Array:")
    print(msh_data_array)

    # Clear the plot from memory
    plt.close()

    