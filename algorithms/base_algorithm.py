from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class BaseAlgorithm(ABC):
    """
    Abstract base class for satellite scheduling algorithms.
    """
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the algorithm with the provided parameters.
        """
        pass
    
    def write_path_output(self, user_paths, handovers_per_user, file_path):
        """
        Writes the path output to the specified file.
        """
        with open(file_path, 'w') as f:
            f.write("Handover Summary\n")
            f.write("================\n\n")
            
            # Calculate total handovers
            total_handovers = sum(handovers_per_user.values())
            f.write(f"Total number of handovers: {total_handovers}\n\n")
            
            # Handovers per commodity
            f.write("Handovers per commodity:\n")
            for i, user in enumerate(sorted(user_paths.keys())):
                handovers = handovers_per_user.get(user, 0)
                f.write(f"Commodity {i}: {handovers} handovers\n")
            
            f.write("\nSatellite Connection Sequences\n")
            f.write("============================\n\n")
            
            # Print each user's path
            for i, (user, path) in enumerate(sorted(user_paths.items())):
                f.write(f"Commodity {i}:\n")
                self._write_user_path(f, path)
                f.write("\n")
    
    def _write_user_path(self, file_handle, path):
        """
        Write a specific user's path to the file.
        
        To be implemented by subclasses if special formatting is needed.
        """
        file_handle.write(str(path))
    
    def write_handover_output(self, handover_count, file_path):
        """
        Writes the total handover count to the specified file.
        """
        with open(file_path, 'w') as f:
            f.write(f"{handover_count}\n")
        print(f"Wrote total handover count ({handover_count}) to {file_path}")

        