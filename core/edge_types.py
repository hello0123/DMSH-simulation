from enum import Enum

class EdgeType(Enum):
    """
    Enumeration of different edge types in the satellite network graph.
    """
    FORWARD = 1
    BACKWARD = 2
    VERTEX_FOR = 3
    VERTEX_REV = 4
    
    def __str__(self):
        return self.name
    
    