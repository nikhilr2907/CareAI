from dataclasses import dataclass, field
from typing import dict, list

@dataclass
class HospitalNode:
    # 1. Basic Identification
    node_id: str  # e.g., "Ward_10817" or "Hub_Area_A" [cite: 89, 243]
    node_type: str  # e.g., 'storage', 'corridor', 'recovery', 'hub' [cite: 216, 259, 268]
    
    # 2. Geometric Metadata (From Floorplan/Specs)
    x: float  # Cartesian coordinates for distance calculation
    y: float
    width_m: float  # Corridor/Door width (e.g., 1000mm for doors) [cite: 407]
    clearance_m: float  # Available space for turning (0.8m - 1.0m ideal) [cite: 504, 505]
    
    # 3. Storage & Interaction Metadata (Verticality)
    max_reach_height: float = 1.35  # Max height robot can reach (1350mm) [cite: 570, 580]
    unit_height: float = 2.1  # Physical height of the tall modular cabinets (2100mm) [cite: 562, 579]
    has_wash_basin: bool = False  # Flag for restricted parking near clinical sinks [cite: 267, 336]
    
    # 4. Sensory & Visual Cues (Digital Twin features)
    visibility_zones: dict[str, list[float]] = field(default_factory=lambda: {
        "lower": [0.5, 1.1], # >= 500mm and < 1100mm [cite: 507, 546]
        "upper": [1.1, 1.5]  # > 1100mm and < 1500mm [cite: 509, 514]
    })
    is_cluttered: bool = False  # Dynamic flag for nursing terminals/trolleys 
    
    # 5. Operational State
    current_occupancy: int = 0
    max_capacity: int = 1  # How many robots can fit in this specific zone