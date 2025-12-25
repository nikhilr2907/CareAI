from dataclasses import dataclass, field

@dataclass
class HospitalEdge:
    from_node: str
    to_node: str
    
    # Static Specs
    distance_m: float  # Physical length
    max_v_ms: float = 1.0  # Max speed (1 m/s from specs) [cite: 583]
    width_m: float = 1.9  # Corridor width 
    
    # Dynamic States (The 'State' part)
    clutter_level: float = 0.0  # 0.0 (clear) to 1.0 (blocked) 
    active_robot_ids: list[int] = field(default_factory=list)
    has_patient_bed: bool = False  # True if a bed is currently passing [cite: 421]
    
    @property
    def current_weight(self) -> float:
        """Calculates dynamic cost for MAPPO to perceive."""
        base_cost = self.distance_m / self.max_v_ms
        congestion_penalty = len(self.active_robot_ids) * 1.5
        clutter_penalty = self.clutter_level * 10
        return base_cost + congestion_penalty + clutter_penalty
    
