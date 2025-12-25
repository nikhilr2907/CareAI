import torch
from typing import List, Dict
from dataclasses import dataclass

from src.environment.graph.node import HospitalNode
from src.environment.graph.edge import HospitalEdge

# Placeholder dataclasses for robot and demand states
@dataclass
class RobotState:
    robot_id: int
    current_node_id: str
    battery_level: float  # 0.0 to 1.0
    is_available: bool

from enum import Enum, auto
from typing import Optional

class TaskType(Enum):
    REPLENISHMENT = auto() # Hub -> Ward
    RETURNS = auto()      # Ward -> Hub
    AD_HOC = auto()       # Others (e.g., Ward <-> Ward / Pharmacy <-> Ward)

@dataclass
class Task:
    task_id: int # Unique identifier for the task
    task_type: TaskType
    from_location_id: str
    to_location_id: str
    item_id: Optional[str] = None # What is being moved
    priority: int = 1 # 1-5 as per gemini.md

def get_concatenated_state(
    nodes: Dict[str, HospitalNode],
    edges: List[HospitalEdge],
    robots: List[RobotState],
    tasks: List[Task]
) -> torch.Tensor:
    """
    Aggregates the state of the entire system into a single flat tensor.

    This function is crucial for feeding the system state into a policy network.
    It flattens the graph structure and combines it with robot and task information.

    Args:
        nodes: A dictionary of all nodes in the graph.
        edges: A list of all edges in the graph.
        robots: A list of the current states of all robots.
        demands: A list of all outstanding tasks.

    Returns:
        A 1D PyTorch tensor representing the full system state.
    """
    
    # 1. Environment State (Graph)
    # Important: The order of nodes and edges must be consistent across calls
    # Sorting by ID is a good way to ensure this.
    
    node_features = []
    sorted_node_ids = sorted(nodes.keys())
    for node_id in sorted_node_ids:
        node = nodes[node_id]
        # Normalize occupancy to prevent large numbers from dominating
        occupancy_ratio = node.current_occupancy / node.max_capacity if node.max_capacity > 0 else 0
        node_features.extend([
            node.x,
            node.y,
            float(node.is_cluttered),
            occupancy_ratio,
            node.max_reach_height
        ])

    edge_features = []
    # Assuming a consistent order for edges as well
    sorted_edges = sorted(edges, key=lambda e: (e.from_node, e.to_node))
    for edge in sorted_edges:
        edge_features.extend([
            edge.current_weight,
            float(edge.has_patient_bed),
            edge.clutter_level
        ])

    # 2. Robot State
    robot_features = []
    for robot in sorted(robots, key=lambda r: r.robot_id):
        # We need to represent the robot's location numerically.
        # One-hot encoding is a common way to do this.
        location_one_hot = [1.0 if nid == robot.current_node_id else 0.0 for nid in sorted_node_ids]
        robot_features.extend([
            robot.battery_level,
            1.0 if robot.is_available else 0.0,
            *location_one_hot
        ])

    # 3. Task State
    task_features = []
    for task in sorted(tasks, key=lambda t: t.task_id):
        # One-hot encode from and to locations
        from_one_hot = [1.0 if nid == task.from_location_id else 0.0 for nid in sorted_node_ids]
        to_one_hot = [1.0 if nid == task.to_location_id else 0.0 for nid in sorted_node_ids]
        
        # One-hot encode task type
        task_type_one_hot = [0.0] * len(TaskType)
        task_type_one_hot[task.task_type.value - 1] = 1.0 # -1 because auto() starts from 1
        
        # Normalize priority
        normalized_priority = task.priority / 5.0
        
        task_features.extend([
            normalized_priority,
            *task_type_one_hot,
            *from_one_hot,
            *to_one_hot
        ])
        
    # Concatenate all features into a single flat tensor
    final_state_tensor = torch.tensor(
        node_features + edge_features + robot_features + task_features,
        dtype=torch.float32
    )
    
    return final_state_tensor
