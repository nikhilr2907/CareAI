from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class RobotTelemetry:
    timestamp: float  # When this telemetry was captured (Unix time or sim time)
    robot_id: int

    # ===== POSITION DATA =====
    x: float  # Current X coordinate (meters)
    y: float  # Current Y coordinate (meters)
    heading: float  # Direction in radians (0 = East, π/2 = North)

    # ===== GRAPH-AWARE POSITION =====
    current_node_index: Optional[int]  # Node index if at a node, None if on edge
    current_edge_index: Optional[int]  # Edge index if on edge, None if at node
    edge_progress: float  # 0.0-1.0 if on edge, 0.0 if at node

    # ===== MOTION DATA =====
    velocity_ms: float  # Current speed in m/s
    is_moving: bool  # Whether robot is actively moving

    # ===== OPERATIONAL STATE =====
    battery_level: float  # 0.0-1.0 (0% to 100%)
    current_capacity: int  # Number of items currently loaded
    is_available: bool  # Ready for new task assignment

    # ===== TASK EXECUTION STATE =====
    active_task_id: Optional[int]  # Currently executing task ID
    remaining_path: list  # List of node indices still to visit
    eta_to_next_node: float  # Estimated seconds to next waypoint

    # ===== CHARGING STATE (with defaults) =====
    is_charging: bool = False  # Robot is docked and charging
    needs_charging: bool = False  # Dynamic threshold crossed, return to hub needed

    @property
    def is_at_node(self) -> bool:
        """Check if robot is stationary at a node."""
        return self.current_node_index is not None and self.current_edge_index is None

    @property
    def is_on_edge(self) -> bool:
        """Check if robot is traveling along an edge."""
        return self.current_edge_index is not None

    def to_dict(self):
        """Convert to dictionary for logging/serialization."""
        return {
            'timestamp': self.timestamp,
            'robot_id': self.robot_id,
            'x': self.x,
            'y': self.y,
            'heading': self.heading,
            'current_node_index': self.current_node_index,
            'current_edge_index': self.current_edge_index,
            'edge_progress': self.edge_progress,
            'velocity_ms': self.velocity_ms,
            'is_moving': self.is_moving,
            'battery_level': self.battery_level,
            'current_capacity': self.current_capacity,
            'is_available': self.is_available,
            'active_task_id': self.active_task_id,
            'eta_to_next_node': self.eta_to_next_node,
            'is_charging': self.is_charging,
            'needs_charging': self.needs_charging
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create telemetry from dictionary (for real data ingestion)."""
        return cls(
            timestamp=data['timestamp'],
            robot_id=data['robot_id'],
            x=data['x'],
            y=data['y'],
            heading=data['heading'],
            current_node_index=data.get('current_node_index'),
            current_edge_index=data.get('current_edge_index'),
            edge_progress=data.get('edge_progress', 0.0),
            velocity_ms=data['velocity_ms'],
            is_moving=data['is_moving'],
            battery_level=data['battery_level'],
            current_capacity=data['current_capacity'],
            is_available=data['is_available'],
            active_task_id=data.get('active_task_id'),
            remaining_path=data.get('remaining_path', []),
            eta_to_next_node=data.get('eta_to_next_node', 0.0),
            is_charging=data.get('is_charging', False),
            needs_charging=data.get('needs_charging', False)
        )