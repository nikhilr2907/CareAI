from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .robot_telemetry import RobotTelemetry


@dataclass
class RobotState:
    """
    Represents the state of an individual robot combining task assignments and telemetry.
    Telemetry can come from simulator or real robot sensors.
    """
    robot_id: int
    max_capacity: int = 12  # Maximum items robot can carry

    # ===== TASK ASSIGNMENTS (Environment-controlled) =====
    queued_tasks: List[int] = field(default_factory=list)  # List of task_ids
    target_node_index: Optional[int] = None  # Ultimate destination for current task
    planned_path: List[int] = field(default_factory=list)  # Full path for current task
    travel_start_time: float = 0.0  # When robot started current journey

    # ===== TELEMETRY (Robot simulator/hardware reports) =====
    telemetry: Optional[RobotTelemetry] = None  # Latest sensor data

    def update_telemetry(self, telemetry: RobotTelemetry):
        """Receive new telemetry update from robot."""
        self.telemetry = telemetry

    @property
    def num_queued_tasks(self) -> int:
        """Get number of queued tasks."""
        return len(self.queued_tasks)

    @property
    def current_position(self) -> tuple:
        """Get latest reported position from telemetry."""
        if self.telemetry:
            return (self.telemetry.x, self.telemetry.y)
        return (0.0, 0.0)

    @property
    def current_node_index(self) -> Optional[int]:
        """Get current node if robot is at a node (from telemetry)."""
        if self.telemetry:
            return self.telemetry.current_node_index
        return None

    @property
    def battery_level(self) -> float:
        """Get current battery level from telemetry."""
        if self.telemetry:
            return self.telemetry.battery_level
        return 0.0

    @property
    def current_capacity(self) -> int:
        """Get current number of items loaded (from telemetry)."""
        if self.telemetry:
            return self.telemetry.current_capacity
        return 0

    @property
    def remaining_capacity(self) -> int:
        """How many more items can be loaded."""
        return self.max_capacity - self.current_capacity

    @property
    def is_available(self) -> bool:
        """Available if no queued tasks AND telemetry confirms idle."""
        if self.telemetry:
            return len(self.queued_tasks) == 0 and self.telemetry.is_available
        return len(self.queued_tasks) == 0

    def compute_distances_to_nodes(self, graph_state) -> np.ndarray:
        """
        Compute Euclidean distances from robot's current position to all nodes.

        Args:
            graph_state: GraphState object containing nodes

        Returns:
            numpy array of distances to all nodes
        """
        distances = []
        robot_x, robot_y = self.current_position

        for node in graph_state.nodes:
            distance = np.sqrt((node.x - robot_x)**2 + (node.y - robot_y)**2)
            distances.append(distance)

        return np.array(distances, dtype=np.float32)


def create_default_robot(robot_id: int) -> RobotState:
    """
    Create a robot with default state.
    Telemetry will be populated by robot simulator.
    """
    return RobotState(
        robot_id=robot_id,
        max_capacity=12
    )
