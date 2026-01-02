from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
import numpy as np
from .robot_telemetry import RobotTelemetry

if TYPE_CHECKING:
    from ..tasks.task_state import Task


@dataclass
class RobotState:
    """
    Represents the state of an individual robot combining task assignments and telemetry.
    Telemetry can come from simulator or real robot sensors.

    Supports multi-capacity task queuing:
    - Robots can carry up to 12 items
    - Can accept multiple tasks if capacity allows
    - Tasks queued by priority (high priority first)
    """
    robot_id: int
    max_capacity: int = 12  # Maximum items robot can carry

    # ===== TASK ASSIGNMENTS (Environment-controlled) =====
    task_queue: List['Task'] = field(default_factory=list)  # Ordered list of Task objects
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
        return len(self.task_queue)

    @property
    def current_task(self) -> Optional['Task']:
        """Get the current task (first in queue)."""
        return self.task_queue[0] if self.task_queue else None

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
    def current_load(self) -> int:
        """Get current number of items loaded (from telemetry)."""
        if self.telemetry:
            return self.telemetry.current_capacity
        return 0

    @property
    def available_capacity(self) -> int:
        """How many more items can be loaded."""
        return self.max_capacity - self.current_load

    @property
    def is_available(self) -> bool:
        """Available if no queued tasks AND telemetry confirms idle."""
        if self.telemetry:
            return len(self.task_queue) == 0 and self.telemetry.is_available
        return len(self.task_queue) == 0

    @property
    def can_accept_task(self) -> bool:
        """Robot can accept tasks even when busy if it has capacity."""
        return self.available_capacity > 0

    def add_task(self, task: 'Task'):
        """
        Add task to robot's queue, maintaining priority order (high priority first).

        Args:
            task: Task to add to queue
        """
        self.task_queue.append(task)
        # Sort by priority (high to low), then by urgency_score
        self.task_queue.sort(key=lambda t: (-t.manual_priority, -t.urgency_score))

    def complete_current_task(self) -> Optional['Task']:
        """
        Mark current task complete and remove from queue.

        Returns:
            Completed task, or None if no tasks in queue
        """
        if self.task_queue:
            completed = self.task_queue.pop(0)
            # Reset path if no more tasks
            if not self.task_queue:
                self.planned_path.clear()
                self.target_node_index = None
            return completed
        return None

    def get_total_queued_items(self) -> int:
        """Get total number of items across all queued tasks."""
        return sum(task.num_items for task in self.task_queue)

    def can_accept_items(self, num_items: int) -> bool:
        """Check if robot can accept additional items."""
        total_items = self.current_load + self.get_total_queued_items()
        return (total_items + num_items) <= self.max_capacity

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
            distance = np.sqrt((node.center_x - robot_x)**2 + (node.center_y - robot_y)**2)
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
