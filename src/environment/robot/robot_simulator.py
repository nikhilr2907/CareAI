"""
Robot simulator that generates telemetry data.
Simulates physical robot movement and can be replaced with real robot interface.
"""
import numpy as np
from typing import Optional, List
from .robot_telemetry import RobotTelemetry


class RobotSimulator:
    """
    Simulates a physical robot and generates telemetry.
    This class can be replaced with a real robot interface that reads from ROS topics/MQTT.
    """

    def __init__(self, robot_id: int, graph_state, initial_node_index: int,
                 max_capacity: int = 12):
        self.robot_id = robot_id
        self.graph_state = graph_state
        self.max_capacity = max_capacity

        # ===== PHYSICAL STATE =====
        initial_node = graph_state.get_node_by_index(initial_node_index)
        self.x = initial_node.x
        self.y = initial_node.y
        self.heading = 0.0
        self.velocity_ms = 0.0
        self.max_velocity_ms = 1.0  # From spec: 1 m/s

        # ===== GRAPH POSITION =====
        self.current_node_index = initial_node_index
        self.current_edge_index = None
        self.edge_progress = 0.0

        # ===== MOTION PLANNING =====
        self.path_queue = []  # List of node indices to visit
        self.current_target_node = None

        # ===== RESOURCES =====
        self.battery_level = 1.0  # Start at 100%
        self.max_battery = 1.0
        self.battery_drain_rate = 0.001  # per meter traveled
        self.current_capacity = 0

        # ===== TASK TRACKING =====
        self.active_task_id = None

    def set_path(self, path: List[int], task_id: int, num_items: int):
        """
        Command robot to follow a path (from task assignment).

        Args:
            path: List of node indices to visit (includes current node)
            task_id: ID of task being executed
            num_items: Number of items to load for this task
        """
        if len(path) > 1:
            self.path_queue = path[1:]  # Exclude current node
            self.current_target_node = path[1]
            self.active_task_id = task_id

            # Start moving toward first target
            self._start_edge_traversal(self.current_node_index, self.current_target_node)

    def _start_edge_traversal(self, from_node_idx: int, to_node_idx: int):
        """Begin traversing an edge."""
        self.current_node_index = from_node_idx
        self.current_edge_index = self._find_edge_index(from_node_idx, to_node_idx)
        self.edge_progress = 0.0
        self.velocity_ms = self.max_velocity_ms

        # Calculate heading
        from_node = self.graph_state.get_node_by_index(from_node_idx)
        to_node = self.graph_state.get_node_by_index(to_node_idx)
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        self.heading = np.arctan2(dy, dx)

    def update(self, time_delta: float) -> RobotTelemetry:
        """
        Simulate robot movement for time_delta seconds.
        Returns updated telemetry.

        Args:
            time_delta: Time step in seconds

        Returns:
            RobotTelemetry object with current state
        """
        if not self.path_queue and not self.current_target_node:
            # Robot is idle
            self.velocity_ms = 0.0
            return self.get_telemetry()

        if self.current_edge_index is not None:
            # Robot is on an edge
            edge = self.graph_state.edges[self.current_edge_index]
            distance_traveled = self.velocity_ms * time_delta

            # Update progress
            progress_delta = distance_traveled / edge.distance_m
            self.edge_progress += progress_delta

            # Update battery
            self.battery_level -= self.battery_drain_rate * distance_traveled
            self.battery_level = max(0.0, self.battery_level)

            # Update position (interpolate along edge)
            from_node = self.graph_state.get_node_by_index(self.current_node_index)
            to_node = self.graph_state.get_node_by_index(self.current_target_node)

            progress = min(self.edge_progress, 1.0)
            self.x = from_node.x + (to_node.x - from_node.x) * progress
            self.y = from_node.y + (to_node.y - from_node.y) * progress

            # Check if reached target node
            if self.edge_progress >= 1.0:
                self._arrive_at_node(self.current_target_node)

        return self.get_telemetry()

    def _arrive_at_node(self, node_idx: int):
        """Handle arrival at a node."""
        # Update position to exact node coordinates
        node = self.graph_state.get_node_by_index(node_idx)
        self.x = node.x
        self.y = node.y

        # Clear edge state
        self.current_node_index = node_idx
        self.current_edge_index = None
        self.edge_progress = 0.0

        # Check if more path segments remain
        if self.path_queue:
            next_node = self.path_queue.pop(0)
            self.current_target_node = next_node
            self._start_edge_traversal(node_idx, next_node)
        else:
            # Reached final destination
            self.current_target_node = None
            self.velocity_ms = 0.0
            # Note: active_task_id and capacity cleared by environment when task completes

    def complete_task(self, num_items: int):
        """
        Called by environment when task is completed.
        Unloads items and clears active task.
        """
        self.current_capacity -= num_items
        self.current_capacity = max(0, self.current_capacity)
        self.active_task_id = None

    def load_items(self, num_items: int):
        """Load items onto the robot (pickup leg)."""
        self.current_capacity += num_items
        self.current_capacity = min(self.current_capacity, self.max_capacity)

    def get_telemetry(self) -> RobotTelemetry:
        """Generate current telemetry snapshot."""
        # Calculate ETA to next node
        eta = 0.0
        if self.current_edge_index is not None:
            edge = self.graph_state.edges[self.current_edge_index]
            remaining_distance = edge.distance_m * (1.0 - self.edge_progress)
            eta = remaining_distance / self.velocity_ms if self.velocity_ms > 0 else 0.0

        # Determine availability
        is_available = (
            not self.path_queue and
            not self.current_target_node and
            self.current_capacity == 0
        )

        return RobotTelemetry(
            timestamp=0.0,  # Will be set by environment
            robot_id=self.robot_id,
            x=self.x,
            y=self.y,
            heading=self.heading,
            current_node_index=self.current_node_index if self.current_edge_index is None else None,
            current_edge_index=self.current_edge_index,
            edge_progress=self.edge_progress,
            velocity_ms=self.velocity_ms,
            is_moving=self.velocity_ms > 0.01,
            battery_level=self.battery_level,
            current_capacity=self.current_capacity,
            is_available=is_available,
            active_task_id=self.active_task_id,
            remaining_path=self.path_queue.copy(),
            eta_to_next_node=eta
        )

    def _find_edge_index(self, from_idx: int, to_idx: int) -> Optional[int]:
        """Find edge index connecting two nodes (bidirectional)."""
        from_node = self.graph_state.get_node_by_index(from_idx)
        to_node = self.graph_state.get_node_by_index(to_idx)

        for i, edge in enumerate(self.graph_state.edges):
            if ((edge.from_node == from_node.node_id and edge.to_node == to_node.node_id) or
                (edge.from_node == to_node.node_id and edge.to_node == from_node.node_id)):
                return i
        return None

    def _find_nearest_node(self, x: float, y: float) -> int:
        """Find nearest node to given coordinates."""
        min_dist = float('inf')
        nearest_idx = 0

        for i, node in enumerate(self.graph_state.nodes):
            dist = np.sqrt((node.x - x)**2 + (node.y - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx
