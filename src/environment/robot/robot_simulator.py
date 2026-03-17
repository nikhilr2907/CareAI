"""
Robot simulator that generates telemetry data.
Simulates physical robot movement and can be replaced with real robot interface.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict


from .robot_telemetry import RobotTelemetry


@dataclass
class EdgeTraversalRecord:
    """Record of a single robot traversal of an edge, used to train edge cost model."""
    edge_index: int
    robot_id: int

    # Timing
    entry_time: float = 0.0
    exit_time: float = 0.0

    # Static edge properties (captured at entry)
    distance_m: float = 0.0
    corridor_width: float = 1.9
    max_v_ms: float = 1.0

    # Congestion snapshot at entry
    num_robots_on_edge: int = 0
    same_direction_count: int = 0
    opposite_direction_count: int = 0
    people_count: int = 0
    clutter_level: float = 0.0
    approaching_robot_count: int = 0
    from_node_occupancy: int = 0
    to_node_occupancy: int = 0

    # Time feature
    time_of_day: float = 0.0  # Normalized 0-1

    # Accumulated during traversal
    stop_count: int = 0  # Number of times velocity dropped to ~0
    velocity_samples: list = field(default_factory=list)  # For variance calculation

    @property
    def actual_traversal_time(self) -> float:
        return self.exit_time - self.entry_time

    @property
    def base_traversal_time(self) -> float:
        if self.max_v_ms <= 0:
            return 0.0
        return self.distance_m / self.max_v_ms

    @property
    def delay_factor(self) -> float:
        base = self.base_traversal_time
        if base <= 0:
            return 1.0
        return self.actual_traversal_time / base

    @property
    def velocity_variance(self) -> float:
        if len(self.velocity_samples) < 2:
            return 0.0
        return float(np.var(self.velocity_samples))

    @property
    def avg_velocity_ratio(self) -> float:
        if not self.velocity_samples or self.max_v_ms <= 0:
            return 1.0
        return float(np.mean(self.velocity_samples)) / self.max_v_ms

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for model input.

        Note: same_direction_count, opposite_direction_count, and stop_count
        are excluded — they are not available at inference time (direction is
        unknown when predicting before traversal; stop_count accumulates during
        traversal). Using them at training but not inference causes a mismatch.
        num_robots_on_edge captures the total congestion signal instead.
        """
        return np.array([
            self.distance_m,
            self.corridor_width,
            self.num_robots_on_edge,
            self.people_count,
            self.clutter_level,
            self.approaching_robot_count,
            self.from_node_occupancy,
            self.to_node_occupancy,
            self.time_of_day,
        ], dtype=np.float32)


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
        self.prev_velocity_ms = 0.0  # For stop detection
        self.max_velocity_ms = 0.5  # From spec: 1 m/s

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

        # ===== CHARGING =====
        self.is_charging = False
        self.charging_rate = 0.001  # battery per second while docked

        # ===== TASK TRACKING =====
        self.active_task_id = None

        # ===== TRAVERSAL TRACKING (for edge cost model) =====
        self.traversal_records: List[EdgeTraversalRecord] = []
        self._current_traversal: Optional[EdgeTraversalRecord] = None

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
        else:
            # No movement needed; clear any stale path/target
            self.path_queue = []
            self.current_target_node = None
            self.current_edge_index = None
            self.edge_progress = 0.0
            self.velocity_ms = 0.0
            self.active_task_id = task_id

    def _start_edge_traversal(self, from_node_idx: int, to_node_idx: int):
        """Begin traversing an edge."""
        self.current_node_index = from_node_idx
        self.current_edge_index = self._find_edge_index(from_node_idx, to_node_idx)
        if self.current_edge_index is None:
            # Invalid edge; reset path state to allow replanning
            self.path_queue = []
            self.current_target_node = None
            self.edge_progress = 0.0
            self.velocity_ms = 0.0
            return
        self.edge_progress = 0.0
        self.velocity_ms = self.max_velocity_ms

        # Calculate heading
        from_node = self.graph_state.get_node_by_index(from_node_idx)
        to_node = self.graph_state.get_node_by_index(to_node_idx)
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        self.heading = np.arctan2(dy, dx)

        # Start tracking traversal for edge cost model
        edge = self.graph_state.edges[self.current_edge_index]

        # Count directional robots on edge
        same_dir = 0
        opposite_dir = 0
        for rid, (prog, fidx, tidx) in edge.active_robot_progress.items():
            if rid == self.robot_id:
                continue
            if fidx == from_node_idx and tidx == to_node_idx:
                same_dir += 1
            else:
                opposite_dir += 1

        # Count approaching robots (have this edge in planned path but not on it yet)
        approaching = 0
        # This will be set by the environment via set_traversal_context()

        # Node occupancy at endpoints
        from_node_occ = len(getattr(from_node, 'current_robot_ids', []))
        to_node_occ = len(getattr(to_node, 'current_robot_ids', []))

        self._current_traversal = EdgeTraversalRecord(
            edge_index=self.current_edge_index,
            robot_id=self.robot_id,
            entry_time=0.0,  # Set by environment via set_traversal_entry_time()
            distance_m=edge.distance_m,
            corridor_width=edge.corridor_width,
            max_v_ms=edge.max_v_ms,
            num_robots_on_edge=len(edge.active_robot_ids),
            same_direction_count=same_dir,
            opposite_direction_count=opposite_dir,
            people_count=edge.people_count,
            clutter_level=edge.clutter_level,
            approaching_robot_count=approaching,
            from_node_occupancy=from_node_occ,
            to_node_occupancy=to_node_occ,
        )

    def update(self, time_delta: float) -> RobotTelemetry:
        """
        Simulate robot movement for time_delta seconds.
        Returns updated telemetry.

        Args:
            time_delta: Time step in seconds

        Returns:
            RobotTelemetry object with current state
        """
        # Handle charging
        if self.is_charging:
            self.battery_level = min(1.0, self.battery_level + self.charging_rate * time_delta)
            if self.battery_level >= 1.0:
                self.stop_charging()
            return self.get_telemetry()

        if not self.path_queue and not self.current_target_node:
            # Robot is idle
            self.velocity_ms = 0.0
            return self.get_telemetry()

        if self.current_edge_index is not None:
            # Robot is on an edge
            edge = self.graph_state.edges[self.current_edge_index]
            num_active = max(1, len(edge.active_robot_ids))
            corridor_capacity = 2
            if num_active > corridor_capacity:
                self.velocity_ms = self.max_velocity_ms * max(0.2, corridor_capacity / num_active)
            else:
                self.velocity_ms = self.max_velocity_ms
            distance_traveled = self.velocity_ms * time_delta

            # Update progress
            progress_delta = distance_traveled / edge.distance_m
            proposed_progress = self.edge_progress + progress_delta
            # Enforce no overlap: maintain headway on the same edge/direction
            headway_m = 0.5
            direction_from = self.current_node_index
            direction_to = self.current_target_node
            max_progress = proposed_progress
            for robot_id, (progress, from_idx, to_idx) in edge.active_robot_progress.items():
                if robot_id == self.robot_id:
                    continue
                if from_idx == direction_from and to_idx == direction_to and progress > self.edge_progress:
                    max_progress = min(max_progress, progress - (headway_m / edge.distance_m))

            if max_progress < self.edge_progress:
                max_progress = self.edge_progress

            self.edge_progress = min(max_progress, 1.0)

            # Track velocity and stops for edge cost model
            if self._current_traversal is not None:
                self._current_traversal.velocity_samples.append(self.velocity_ms)
                # Detect stop: was moving, now stopped
                if self.prev_velocity_ms > 0.05 and self.velocity_ms < 0.05:
                    self._current_traversal.stop_count += 1
            self.prev_velocity_ms = self.velocity_ms

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
        # Finalize traversal record before clearing edge state
        if self._current_traversal is not None:
            # exit_time is set by environment via finalize_traversal()
            self.traversal_records.append(self._current_traversal)
            self._current_traversal = None

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

    def start_charging(self):
        """Start charging the robot at a charging dock."""
        self.is_charging = True
        self.velocity_ms = 0.0

    def stop_charging(self):
        """Stop charging the robot."""
        self.is_charging = False

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
            self.current_capacity == 0 and
            not self.is_charging
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
            eta_to_next_node=eta,
            is_charging=self.is_charging
        )

    def set_traversal_entry_time(self, time: float):
        """Called by environment to stamp entry time on current traversal."""
        if self._current_traversal is not None:
            self._current_traversal.entry_time = time
            self._current_traversal.time_of_day = (time % 86400.0) / 86400.0

    def set_traversal_approaching_count(self, count: int):
        """Called by environment to set approaching robot count (needs global view)."""
        if self._current_traversal is not None:
            self._current_traversal.approaching_robot_count = count

    def finalize_current_traversal(self, exit_time: float):
        """Called by environment to stamp exit time on most recent completed traversal."""
        if self.traversal_records:
            latest = self.traversal_records[-1]
            if latest.exit_time == 0.0:
                latest.exit_time = exit_time

    def get_and_clear_traversal_records(self) -> List[EdgeTraversalRecord]:
        """Retrieve all completed traversal records and clear buffer."""
        records = self.traversal_records
        self.traversal_records = []
        return records

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
