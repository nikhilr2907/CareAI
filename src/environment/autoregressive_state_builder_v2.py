"""
Autoregressive state builder v2 with telemetry support.
Builds state for ONE task assignment decision using real-time telemetry data.
"""
import numpy as np
from typing import List
from dataclasses import dataclass
from .graph_helpers import (
    estimate_travel_time,
    estimate_path_congestion,
    compute_graph_distance,
    compute_graph_metrics
)


@dataclass
class AutoregressiveState:
    """Container for autoregressive state features."""
    task_features: np.ndarray  # 12 features
    node_features: np.ndarray  # 50 features (10 nodes × 5)
    edge_features: np.ndarray  # 60 features (20 edges × 3)
    graph_metrics: np.ndarray  # 6 features
    robot_features: np.ndarray  # 75 features (5 robots × 15)
    queue_features: np.ndarray  # 5 features

    def to_array(self) -> np.ndarray:
        """Concatenate all features into single array."""
        return np.concatenate([
            self.task_features,
            self.node_features,
            self.edge_features,
            self.graph_metrics,
            self.robot_features,
            self.queue_features
        ])

    @property
    def total_features(self) -> int:
        """Total number of features."""
        return len(self.to_array())


def build_robot_features_with_telemetry(
    robot,
    current_task,
    graph_state,
    tasks: List,
    current_time: float
) -> np.ndarray:
    """
    Build robot features using telemetry data.

    Features (15 per robot):
    - Basic: robot_id (1)
    - Telemetry position: x, y, velocity (3)
    - Telemetry state: battery, is_moving (2)
    - Graph position: current_node_idx, current_edge_idx, edge_progress, eta (4)
    - Task state: num_queued, capacity, remaining_capacity, is_available (4)
    - Current task context: dest_idx (if active), distance_to_dest, time_to_deadline (3) - REMOVED to be 15

    Actually 12 features to keep it cleaner.
    """
    telemetry = robot.telemetry

    if telemetry is None:
        # Fallback if no telemetry
        return np.zeros(12, dtype=np.float32)

    # ===== BASIC INFO =====
    features = [float(robot.robot_id)]

    # ===== TELEMETRY: Real-time Physical State =====
    features.extend([
        telemetry.x,
        telemetry.y,
        telemetry.velocity_ms,
        telemetry.battery_level,
        float(telemetry.is_moving),
    ])

    # ===== TELEMETRY: Graph Position =====
    features.extend([
        float(telemetry.current_node_index if telemetry.current_node_index is not None else -1),
        float(telemetry.current_edge_index if telemetry.current_edge_index is not None else -1),
        telemetry.edge_progress,
        telemetry.eta_to_next_node,
    ])

    # ===== TASK ASSIGNMENT STATE =====
    features.extend([
        float(robot.num_queued_tasks),
        float(telemetry.current_capacity),
        float(robot.remaining_capacity),
    ])

    return np.array(features, dtype=np.float32)


def build_autoregressive_state(
    tasks: List,
    robots: List,
    graph_state,
    current_time: float,
    current_task_index: int
) -> AutoregressiveState:
    """
    Build state for ONE task assignment decision.

    State structure:
    - Task features (12): from_idx, to_idx, duration, age, time_to_deadline,
                         queue_pos, num_items, time_to_stockout, type_flags(4)
    - Graph node features (50): 10 nodes × 5
    - Graph edge features (60): 20 edges × 3
    - Graph metrics (6): aggregated statistics
    - Robot features (60): 5 robots × 12
    - Queue features (5): summary stats
    Total: 193 features

    Args:
        tasks: List of all tasks
        robots: List of RobotState objects with telemetry
        graph_state: GraphState object
        current_time: Current simulation time
        current_task_index: Index of task being assigned

    Returns:
        AutoregressiveState object
    """
    # Handle case where no tasks remain
    if current_task_index >= len(tasks):
        # Return zero state
        return AutoregressiveState(
            task_features=np.zeros(12, dtype=np.float32),
            node_features=np.zeros(50, dtype=np.float32),
            edge_features=np.zeros(60, dtype=np.float32),
            graph_metrics=np.zeros(6, dtype=np.float32),
            robot_features=np.zeros(60, dtype=np.float32),
            queue_features=np.zeros(5, dtype=np.float32)
        )

    current_task = tasks[current_task_index]
    remaining_tasks = tasks[current_task_index + 1:]

    # 1. TASK FEATURES (12 features)
    task_features = current_task.get_features(current_time)

    # 2. GRAPH NODE FEATURES (50 features: 10 nodes × 5)
    node_features = graph_state.get_node_features().flatten()

    # 3. GRAPH EDGE FEATURES (60 features: 20 edges × 3)
    edge_features = graph_state.get_edge_features().flatten()

    # 4. GRAPH METRICS (6 features: aggregated)
    graph_metrics = compute_graph_metrics(graph_state, current_task)

    # 5. ROBOT FEATURES (60 features: 5 robots × 12)
    robot_features = []
    for robot in robots:
        robot_feat = build_robot_features_with_telemetry(
            robot,
            current_task,
            graph_state,
            tasks,
            current_time
        )
        robot_features.extend(robot_feat)

    robot_features = np.array(robot_features, dtype=np.float32)

    # 6. QUEUE FEATURES (5 features: summary)
    if remaining_tasks:
        num_remaining = len(remaining_tasks)
        avg_priority = np.mean([t.manual_priority for t in remaining_tasks])
        num_urgent = sum(1 for t in remaining_tasks if t.manual_priority >= 4)
        oldest_age = max([t.get_age(current_time) for t in remaining_tasks])
        num_near_deadline = sum(1 for t in remaining_tasks if t.get_time_to_deadline(current_time) < 300)

        queue_features = np.array([
            float(num_remaining),
            avg_priority,
            float(num_urgent),
            oldest_age,
            float(num_near_deadline)
        ], dtype=np.float32)
    else:
        queue_features = np.zeros(5, dtype=np.float32)

    return AutoregressiveState(
        task_features=task_features,
        node_features=node_features,
        edge_features=edge_features,
        graph_metrics=graph_metrics,
        robot_features=robot_features,
        queue_features=queue_features
    )


def get_state_dim() -> int:
    """Get total state dimension."""
    # Task(12) + Nodes(50) + Edges(60) + Metrics(6) + Robots(60) + Queue(5)
    return 193
