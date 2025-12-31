"""
Autoregressive state builder for task assignment.
Builds state for ONE task assignment decision at a time.
"""
import numpy as np
from typing import List
from .graph_helpers import (
    estimate_travel_time,
    estimate_path_congestion,
    compute_graph_distance,
    compute_graph_metrics
)


def build_autoregressive_state(
    current_task,
    robots: List,
    graph_state,
    remaining_tasks: List,
    current_time: float
) -> np.ndarray:
    """
    Build state for ONE task assignment decision.

    State structure:
    - Task features (10)
    - Graph node features (50: 10 nodes × 5)
    - Graph edge features (60: 20 edges × 3)
    - Graph metrics (6: aggregated statistics)
    - Robot features (40: 5 robots × 8)
    - Queue features (5: summary stats)
    Total: 171 features

    Args:
        current_task: Task being assigned
        robots: List of RobotState objects
        graph_state: GraphState object
        remaining_tasks: List of tasks still in queue (after this one)
        current_time: Current simulation time

    Returns:
        numpy array of shape (171,)
    """
    # 1. TASK FEATURES (10 features)
    task_features = current_task.get_features(current_time)

    # 2. GRAPH NODE FEATURES (50 features: 10 nodes × 5)
    node_features = graph_state.get_node_features().flatten()

    # 3. GRAPH EDGE FEATURES (60 features: 20 edges × 3)
    edge_features = graph_state.get_edge_features().flatten()

    # 4. GRAPH METRICS (6 features: aggregated)
    graph_metrics = compute_graph_metrics(graph_state, current_task)

    # 5. ROBOT FEATURES (40 features: 5 robots × 8)
    robot_features = []
    for robot in robots:
        # Get robot's current position
        current_node = graph_state.get_node_by_index(robot.current_node_index)

        # Compute graph-aware metrics for this robot-task pair
        travel_time = estimate_travel_time(robot, current_task, graph_state)
        path_congestion = estimate_path_congestion(robot, current_task, graph_state)
        graph_distance = compute_graph_distance(robot, current_task, graph_state)

        # Robot features: [id, battery, available, x, y, num_queued, travel_time, congestion]
        robot_features.extend([
            float(robot.robot_id),
            robot.battery_level,
            float(robot.is_available),
            current_node.x,
            current_node.y,
            float(robot.num_queued_tasks),
            travel_time,
            path_congestion,
        ])

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

    # CONCATENATE ALL FEATURES
    state = np.concatenate([
        task_features,      # 10
        node_features,      # 50
        edge_features,      # 60
        graph_metrics,      # 6
        robot_features,     # 40
        queue_features,     # 5
    ])

    return state.astype(np.float32)


def get_state_dim() -> int:
    """
    Get the dimension of the autoregressive state.

    Returns:
        Integer dimension (171)
    """
    task_dim = 10
    node_dim = 10 * 5  # 10 nodes × 5 features
    edge_dim = 20 * 3  # 20 edges × 3 features
    graph_metrics_dim = 6
    robot_dim = 5 * 8  # 5 robots × 8 features
    queue_dim = 5

    return task_dim + node_dim + edge_dim + graph_metrics_dim + robot_dim + queue_dim
