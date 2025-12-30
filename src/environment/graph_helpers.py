"""
Graph-aware helper functions for computing derived metrics.
These helpers compute travel time, path congestion, and graph statistics.
"""
import numpy as np
from typing import List, Tuple, Optional
import heapq


def dijkstra_shortest_path(
    start_node_idx: int,
    goal_node_idx: int,
    graph_state,
    num_nodes: int = 10
) -> Tuple[List[int], float]:
    """
    Find shortest path using Dijkstra's algorithm with dynamic edge weights.

    Args:
        start_node_idx: Starting node index
        goal_node_idx: Goal node index
        graph_state: GraphState object with nodes and edges
        num_nodes: Total number of nodes in graph

    Returns:
        (path, cost) where path is list of node indices, cost is total weight
    """
    if start_node_idx == goal_node_idx:
        return [start_node_idx], 0.0

    # Build adjacency list with dynamic weights
    adjacency = {i: [] for i in range(num_nodes)}
    for edge in graph_state.edges:
        # Find node indices from node_id
        from_idx = None
        to_idx = None
        for idx, node in enumerate(graph_state.nodes):
            if node.node_id == edge.from_node:
                from_idx = idx
            if node.node_id == edge.to_node:
                to_idx = idx

        if from_idx is not None and to_idx is not None:
            # Use current_weight (includes congestion)
            adjacency[from_idx].append((to_idx, edge.current_weight))
            # Bidirectional
            adjacency[to_idx].append((from_idx, edge.current_weight))

    # Dijkstra's algorithm
    distances = {i: float('inf') for i in range(num_nodes)}
    distances[start_node_idx] = 0.0
    previous = {i: None for i in range(num_nodes)}

    pq = [(0.0, start_node_idx)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_node == goal_node_idx:
            break

        if current_dist > distances[current_node]:
            continue

        for neighbor, weight in adjacency[current_node]:
            distance = current_dist + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    # Reconstruct path
    if distances[goal_node_idx] == float('inf'):
        # No path found, return direct estimate
        from_node = graph_state.nodes[start_node_idx]
        to_node = graph_state.nodes[goal_node_idx]
        euclidean_dist = np.sqrt((from_node.x - to_node.x)**2 + (from_node.y - to_node.y)**2)
        return [start_node_idx, goal_node_idx], euclidean_dist

    path = []
    current = goal_node_idx
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()

    return path, distances[goal_node_idx]


def estimate_travel_time(robot, task, graph_state) -> float:
    """
    Estimate travel time from robot's current location to task start.
    Uses graph structure + dynamic congestion weights.

    Args:
        robot: RobotState object
        task: Task object
        graph_state: GraphState object

    Returns:
        Estimated travel time in seconds
    """
    path, cost = dijkstra_shortest_path(
        robot.current_node_index,
        task.from_location_index,
        graph_state
    )

    return cost


def estimate_path_congestion(robot, task, graph_state) -> float:
    """
    Estimate average congestion along robot's path to task.

    Args:
        robot: RobotState object
        task: Task object
        graph_state: GraphState object

    Returns:
        Average congestion score (0.0 - 1.0+)
    """
    path, _ = dijkstra_shortest_path(
        robot.current_node_index,
        task.from_location_index,
        graph_state
    )

    if len(path) <= 1:
        return 0.0

    congestion_scores = []

    # Find edges along path
    for i in range(len(path) - 1):
        from_idx = path[i]
        to_idx = path[i + 1]

        from_node_id = graph_state.nodes[from_idx].node_id
        to_node_id = graph_state.nodes[to_idx].node_id

        # Find corresponding edge
        for edge in graph_state.edges:
            if ((edge.from_node == from_node_id and edge.to_node == to_node_id) or
                (edge.to_node == from_node_id and edge.from_node == to_node_id)):

                # Compute congestion: clutter + active robots
                edge_congestion = (
                    edge.clutter_level +
                    len(edge.active_robot_ids) * 0.2
                )
                congestion_scores.append(edge_congestion)
                break

    if not congestion_scores:
        return 0.0

    return np.mean(congestion_scores)


def compute_graph_distance(robot, task, graph_state) -> float:
    """
    Compute graph distance (number of hops) from robot to task.

    Args:
        robot: RobotState object
        task: Task object
        graph_state: GraphState object

    Returns:
        Number of edges in shortest path
    """
    path, _ = dijkstra_shortest_path(
        robot.current_node_index,
        task.from_location_index,
        graph_state
    )

    return float(len(path) - 1)  # Number of edges = nodes - 1


def compute_graph_metrics(graph_state, current_task) -> np.ndarray:
    """
    Compute aggregated graph statistics.

    Args:
        graph_state: GraphState object
        current_task: Task object (for task-specific local metrics)

    Returns:
        Array of 6 features: [mean_congestion, max_congestion,
                             start_cluttered, end_cluttered,
                             start_congestion, end_congestion]
    """
    # Global metrics
    all_edge_weights = [e.current_weight for e in graph_state.edges]
    mean_congestion = np.mean(all_edge_weights) if all_edge_weights else 0.0
    max_congestion = np.max(all_edge_weights) if all_edge_weights else 0.0

    # Local metrics (around task locations)
    start_node = graph_state.get_node_by_index(current_task.from_location_index)
    end_node = graph_state.get_node_by_index(current_task.to_location_index)

    # Find edges connected to task start node
    start_edges = [
        e for e in graph_state.edges
        if e.from_node == start_node.node_id or e.to_node == start_node.node_id
    ]
    start_congestion = np.mean([e.current_weight for e in start_edges]) if start_edges else 0.0

    # Find edges connected to task end node
    end_edges = [
        e for e in graph_state.edges
        if e.from_node == end_node.node_id or e.to_node == end_node.node_id
    ]
    end_congestion = np.mean([e.current_weight for e in end_edges]) if end_edges else 0.0

    return np.array([
        mean_congestion,
        max_congestion,
        float(start_node.is_cluttered),
        float(end_node.is_cluttered),
        start_congestion,
        end_congestion
    ], dtype=np.float32)
