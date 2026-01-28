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
    num_nodes: int = 10,
    edge_cost_manager=None,
    all_robots=None,
) -> Tuple[List[int], float]:
    """
    Find shortest path using Dijkstra's algorithm.

    Uses learned edge costs when edge_cost_manager is provided and ready,
    otherwise falls back to heuristic current_weight on each edge.

    Args:
        start_node_idx: Starting node index
        goal_node_idx: Goal node index
        graph_state: GraphState object with nodes and edges
        num_nodes: Total number of nodes in graph
        edge_cost_manager: Optional EdgeCostManager for learned costs
        all_robots: Optional list of robots (for approaching count in learned model)

    Returns:
        (path, cost) where path is list of node indices, cost is total weight
    """
    if start_node_idx == goal_node_idx:
        return [start_node_idx], 0.0

    # Precompute learned edge costs if model is available
    learned_costs = None
    if edge_cost_manager is not None:
        features = edge_cost_manager.build_edge_features(graph_state, all_robots)
        learned_costs = edge_cost_manager.predict_edge_costs(features)
        # learned_costs: [num_edges] — one per edge in graph_state.edges

    # Build adjacency list
    adjacency = {i: [] for i in range(num_nodes)}
    for edge_idx, edge in enumerate(graph_state.edges):
        # Find node indices from node_id
        from_idx = None
        to_idx = None
        for idx, node in enumerate(graph_state.nodes):
            if node.node_id == edge.from_node:
                from_idx = idx
            if node.node_id == edge.to_node:
                to_idx = idx

        if from_idx is not None and to_idx is not None:
            if learned_costs is not None:
                weight = float(learned_costs[edge_idx])
            else:
                weight = edge.current_weight

            adjacency[from_idx].append((to_idx, weight))
            # Bidirectional
            adjacency[to_idx].append((from_idx, weight))

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


# ==============================================================================
# REGION-BASED POSITION MAPPING
# These functions map continuous (x, y) positions to graph nodes/edges
# ==============================================================================

def get_current_node_index(robot_x: float, robot_y: float, graph_state) -> Optional[int]:
    """
    Determine which node region contains the robot.

    Args:
        robot_x, robot_y: Robot's continuous position
        graph_state: GraphState object

    Returns:
        Node index if robot is inside a node region, None if in corridor
    """
    for idx, node in enumerate(graph_state.nodes):
        if node.contains_point(robot_x, robot_y):
            return idx
    return None  # Robot is in corridor (between regions)


def get_current_edge_index(robot_x: float, robot_y: float, graph_state, tolerance: float = 1.0) -> Optional[int]:
    """
    Find which edge (corridor) the robot is on (if not in a node).

    Args:
        robot_x, robot_y: Robot's continuous position
        graph_state: GraphState object
        tolerance: Max perpendicular distance from corridor centerline (meters)

    Returns:
        Edge index if robot is on an edge, None if in unmapped space
    """
    for idx, edge in enumerate(graph_state.edges):
        if edge.is_point_on_corridor(robot_x, robot_y, tolerance):
            return idx
    return None  # Robot is in unmapped space


def compute_edge_progress(robot_x: float, robot_y: float, graph_state, edge_index: int) -> float:
    """
    Calculate how far along an edge the robot is (0.0 = entry, 1.0 = exit).

    Args:
        robot_x, robot_y: Robot's continuous position
        graph_state: GraphState object
        edge_index: Index of the edge

    Returns:
        Progress value between 0.0 and 1.0
    """
    if edge_index is None or edge_index < 0 or edge_index >= len(graph_state.edges):
        return 0.0

    edge = graph_state.edges[edge_index]
    return edge.compute_progress(robot_x, robot_y)


def update_robot_graph_position(robot, graph_state):
    """
    Update robot's graph-aware position (current_node_index, current_edge_index, edge_progress)
    based on continuous (x, y) position from telemetry.

    This function should be called after robot telemetry is updated.

    Args:
        robot: RobotState object with telemetry
        graph_state: GraphState object

    Modifies:
        robot.telemetry.current_node_index
        robot.telemetry.current_edge_index
        robot.telemetry.edge_progress
    """
    if robot.telemetry is None:
        return

    robot_x = robot.telemetry.x
    robot_y = robot.telemetry.y

    # First check if robot is in a node region
    node_idx = get_current_node_index(robot_x, robot_y, graph_state)

    if node_idx is not None:
        # Robot is in a node region
        robot.telemetry.current_node_index = node_idx
        robot.telemetry.current_edge_index = None
        robot.telemetry.edge_progress = 0.0
    else:
        # Robot is in corridor - find which edge
        edge_idx = get_current_edge_index(robot_x, robot_y, graph_state)

        if edge_idx is not None:
            # Robot is on an edge
            robot.telemetry.current_node_index = None
            robot.telemetry.current_edge_index = edge_idx
            robot.telemetry.edge_progress = compute_edge_progress(robot_x, robot_y, graph_state, edge_idx)
        else:
            # Robot is in unmapped space - keep previous values
            pass


def estimate_travel_distance(robot, target_node_idx: int, graph_state) -> float:
    """
    Estimate distance robot must travel to reach target node.
    Accounts for current position on graph (node or edge).

    Args:
        robot: RobotState object with telemetry
        target_node_idx: Target node index
        graph_state: GraphState object

    Returns:
        Estimated travel distance in meters
    """
    if robot.current_node_index is not None:
        # Robot is in a node - use graph shortest path
        path, distance = dijkstra_shortest_path(
            robot.current_node_index,
            target_node_idx,
            graph_state
        )
        return distance

    elif robot.telemetry and robot.telemetry.current_edge_index is not None:
        # Robot is on edge - compute remaining distance on current edge + path from edge end
        edge_idx = robot.telemetry.current_edge_index
        edge = graph_state.edges[edge_idx]

        # Remaining distance on current edge
        remaining_on_edge = (1.0 - robot.telemetry.edge_progress) * edge.distance_m

        # Find which node this edge leads to (closer to target)
        from_node_idx = get_node_index_by_id(edge.from_node, graph_state)
        to_node_idx = get_node_index_by_id(edge.to_node, graph_state)

        # Get distances from both ends of edge to target
        _, dist_from_end = dijkstra_shortest_path(to_node_idx, target_node_idx, graph_state)
        _, dist_from_start = dijkstra_shortest_path(from_node_idx, target_node_idx, graph_state)

        # Choose the direction that gets closer to target
        if dist_from_end < dist_from_start:
            # Continue forward on edge
            return remaining_on_edge + dist_from_end
        else:
            # Turn back
            distance_back = robot.telemetry.edge_progress * edge.distance_m
            return distance_back + dist_from_start

    else:
        # Robot position unknown - use Euclidean as fallback
        robot_x, robot_y = robot.current_position
        target_node = graph_state.nodes[target_node_idx]
        return np.sqrt((target_node.center_x - robot_x)**2 + (target_node.center_y - robot_y)**2)


def get_node_index_by_id(node_id: str, graph_state) -> Optional[int]:
    """
    Get node index from node ID.

    Args:
        node_id: Node ID string
        graph_state: GraphState object

    Returns:
        Node index or None if not found
    """
    for idx, node in enumerate(graph_state.nodes):
        if node.node_id == node_id:
            return idx
    return None


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
