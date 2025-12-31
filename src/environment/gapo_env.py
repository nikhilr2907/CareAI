"""
GAPO-compatible environment for hospital robot task allocation.

Returns graph-structured states (dict) instead of flat vectors.
Compatible with GNN-based GAPO policy network.
"""
import gym
from gym import spaces
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

from .graph.graph_state import GraphState
from .robot.robot_state import RobotState, create_default_robot
from .robot.robot_simulator import RobotSimulator
from .tasks.task_state import Task, TaskQueue, rank_tasks
from .tasks.task_generator import (
    generate_inventory_tasks,
    generate_random_ad_hoc_tasks,
    update_inventory_levels
)
from .graph_helpers import dijkstra_shortest_path


class GAPOTaskAssignmentEnv(gym.Env):
    """
    GAPO-compatible environment with graph-structured states.

    Key differences from V2:
    - Returns state as dictionary (not flat vector)
    - Includes graph connectivity (edge_index)
    - Provides robot positions explicitly
    - Compatible with GNN encoders
    """

    def __init__(
        self,
        num_robots=5,
        num_nodes=10,
        max_episode_time=28800.0,
        timestep_seconds=1.0
    ):
        super(GAPOTaskAssignmentEnv, self).__init__()

        self.num_robots = num_robots
        self.num_nodes = num_nodes
        self.max_episode_time = max_episode_time
        self.timestep_seconds = timestep_seconds

        # Action space
        self.action_space = spaces.Discrete(num_robots + 1)
        self.HOLD_ACTION = num_robots

        # Observation space (dict-based)
        self.observation_space = spaces.Dict({
            'task_features': spaces.Box(-np.inf, np.inf, (12,), np.float32),
            'node_features': spaces.Box(-np.inf, np.inf, (num_nodes, 5), np.float32),
            'edge_features': spaces.Box(-np.inf, np.inf, (20, 3), np.float32),
            'robot_features': spaces.Box(-np.inf, np.inf, (num_robots, 12), np.float32),
            'robot_positions': spaces.Box(-np.inf, np.inf, (num_robots, 2), np.float32),
            'queue_features': spaces.Box(-np.inf, np.inf, (5,), np.float32),
        })

        # Environment components
        self.graph_state = None
        self.robots = []
        self.robot_simulators = []
        self.tasks = []
        self.next_task_id = 0

        # Time tracking
        self.current_time = 0.0
        self.last_inventory_check = 0.0
        self.inventory_check_interval = 300.0

        # Episode metrics
        self.episode_assignments = []
        self.episode_stockouts = 0
        self.cumulative_stockout_penalty = 0.0

        # Current state
        self.current_task_index = 0

    def reset(self):
        """Reset environment and return initial state dict."""
        # Initialize graph
        self.graph_state = GraphState()

        # Initialize robots
        self.robots = []
        self.robot_simulators = []

        for i in range(self.num_robots):
            initial_node = np.random.randint(0, self.num_nodes)

            simulator = RobotSimulator(i, self.graph_state, initial_node)
            self.robot_simulators.append(simulator)

            robot_state = create_default_robot(i)
            telemetry = simulator.get_telemetry()
            telemetry.timestamp = self.current_time
            robot_state.update_telemetry(telemetry)
            self.robots.append(robot_state)

        # Generate initial tasks
        self.tasks = []
        self.next_task_id = 0
        self.current_time = 0.0
        self.last_inventory_check = 0.0

        new_tasks, self.next_task_id = generate_inventory_tasks(
            self.graph_state, self.current_time, self.next_task_id
        )
        self.tasks.extend(new_tasks)

        ad_hoc_tasks, self.next_task_id = generate_random_ad_hoc_tasks(
            self.graph_state, self.current_time, 3, self.next_task_id
        )
        self.tasks.extend(ad_hoc_tasks)

        # Rank tasks
        task_queue = TaskQueue(tasks=self.tasks)
        ranked = rank_tasks(task_queue, self.current_time)
        self.tasks = ranked

        # Reset metrics
        self.episode_assignments = []
        self.episode_stockouts = 0
        self.cumulative_stockout_penalty = 0.0
        self.current_task_index = 0

        return self._get_state_dict()

    def step(self, action: int):
        """Execute action and return next state dict."""
        reward = 0.0
        assignment_made = False

        # Make assignment decision
        if self.current_task_index < len(self.tasks):
            current_task = self.tasks[self.current_task_index]

            if action != self.HOLD_ACTION:
                available_robots = self._get_available_robots()

                if action < len(available_robots):
                    robot = available_robots[action]
                    assignment_made = True

                    reward += self._compute_assignment_reward(robot, current_task)
                    self._assign_task_to_robot(robot, current_task)

                    self.current_task_index += 1
                else:
                    reward -= 10.0  # Invalid action
            else:
                reward += self._compute_hold_reward(current_task)
                self.current_task_index += 1

        # Advance simulation
        time_delta = self.timestep_seconds
        self.current_time += time_delta
        time_delta_hours = time_delta / 3600.0

        self._update_robot_telemetry(time_delta)
        update_inventory_levels(self.graph_state, time_delta_hours)

        stockout_penalty = self._check_stockouts()
        reward += stockout_penalty

        self._update_edge_congestion()

        # Generate new tasks
        if self.current_time - self.last_inventory_check > self.inventory_check_interval:
            new_tasks, self.next_task_id = generate_inventory_tasks(
                self.graph_state, self.current_time, self.next_task_id
            )

            if new_tasks:
                self.tasks.extend(new_tasks)
                task_queue = TaskQueue(tasks=self.tasks)
                ranked = rank_tasks(task_queue, self.current_time)
                self.tasks = ranked

            self.last_inventory_check = self.current_time

        # Check termination
        done = (
            self.current_task_index >= len(self.tasks) or
            self.current_time >= self.max_episode_time
        )

        info = {
            'current_time': self.current_time,
            'tasks_assigned': len(self.episode_assignments),
            'tasks_remaining': len(self.tasks) - self.current_task_index,
            'stockouts': self.episode_stockouts,
            'assignment_made': assignment_made
        }

        return self._get_state_dict(), reward, done, info

    def _get_state_dict(self) -> Dict[str, np.ndarray]:
        """
        Build graph-structured state dictionary for GAPO.

        Returns dict with:
        - task_features: [12]
        - node_features: [num_nodes, 5]
        - edge_features: [num_edges, 3]
        - edge_index: [2, num_edges] - connectivity
        - robot_features: [num_robots, 12]
        - robot_positions: [num_robots, 2]
        - queue_features: [5]
        """
        # Task features
        if self.current_task_index < len(self.tasks):
            current_task = self.tasks[self.current_task_index]
            task_features = current_task.get_features(self.current_time)
        else:
            task_features = np.zeros(12, dtype=np.float32)

        # Node features
        node_features = self.graph_state.get_node_features()  # [num_nodes, 5]

        # Edge features
        edge_features = self.graph_state.get_edge_features()  # [num_edges, 3]

        # Edge index (connectivity)
        edge_index = self._build_edge_index()  # [2, num_edges]

        # Robot features
        robot_features = []
        robot_positions = []

        for robot in self.robots:
            if robot.telemetry:
                feat = [
                    float(robot.robot_id),
                    robot.telemetry.x,
                    robot.telemetry.y,
                    robot.telemetry.velocity_ms,
                    robot.telemetry.battery_level,
                    float(robot.telemetry.is_moving),
                    float(robot.telemetry.current_node_index if robot.telemetry.current_node_index is not None else -1),
                    float(robot.telemetry.current_edge_index if robot.telemetry.current_edge_index is not None else -1),
                    robot.telemetry.edge_progress,
                    robot.telemetry.eta_to_next_node,
                    float(robot.num_queued_tasks),
                    float(robot.telemetry.current_capacity),
                ]
                robot_features.append(feat)
                robot_positions.append([robot.telemetry.x, robot.telemetry.y])
            else:
                robot_features.append(np.zeros(12))
                robot_positions.append([0.0, 0.0])

        robot_features = np.array(robot_features, dtype=np.float32)
        robot_positions = np.array(robot_positions, dtype=np.float32)

        # Queue features
        remaining_tasks = self.tasks[self.current_task_index + 1:]
        if remaining_tasks:
            num_remaining = len(remaining_tasks)
            avg_priority = np.mean([t.manual_priority for t in remaining_tasks])
            num_urgent = sum(1 for t in remaining_tasks if t.manual_priority >= 4)
            oldest_age = max([t.get_age(self.current_time) for t in remaining_tasks])
            num_near_deadline = sum(1 for t in remaining_tasks if t.get_time_to_deadline(self.current_time) < 300)

            queue_features = np.array([
                float(num_remaining),
                avg_priority,
                float(num_urgent),
                oldest_age,
                float(num_near_deadline)
            ], dtype=np.float32)
        else:
            queue_features = np.zeros(5, dtype=np.float32)

        return {
            'task_features': task_features,
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_index': edge_index,
            'robot_features': robot_features,
            'robot_positions': robot_positions,
            'queue_features': queue_features
        }

    def _build_edge_index(self) -> np.ndarray:
        """
        Build edge connectivity matrix from graph edges.

        Returns:
            edge_index: [2, num_edges] - [source_nodes, target_nodes]
        """
        edge_index = []

        for edge in self.graph_state.edges:
            # Find node indices
            from_idx = self._find_node_index(edge.from_node)
            to_idx = self._find_node_index(edge.to_node)

            if from_idx is not None and to_idx is not None:
                edge_index.append([from_idx, to_idx])
                edge_index.append([to_idx, from_idx])  # Undirected

        if edge_index:
            return np.array(edge_index, dtype=np.int64).T
        else:
            # Empty graph - create self-loops
            return np.array([[i, i] for i in range(self.num_nodes)], dtype=np.int64).T

    def _find_node_index(self, node_id: str) -> Optional[int]:
        """Find node index by ID."""
        for i, node in enumerate(self.graph_state.nodes):
            if node.node_id == node_id:
                return i
        return None

    def get_action_mask(self) -> np.ndarray:
        """Get boolean mask for valid actions."""
        mask = np.ones(self.num_robots + 1, dtype=bool)

        if self.current_task_index >= len(self.tasks):
            mask[:self.num_robots] = False
            return mask

        current_task = self.tasks[self.current_task_index]

        for i, robot in enumerate(self.robots):
            if not robot.is_available:
                mask[i] = False
                continue

            if robot.current_capacity + current_task.num_items > robot.max_capacity:
                mask[i] = False
                continue

            if robot.current_node_index is not None:
                _, distance = dijkstra_shortest_path(
                    robot.current_node_index,
                    current_task.to_location_index,
                    self.graph_state,
                    self.num_nodes
                )
                battery_needed = distance * 0.001
                if robot.battery_level < battery_needed:
                    mask[i] = False

        return mask

    # ===== HELPER METHODS (same as V2) =====
    def _assign_task_to_robot(self, robot, task):
        """Assign task to robot."""
        simulator = self.robot_simulators[robot.robot_id]

        start_node = robot.current_node_index
        if start_node is None:
            start_node = self._find_nearest_node(robot.telemetry.x, robot.telemetry.y)

        path, distance = dijkstra_shortest_path(
            start_node, task.to_location_index, self.graph_state, self.num_nodes
        )

        simulator.set_path(path, task.task_id, task.num_items)

        robot.queued_tasks.append(task.task_id)
        robot.planned_path = path
        robot.target_node_index = task.to_location_index
        robot.travel_start_time = self.current_time

        task.is_assigned = True
        task.assigned_robot_id = robot.robot_id

        self.episode_assignments.append({
            'time': self.current_time,
            'robot_id': robot.robot_id,
            'task_id': task.task_id
        })

    def _update_robot_telemetry(self, time_delta):
        """Update robot simulators."""
        for simulator, robot_state in zip(self.robot_simulators, self.robots):
            telemetry = simulator.update(time_delta)
            telemetry.timestamp = self.current_time
            robot_state.update_telemetry(telemetry)

            if telemetry.is_at_node and robot_state.queued_tasks:
                task = self.tasks[robot_state.queued_tasks[0]]

                if telemetry.current_node_index == task.to_location_index:
                    dest_node = self.graph_state.nodes[task.to_location_index]
                    dest_node.restock(task.num_items)

                    robot_state.queued_tasks.pop(0)
                    simulator.complete_task(task.num_items)

    def _update_edge_congestion(self):
        """Update edge congestion."""
        for edge in self.graph_state.edges:
            edge.active_robot_ids.clear()

        for robot in self.robots:
            if robot.telemetry and robot.telemetry.is_on_edge:
                edge_idx = robot.telemetry.current_edge_index
                if edge_idx is not None and edge_idx < len(self.graph_state.edges):
                    self.graph_state.edges[edge_idx].active_robot_ids.append(robot.robot_id)

    def _check_stockouts(self):
        """Check stockouts and return penalty."""
        penalty = 0.0

        for node in self.graph_state.nodes:
            if node.consumption_rate > 0:
                if node.is_stockout:
                    penalty -= 100.0
                    self.episode_stockouts += 1
                elif node.time_to_stockout < 0.5:
                    penalty -= 20.0 * (0.5 - node.time_to_stockout)

        self.cumulative_stockout_penalty += penalty
        return penalty

    def _compute_assignment_reward(self, robot, task):
        """Compute assignment reward."""
        reward = 5.0

        start_node = robot.current_node_index or 0
        path, distance = dijkstra_shortest_path(
            start_node, task.to_location_index, self.graph_state, self.num_nodes
        )
        reward -= 0.1 * distance

        if robot.battery_level < 0.3:
            reward -= 3.0

        if robot.current_capacity + task.num_items <= robot.max_capacity:
            capacity_ratio = (robot.current_capacity + task.num_items) / robot.max_capacity
            reward += 2.0 * capacity_ratio
        else:
            reward -= 20.0

        reward += task.manual_priority * 0.5

        if task.time_to_stockout < 1.0:
            reward += 5.0 * (1.0 - task.time_to_stockout)

        return reward

    def _compute_hold_reward(self, task):
        """Compute HOLD reward."""
        reward = -1.0

        available_robots = self._get_available_robots()
        if len(available_robots) == 0:
            reward += 2.0

        if task.manual_priority >= 4:
            reward -= 5.0

        return reward

    def _get_available_robots(self):
        """Get available robots."""
        return [r for r in self.robots if r.is_available]

    def _find_nearest_node(self, x, y):
        """Find nearest node."""
        min_dist = float('inf')
        nearest_idx = 0

        for i, node in enumerate(self.graph_state.nodes):
            dist = np.sqrt((node.x - x)**2 + (node.y - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx
