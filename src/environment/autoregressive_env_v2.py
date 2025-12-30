"""
Autoregressive environment v2 with continuous time simulation and telemetry.
Major improvements:
- Robot telemetry-based position tracking
- Continuous time simulation with inventory dynamics
- Stockout penalties
- Action masking for capacity/battery constraints
- Inventory-driven task generation
"""
import gym
from gym import spaces
import numpy as np
from typing import List, Tuple, Dict, Optional

from .graph.graph_state import GraphState
from .robot.robot_state import RobotState, create_default_robot
from .robot.robot_simulator import RobotSimulator
from .robot.robot_telemetry import RobotTelemetry
from .tasks.task_state import Task, TaskQueue, rank_tasks
from .tasks.task_generator import (
    generate_inventory_tasks,
    generate_random_ad_hoc_tasks,
    update_inventory_levels
)
from .graph_helpers import dijkstra_shortest_path
from .autoregressive_state_builder import build_autoregressive_state, get_state_dim


class AutoregressiveTaskAssignmentEnvV2(gym.Env):
    """
    Continuous-time autoregressive environment with telemetry integration.

    Key features:
    - Robots report position via telemetry (simulator or real hardware)
    - Inventory depletes over time, triggering new tasks
    - Action masking prevents invalid assignments
    - Stockout penalties in reward function
    """

    def __init__(
        self,
        num_robots=5,
        num_nodes=10,
        max_episode_time=28800.0,  # 8 hours in seconds
        timestep_seconds=1.0  # Simulation timestep
    ):
        super(AutoregressiveTaskAssignmentEnvV2, self).__init__()

        self.num_robots = num_robots
        self.num_nodes = num_nodes
        self.max_episode_time = max_episode_time
        self.timestep_seconds = timestep_seconds

        # Action space: [robot_0, robot_1, ..., robot_N, HOLD]
        self.action_space = spaces.Discrete(num_robots + 1)
        self.HOLD_ACTION = num_robots

        # Observation space
        state_dim = get_state_dim()  # Will need updating for new features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Initialize environment components
        self.graph_state = None
        self.robots = []
        self.robot_simulators = []
        self.tasks = []
        self.next_task_id = 0

        # Time tracking
        self.current_time = 0.0
        self.last_inventory_check = 0.0
        self.inventory_check_interval = 300.0  # Check every 5 minutes

        # Episode metrics
        self.episode_assignments = []
        self.episode_stockouts = 0
        self.cumulative_stockout_penalty = 0.0

        # Current decision state
        self.state = None
        self.current_task_index = 0

    def reset(self):
        """Reset environment for new episode."""
        # Initialize graph with inventory
        self.graph_state = GraphState()

        # Initialize robot simulators and states
        self.robots = []
        self.robot_simulators = []

        for i in range(self.num_robots):
            # Random starting node
            initial_node = np.random.randint(0, self.num_nodes)

            # Create simulator
            simulator = RobotSimulator(i, self.graph_state, initial_node)
            self.robot_simulators.append(simulator)

            # Create robot state
            robot_state = create_default_robot(i)
            # Get initial telemetry
            telemetry = simulator.get_telemetry()
            telemetry.timestamp = self.current_time
            robot_state.update_telemetry(telemetry)
            self.robots.append(robot_state)

        # Generate initial tasks
        self.tasks = []
        self.next_task_id = 0
        self.current_time = 0.0
        self.last_inventory_check = 0.0

        # Generate initial inventory-based tasks
        new_tasks, self.next_task_id = generate_inventory_tasks(
            self.graph_state,
            self.current_time,
            self.next_task_id
        )
        self.tasks.extend(new_tasks)

        # Add some random ad-hoc tasks
        ad_hoc_tasks, self.next_task_id = generate_random_ad_hoc_tasks(
            self.graph_state,
            self.current_time,
            num_tasks=3,
            next_task_id=self.next_task_id
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

        # Build initial state
        self.state = build_autoregressive_state(
            self.tasks,
            self.robots,
            self.graph_state,
            self.current_time,
            self.current_task_index
        )

        return self.state.to_array()

    def step(self, action: int):
        """
        Execute one step: assign task or HOLD, then advance simulation.

        Args:
            action: Robot index to assign task to, or HOLD_ACTION

        Returns:
            observation, reward, done, info
        """
        # ===== 1. MAKE ASSIGNMENT DECISION =====
        reward = 0.0
        assignment_made = False

        # Get current task
        if self.current_task_index < len(self.tasks):
            current_task = self.tasks[self.current_task_index]

            if action != self.HOLD_ACTION:
                # Assign task to robot
                available_robots = self._get_available_robots()

                if action < len(available_robots):
                    robot = available_robots[action]
                    assignment_made = True

                    # Compute assignment reward
                    reward += self._compute_assignment_reward(robot, current_task)

                    # Execute assignment
                    self._assign_task_to_robot(robot, current_task)

                    # Move to next task
                    self.current_task_index += 1
                else:
                    # Invalid action - negative reward
                    reward -= 10.0
            else:
                # HOLD action
                reward += self._compute_hold_reward(current_task)
                self.current_task_index += 1

        # ===== 2. ADVANCE SIMULATION TIME =====
        time_delta = self.timestep_seconds
        self.current_time += time_delta
        time_delta_hours = time_delta / 3600.0

        # Update robot telemetry
        self._update_robot_telemetry(time_delta)

        # Update inventory levels
        update_inventory_levels(self.graph_state, time_delta_hours)

        # Check for stockouts (penalties)
        stockout_penalty = self._check_stockouts()
        reward += stockout_penalty

        # Update edge congestion
        self._update_edge_congestion()

        # ===== 3. GENERATE NEW TASKS =====
        # Periodic inventory check
        if self.current_time - self.last_inventory_check > self.inventory_check_interval:
            new_tasks, self.next_task_id = generate_inventory_tasks(
                self.graph_state,
                self.current_time,
                self.next_task_id
            )

            if new_tasks:
                self.tasks.extend(new_tasks)
                # Re-rank all tasks
                task_queue = TaskQueue(tasks=self.tasks)
                ranked = rank_tasks(task_queue, self.current_time)
                self.tasks = ranked

            self.last_inventory_check = self.current_time

        # ===== 4. REBUILD STATE =====
        self.state = build_autoregressive_state(
            self.tasks,
            self.robots,
            self.graph_state,
            self.current_time,
            self.current_task_index
        )

        # ===== 5. CHECK TERMINATION =====
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

        return self.state.to_array(), reward, done, info

    def _assign_task_to_robot(self, robot: RobotState, task: Task):
        """Assign task and command robot simulator to execute."""
        simulator = self.robot_simulators[robot.robot_id]

        # Get robot's current node (from telemetry)
        start_node = robot.current_node_index
        if start_node is None:
            # Robot is on an edge - use nearest node
            start_node = self._find_nearest_node(robot.telemetry.x, robot.telemetry.y)

        # Plan path using Dijkstra
        path, distance = dijkstra_shortest_path(
            start_node,
            task.to_location_index,
            self.graph_state,
            self.num_nodes
        )

        # Command simulator to follow path
        simulator.set_path(path, task.task_id, task.num_items)

        # Update robot state
        robot.queued_tasks.append(task.task_id)
        robot.planned_path = path
        robot.target_node_index = task.to_location_index
        robot.travel_start_time = self.current_time

        # Update task
        task.is_assigned = True
        task.assigned_robot_id = robot.robot_id

        # Track assignment
        self.episode_assignments.append({
            'time': self.current_time,
            'robot_id': robot.robot_id,
            'task_id': task.task_id
        })

    def _update_robot_telemetry(self, time_delta: float):
        """Update all robot simulators and receive telemetry."""
        for simulator, robot_state in zip(self.robot_simulators, self.robots):
            # Simulate robot movement
            telemetry = simulator.update(time_delta)
            telemetry.timestamp = self.current_time

            # Update robot state with new telemetry
            robot_state.update_telemetry(telemetry)

            # Check for task completion
            if telemetry.is_at_node and robot_state.queued_tasks:
                task = self.tasks[robot_state.queued_tasks[0]]

                if telemetry.current_node_index == task.to_location_index:
                    # Task delivered - update inventory
                    dest_node = self.graph_state.nodes[task.to_location_index]
                    dest_node.restock(task.num_items)

                    # Remove task from queue
                    robot_state.queued_tasks.pop(0)

                    # Update simulator capacity
                    simulator.complete_task(task.num_items)

    def _update_edge_congestion(self):
        """Update edge congestion based on robot telemetry."""
        # Clear all edge congestion
        for edge in self.graph_state.edges:
            edge.active_robot_ids.clear()

        # Add robots currently on edges
        for robot in self.robots:
            if robot.telemetry and robot.telemetry.is_on_edge:
                edge_idx = robot.telemetry.current_edge_index
                if edge_idx is not None and edge_idx < len(self.graph_state.edges):
                    self.graph_state.edges[edge_idx].active_robot_ids.append(robot.robot_id)

    def _check_stockouts(self) -> float:
        """Check for stockouts and return penalty."""
        penalty = 0.0

        for node in self.graph_state.nodes:
            if node.consumption_rate > 0:  # Only check inventory nodes
                if node.is_stockout:
                    penalty -= 100.0  # Severe penalty
                    self.episode_stockouts += 1
                elif node.time_to_stockout < 0.5:  # < 30 minutes
                    # Penalty increases as stockout approaches
                    penalty -= 20.0 * (0.5 - node.time_to_stockout)

        self.cumulative_stockout_penalty += penalty
        return penalty

    def _compute_assignment_reward(self, robot: RobotState, task: Task) -> float:
        """Compute reward for assigning task to robot."""
        reward = 0.0

        # Base assignment reward
        reward += 5.0

        # Travel efficiency (negative penalty for long distances)
        start_node = robot.current_node_index or 0
        path, distance = dijkstra_shortest_path(
            start_node,
            task.to_location_index,
            self.graph_state,
            self.num_nodes
        )
        travel_penalty = -0.1 * distance
        reward += travel_penalty

        # Battery consideration
        if robot.battery_level < 0.3:
            reward -= 3.0  # Avoid using low battery robots

        # Capacity utilization bonus
        if robot.current_capacity + task.num_items <= robot.max_capacity:
            capacity_ratio = (robot.current_capacity + task.num_items) / robot.max_capacity
            reward += 2.0 * capacity_ratio
        else:
            reward -= 20.0  # Major penalty for exceeding capacity

        # Urgency bonus (higher priority tasks)
        reward += task.manual_priority * 0.5

        # Time-to-stockout bonus (critical tasks)
        if task.time_to_stockout < 1.0:  # < 1 hour
            reward += 5.0 * (1.0 - task.time_to_stockout)

        return reward

    def _compute_hold_reward(self, task: Task) -> float:
        """Compute reward for deferring task."""
        reward = -1.0  # Base penalty for deferring

        # Check if all robots busy
        available_robots = self._get_available_robots()
        if len(available_robots) == 0:
            reward += 2.0  # Bonus for holding when no robots available

        # Urgency penalty
        if task.manual_priority >= 4:
            reward -= 5.0  # Don't defer urgent tasks

        return reward

    def _get_available_robots(self) -> List[RobotState]:
        """Get robots that are available for new assignments."""
        return [r for r in self.robots if r.is_available]

    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for current state.
        Masks invalid actions (capacity/battery constraints).

        Returns:
            Boolean array where True = valid action
        """
        mask = np.ones(self.num_robots + 1, dtype=bool)  # +1 for HOLD

        if self.current_task_index >= len(self.tasks):
            # No tasks left - only HOLD valid
            mask[:self.num_robots] = False
            return mask

        current_task = self.tasks[self.current_task_index]
        available_robots = self._get_available_robots()

        # Mask unavailable robots
        for i in range(self.num_robots):
            robot = self.robots[i]

            if not robot.is_available:
                mask[i] = False
                continue

            # Check capacity constraint
            if robot.current_capacity + current_task.num_items > robot.max_capacity:
                mask[i] = False
                continue

            # Check battery constraint (rough estimate)
            if robot.current_node_index is not None:
                _, distance = dijkstra_shortest_path(
                    robot.current_node_index,
                    current_task.to_location_index,
                    self.graph_state,
                    self.num_nodes
                )
                # Assume 0.1% battery per meter
                battery_needed = distance * 0.001
                if robot.battery_level < battery_needed:
                    mask[i] = False

        return mask

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
