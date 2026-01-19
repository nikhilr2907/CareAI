"""
GAPO-compatible environment for hospital robot task allocation.

Returns graph-structured states (dict) instead of flat vectors.
Compatible with GNN-based GAPO policy network.
"""
# import gym
# from gym import spaces
import numpy as np
from typing import List, Dict, Optional

from .graph.graph_state import GraphState
from .graph.hospital_config import HospitalConfig
from .robot.robot_state import RobotState, create_default_robot
from .robot.robot_simulator import RobotSimulator
from .tasks.task_state import Task, TaskQueue, rank_tasks
from .tasks.task_generator import (
    generate_inventory_tasks,
    generate_random_ad_hoc_tasks,
    update_inventory_levels
)
from .graph_helpers import dijkstra_shortest_path


class GAPOTaskAssignmentEnv:#(gym.Env):
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
        timestep_seconds=1.0,
        hospital_config: Optional[HospitalConfig] = None
    ):
        super(GAPOTaskAssignmentEnv, self).__init__()

        self.num_robots = num_robots
        self.num_nodes = num_nodes
        self.max_episode_time = max_episode_time
        self.timestep_seconds = timestep_seconds
        self.hospital_config = hospital_config  # Optional custom config

        # Action space
        # self.action_space = spaces.Discrete(num_robots)

        # Observation space (dict-based)
        # self.observation_space = spaces.Dict({
        #     'task_features': spaces.Box(-np.inf, np.inf, (12,), np.float32),
        #     'node_features': spaces.Box(-np.inf, np.inf, (num_nodes, 5), np.float32),
        #     'edge_features': spaces.Box(-np.inf, np.inf, (20, 3), np.float32),
        #     'robot_features': spaces.Box(-np.inf, np.inf, (num_robots, 12), np.float32),
        #     'robot_positions': spaces.Box(-np.inf, np.inf, (num_robots, 2), np.float32),
        #     'queue_features': spaces.Box(-np.inf, np.inf, (5,), np.float32),
        # })

        # Environment components
        self.graph_state = None
        self.robots = []
        self.robot_simulators = []
        self.next_task_id = 0

        # Task management (continuous operation)
        self.pending_tasks = []  # Tasks waiting for assignment
        self.completed_tasks = []  # Tasks that have been completed

        # Time tracking
        self.current_time = 0.0
        self.last_inventory_check = 0.0
        self.inventory_check_interval = 60.0  # Check every minute

        # Episode metrics
        self.episode_stockouts = 0
        self.cumulative_reward = 0.0
        self.episode_collisions = 0
        self.last_collision_count = 0

    def reset(self):
        """Reset environment and return initial state dict."""
        # Initialize graph (use custom graph if provided, otherwise use config)
        if hasattr(self, '_custom_graph_state') and self._custom_graph_state is not None:
            # Use the custom graph state (from config file)
            # Note: Reusing the same graph_state object, so inventory persists across episodes
            self.graph_state = self._custom_graph_state

            # Reset edge congestion states
            for edge in self.graph_state.edges:
                edge.active_robot_ids = []
        else:
            # Create new graph from hospital_config or default
            self.graph_state = GraphState(config=self.hospital_config)

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
        self.pending_tasks = []
        self.completed_tasks = []
        self.next_task_id = 0
        self.current_time = 0.0
        self.last_inventory_check = 0.0

        # Generate initial inventory tasks
        new_tasks, self.next_task_id = generate_inventory_tasks(
            self.graph_state, self.current_time, self.next_task_id
        )
        self.pending_tasks.extend(new_tasks)

        # Add some ad-hoc tasks
        ad_hoc_tasks, self.next_task_id = generate_random_ad_hoc_tasks(
            self.graph_state, self.current_time, 2, self.next_task_id
        )
        self.pending_tasks.extend(ad_hoc_tasks)

        # Rank pending tasks by urgency
        if self.pending_tasks:
            task_queue = TaskQueue(tasks=self.pending_tasks)
            ranked = rank_tasks(task_queue, self.current_time)
            self.pending_tasks = ranked

        # Reset metrics
        self.episode_stockouts = 0
        self.cumulative_reward = 0.0
        self.episode_collisions = 0
        self.last_collision_count = 0

        return self._get_state_dict()

    def step(self, Δt: float = None):
        """
        Advance simulation by Δt seconds (continuous operation).

        This is the NEW continuous step function. Policy assigns tasks externally
        using assign_task_to_robot().

        Args:
            Δt: Time step in seconds (default: self.timestep_seconds)

        Returns:
            state_dict: Current state
            reward: Reward for this timestep
            done: False (continuous) or True if max_time reached
            info: Debug information
        """
        if Δt is None:
            Δt = self.timestep_seconds

        # 1. Update inventory levels (consumption)
        time_delta_hours = Δt / 3600.0
        update_inventory_levels(self.graph_state, time_delta_hours)

        # 2. Generate tasks from low-stock nodes
        if self.current_time - self.last_inventory_check >= self.inventory_check_interval:
            new_tasks, self.next_task_id = generate_inventory_tasks(
                self.graph_state, self.current_time, self.next_task_id
            )

            if new_tasks:
                self.pending_tasks.extend(new_tasks)

                # Re-rank all pending tasks
                task_queue = TaskQueue(tasks=self.pending_tasks)
                ranked = rank_tasks(task_queue, self.current_time)
                self.pending_tasks = ranked

            self.last_inventory_check = self.current_time

        # Occasionally add random ad-hoc tasks for testing
        if np.random.random() < 0.01:  # 1% chance per step
            ad_hoc, self.next_task_id = generate_random_ad_hoc_tasks(
                self.graph_state, self.current_time, 1, self.next_task_id
            )
            if ad_hoc:
                self.pending_tasks.extend(ad_hoc)

        # 3. Move robots along their paths
        self._update_robot_positions(Δt)

        # 4. Check task completions
        completed_tasks = self._check_task_completions()

        # 5. Compute rewards
        reward = self._compute_timestep_reward(completed_tasks)
        self.cumulative_reward += reward

        # 6. Update edge congestion
        self._update_edge_congestion()

        # 7. Advance time
        self.current_time += Δt

        # 8. Check termination
        done = self.current_time >= self.max_episode_time

        # 9. Build state
        state_dict = self._get_state_dict()

        info = {
            'current_time': self.current_time,
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(completed_tasks),
            'total_robot_tasks': sum(r.num_queued_tasks for r in self.robots),
            'stockouts': sum(1 for n in self.graph_state.nodes if n.is_stockout),
            'collisions': self.last_collision_count,
            'cumulative_reward': self.cumulative_reward
        }

        return state_dict, reward, done, info

    def assign_task_to_robot(self, robot_id: int, task: Task) -> bool:
        """
        Assign a task to a robot (called by policy/controller).

        Supports multi-capacity: robot can have multiple tasks queued
        if it has capacity.

        Args:
            robot_id: ID of robot to assign task to
            task: Task object to assign

        Returns:
            True if assignment successful, False otherwise
        """
        if robot_id < 0 or robot_id >= len(self.robots):
            return False

        robot = self.robots[robot_id]
        simulator = self.robot_simulators[robot_id]

        # Allow assignment regardless of capacity to avoid HOLD behavior

        # Split into pickup + dropoff legs
        pickup_task = Task(
            task_id=self.next_task_id,
            from_location_index=task.from_location_index,
            to_location_index=task.from_location_index,
            manual_priority=task.manual_priority,
            deadline=task.deadline,
            arrival_time=task.arrival_time,
            estimated_duration=task.estimated_duration,
            task_type=task.task_type,
            num_items=task.num_items,
            source_stock_level=task.source_stock_level,
            time_to_stockout=task.time_to_stockout,
            leg_type="pickup",
            parent_task_id=task.task_id,
            learned_score=task.learned_score
        )
        self.next_task_id += 1

        dropoff_task = Task(
            task_id=self.next_task_id,
            from_location_index=task.from_location_index,
            to_location_index=task.to_location_index,
            manual_priority=task.manual_priority,
            deadline=task.deadline,
            arrival_time=task.arrival_time,
            estimated_duration=task.estimated_duration,
            task_type=task.task_type,
            num_items=task.num_items,
            source_stock_level=task.source_stock_level,
            time_to_stockout=task.time_to_stockout,
            leg_type="dropoff",
            parent_task_id=task.task_id,
            learned_score=task.learned_score
        )
        self.next_task_id += 1

        # Add legs to robot's queue (auto-sorted by priority)
        robot.add_task(pickup_task)
        robot.add_task(dropoff_task)

        # Mark tasks as assigned
        pickup_task.is_assigned = True
        pickup_task.assigned_robot_id = robot_id
        dropoff_task.is_assigned = True
        dropoff_task.assigned_robot_id = robot_id

        # Remove from pending
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)

        # If this is the first task (robot was idle), plan path immediately
        if len(robot.task_queue) == 1:
            self._plan_path_for_robot(robot, pickup_task, simulator)

        return True

    def _plan_path_for_robot(self, robot: RobotState, task: Task, simulator):
        """
        Plan path for robot to complete task.

        Args:
            robot: RobotState object
            task: Task to plan path for
            simulator: RobotSimulator for this robot
        """
        # Get robot's current location
        start_node = robot.current_node_index
        if start_node is None:
            # Robot is between nodes, find nearest
            start_node = self._find_nearest_node(robot.telemetry.x, robot.telemetry.y)

        # Plan path from current location to task destination
        path, distance = dijkstra_shortest_path(
            start_node,
            task.to_location_index,
            self.graph_state,
            len(self.graph_state.nodes)  # Use actual number of nodes in graph
        )

        # Set path in simulator
        simulator.set_path(path, task.task_id, task.num_items)

        # Update robot state
        robot.planned_path = path
        robot.target_node_index = task.to_location_index
        robot.travel_start_time = self.current_time
        task.estimated_completion_time = self.current_time + distance

    def _update_robot_positions(self, Δt: float):
        """
        Move robots along their planned paths.

        Args:
            Δt: Time elapsed in seconds
        """
        for robot, simulator in zip(self.robots, self.robot_simulators):
            # Update simulator (moves robot)
            telemetry = simulator.update(Δt)
            telemetry.timestamp = self.current_time

            # Update robot telemetry
            robot.update_telemetry(telemetry)

    def _check_task_completions(self) -> List[Task]:
        """
        Check if any robots have completed their current tasks.

        Returns:
            List of completed tasks
        """
        completed_tasks = []

        for robot, simulator in zip(self.robots, self.robot_simulators):
            # Check if robot has arrived at destination
            # Robot completed task if:
            # 1. Has a current task
            # 2. Is at the target node
            # 3. No more path segments to traverse
            if (robot.current_task and
                robot.current_node_index == robot.target_node_index and
                not simulator.path_queue and
                simulator.current_target_node is None):

                task = robot.current_task

                # Complete delivery
                if task.leg_type == "pickup":
                    simulator.load_items(task.num_items)
                    robot.mark_pickup_complete(task.parent_task_id)
                elif task.task_type == 'replenishment':
                    to_node = self.graph_state.nodes[task.to_location_index]
                    to_node.restock(task.num_items)

                # Unload items from simulator on dropoff
                if task.leg_type == "dropoff":
                    simulator.complete_task(task.num_items)
                    robot.mark_dropoff_complete(task.parent_task_id)

                # Remove task from robot queue
                completed_task = robot.complete_current_task()
                if completed_task:
                    if completed_task.leg_type == "dropoff":
                        self.completed_tasks.append(completed_task)
                        completed_tasks.append(completed_task)

                # If robot has more tasks, plan next one
                if robot.current_task:
                    while robot.current_task and robot.current_task.leg_type == "dropoff" and not robot.is_pickup_complete(robot.current_task.parent_task_id):
                        robot.demote_current_task()
                    if robot.current_task:
                        self._plan_path_for_robot(robot, robot.current_task, simulator)

        return completed_tasks

    def _compute_timestep_reward(self, completed_tasks: List[Task]) -> float:
        """
        Compute reward for this timestep.

        Args:
            completed_tasks: List of tasks completed this timestep

        Returns:
            Reward scalar
        """
        reward = 0.0

        # 1. Task completion rewards
        for task in completed_tasks:
            reward += 25.0  # Heavy completion bonus

            completion_time = self.current_time
            if task.deadline:
                time_to_depletion = task.deadline - completion_time
                if time_to_depletion > 0:
                    reward += 10.0 * (time_to_depletion / 3600.0)
                else:
                    # Late completion penalty (inventory depleted)
                    reward -= 400.0 * (abs(time_to_depletion) / 60.0)

            if task.task_type == 'replenishment' and task.leg_type == "dropoff":
                to_node = self.graph_state.nodes[task.to_location_index]
                capacity_ratio = task.num_items / max(to_node.max_stock, 1.0)
                reward += 30.0 * min(1.0, capacity_ratio)
                if to_node.is_stockout:
                    reward -= 500.0

        # 2. Pending task penalties (age accumulation)
        for task in self.pending_tasks:
            age = task.get_age(self.current_time)

            # Higher penalty for urgent tasks
            if task.manual_priority >= 4:
                reward -= 0.02 * age
            else:
                reward -= 0.002 * age

        # 2b. Backlog penalty (discourage large queues)
        backlog_size = len(self.pending_tasks)
        if backlog_size > 0:
            reward -= 0.2 * backlog_size

        # 3. Stockout penalties
        stockout_count = 0
        for node in self.graph_state.nodes:
            if node.is_stockout and node.node_type == 'recovery':
                reward -= 5.0  # Per timestep stockout penalty
                stockout_count += 1

        self.episode_stockouts = stockout_count

        # 4. Load balancing bonus
        if len(self.robots) > 0:
            loads = [robot.current_load for robot in self.robots]
            load_variance = np.var(loads)
            reward -= 0.1 * load_variance

        # 5. Congestion penalties (heavy)
        congestion_penalty = 0.0
        for edge in self.graph_state.edges:
            num_active = len(edge.active_robot_ids)
            if num_active > 0:
                corridor_capacity = max(1.0, edge.corridor_width / 0.6)
                congestion_factor = num_active / corridor_capacity
                congestion_penalty += congestion_factor ** 2
        reward -= 5.0 * congestion_penalty

        # 6. Collision penalty (robots too close)
        collision_distance_m = 0.5
        collision_count = 0
        for i in range(len(self.robots)):
            ri = self.robots[i]
            if not ri.telemetry:
                continue
            for j in range(i + 1, len(self.robots)):
                rj = self.robots[j]
                if not rj.telemetry:
                    continue
                dx = ri.telemetry.x - rj.telemetry.x
                dy = ri.telemetry.y - rj.telemetry.y
                if (dx * dx + dy * dy) ** 0.5 < collision_distance_m:
                    collision_count += 1

        if collision_count > 0:
            reward -= 5.0 * collision_count

        self.episode_collisions += collision_count
        self.last_collision_count = collision_count

        return reward

    def get_robot_availability_mask(self, task: Task) -> np.ndarray:
        """
        Get boolean mask for which robots can accept a task.

        In continuous mode, robots can accept tasks even when busy
        if they have capacity.

        Args:
            task: Task to check availability for

        Returns:
            Boolean numpy array [num_robots]
        """
        return np.ones(len(self.robots), dtype=bool)

    def _get_state_dict(self) -> Dict[str, np.ndarray]:
        """
        Build graph-structured state dictionary for GAPO.

        Returns dict with:
        - task_features: [12]
        - node_continuous: [num_nodes, 15] - continuous node features
        - node_categorical: [num_nodes, 1] - node_type_id
        - edge_features: [num_edges, 12] - continuous edge features
        - edge_node_indices: [num_edges, 2] - (from_node_idx, to_node_idx)
        - edge_index: [2, num_edges] - graph connectivity for GNN
        - robot_features: [num_robots, 12]
        - robot_positions: [num_robots, 2]
        - queue_features: [11]
        """
        # Task features (get most urgent pending task, or zeros if none)
        if self.pending_tasks:
            current_task = self.pending_tasks[0]  # Most urgent task
            task_features = current_task.get_features(self.current_time)
        else:
            task_features = np.zeros(12, dtype=np.float32)

        # Node features (COMPLETE extraction)
        node_continuous, node_categorical = self.graph_state.get_node_features_complete()
        # node_continuous: [num_nodes, 15]
        # node_categorical: [num_nodes, 1]

        # Edge features (COMPLETE extraction)
        edge_features_orig, edge_node_indices_orig = self.graph_state.get_edge_features_complete()
        # edge_features_orig: [num_orig_edges, 12]
        # edge_node_indices_orig: [num_orig_edges, 2]

        # Make edges bidirectional to match edge_index
        # Edge index is bidirectional, so edge features must be too
        edge_features = np.vstack([edge_features_orig, edge_features_orig])  # [num_edges * 2, 12]
        edge_node_indices = np.vstack([
            edge_node_indices_orig,                          # Original: from -> to
            edge_node_indices_orig[:, [1, 0]]                # Reverse: to -> from
        ])  # [num_edges * 2, 2]

        # Edge index (connectivity for GNN message passing)
        edge_index = self._build_edge_index()  # [2, num_edges * 2]

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

        # Queue features (based on pending tasks + robot queues)
        if self.pending_tasks:
            num_pending = len(self.pending_tasks)
            avg_priority = np.mean([t.manual_priority for t in self.pending_tasks])
            num_urgent = sum(1 for t in self.pending_tasks if t.manual_priority >= 4)
            oldest_age = max([t.get_age(self.current_time) for t in self.pending_tasks])
            num_near_deadline = sum(1 for t in self.pending_tasks if t.get_time_to_deadline(self.current_time) < 300)
        else:
            num_pending = 0
            avg_priority = 0.0
            num_urgent = 0
            oldest_age = 0.0
            num_near_deadline = 0

        main_pickups = 0
        main_dropoffs = 0
        overflow_pickups = 0
        overflow_dropoffs = 0
        total_free_slots = 0
        total_overflow = 0

        for robot in self.robots:
            total_free_slots += max(0, robot.max_capacity - robot.current_load)
            total_overflow += len(robot.overflow_queue)
            for t in robot.task_queue:
                if getattr(t, "leg_type", "full") == "pickup":
                    main_pickups += 1
                elif getattr(t, "leg_type", "full") == "dropoff":
                    main_dropoffs += 1
            for t in robot.overflow_queue:
                if getattr(t, "leg_type", "full") == "pickup":
                    overflow_pickups += 1
                elif getattr(t, "leg_type", "full") == "dropoff":
                    overflow_dropoffs += 1

        avg_free_slots = total_free_slots / max(len(self.robots), 1)
        avg_overflow = total_overflow / max(len(self.robots), 1)

        queue_features = np.array([
            float(num_pending),
            avg_priority,
            float(num_urgent),
            oldest_age,
            float(num_near_deadline),
            float(main_pickups),
            float(main_dropoffs),
            float(overflow_pickups),
            float(overflow_dropoffs),
            float(avg_free_slots),
            float(avg_overflow)
        ], dtype=np.float32)

        return {
            'task_features': task_features,
            'node_continuous': node_continuous,
            'node_categorical': node_categorical,
            'edge_features': edge_features,
            'edge_node_indices': edge_node_indices,
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
        mask = np.ones(self.num_robots, dtype=bool)

        if self.current_task_index >= len(self.tasks):
            mask[:] = False
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
                    len(self.graph_state.nodes)
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
            start_node, task.to_location_index, self.graph_state, len(self.graph_state.nodes)
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
            start_node, task.to_location_index, self.graph_state, len(self.graph_state.nodes)
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
