"""
GAPO-compatible environment for hospital robot task allocation.

Returns graph-structured states (dict) instead of flat vectors.
Compatible with GNN-based GAPO policy network.
"""
import numpy as np
from typing import List, Dict, Optional

from .graph.graph_state import GraphState
from .graph.hospital_config import HospitalConfig
from .robot.robot_state import RobotState, create_default_robot
from .robot.robot_simulator import RobotSimulator
from .tasks.task_state import Task
from .tasks.task_generator import (
    generate_inventory_tasks,
    generate_random_ad_hoc_tasks,
    update_inventory_levels
)
from ..multi_agent_ppo.task_creation_actor import TaskCreationActor
from .graph_helpers import dijkstra_shortest_path
from ..multi_agent_ppo.learned_edge_cost import EdgeCostManager


class GAPOTaskAssignmentEnv:
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
        hospital_config: Optional[HospitalConfig] = None,
        stochastic_tasks_per_hour: float = 2.0,
        max_stochastic_tasks_per_hour: int = 2,
        initial_stochastic_tasks: int = 0,
        use_task_creation_actor: bool = True,
        device: str = 'cpu',
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
        # Fallback-only heuristic queue ranker.
        # Default OFF so training/deployment can use learned ranking as source of truth.
        self.use_heuristic_pending_rank = False

        # Time tracking
        self.current_time = 0.0
        self.last_inventory_check = 0.0
        self.inventory_check_interval = 10.0  # Check every 10 seconds
        self.stochastic_tasks_per_hour = max(0.0, float(stochastic_tasks_per_hour))
        self.max_stochastic_tasks_per_hour = max(0, int(max_stochastic_tasks_per_hour))
        self.initial_stochastic_tasks = max(0, int(initial_stochastic_tasks))
        self._stochastic_task_timestamps = []

        # Episode metrics
        self.episode_stockouts = 0
        self.cumulative_reward = 0.0
        self.episode_collisions = 0
        self.last_collision_count = 0

        # Per-task completion credits from the most recent timestep (populated by
        # _compute_timestep_reward, consumed by step() info dict and run_training.py).
        self._last_per_task_credits: Dict[int, float] = {}

        # Proactive SKU task creation actor (#7 factorizer + #8 scorer)
        # Creates inventory-aware stochastic replenishment tasks instead of blind random.
        if use_task_creation_actor:
            self.task_creation_actor = TaskCreationActor(device=device)
        else:
            self.task_creation_actor = None

        # Learned edge cost model (supervised, trains from traversal data)
        self.edge_cost_manager = EdgeCostManager(
            hidden_dim=64,
            lr=1e-3,
            risk_sensitivity=0.5,
            min_train_samples=100,
            train_interval_records=50,
            train_steps_per_interval=10,
        )

        # Battery management
        self._offline_robots: set = set()
        self._robots_routing_to_charge: set = set()
        self.battery_safety_margin = 1.5
        self.battery_critical_threshold = 0.10
        self.battery_low_step_penalty = 0.05
        self.battery_critical_task_penalty = 1.0
        self.emergency_tasks: list = []
        self._last_battery_penalty = 0.0

    def reset(self):
        """Reset environment and return initial state dict."""
        # Initialize graph (use custom graph if provided, otherwise use config)
        if hasattr(self, '_custom_graph_state') and self._custom_graph_state is not None:
            # Use the custom graph state (from config file)
            # Note: Reusing the same graph_state object, so inventory persists across episodes
            self.graph_state = self._custom_graph_state

            # Reset edge and node occupancy (graph object is reused, so these must
            # be explicitly cleared — otherwise stale robot IDs from the previous
            # episode corrupt the occupancy_count node feature fed to the GNN)
            for edge in self.graph_state.edges:
                edge.active_robot_ids = []
            for node in self.graph_state.nodes:
                node.current_robot_ids = []
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
        self.graph_state.current_time = self.current_time
        self._stochastic_task_timestamps = []

        # Generate initial inventory tasks (no existing tasks yet at reset)
        new_tasks, self.next_task_id = generate_inventory_tasks(
            self.graph_state, self.current_time, self.next_task_id,
            existing_task_keys=set()
        )
        for task in new_tasks:
            task.source = 'deterministic'
        self.pending_tasks.extend(new_tasks)

        # Optional initial stochastic tasks, bounded by hourly cap.
        if self.initial_stochastic_tasks > 0:
            initial_budget = self._remaining_stochastic_task_budget()
            initial_count = min(self.initial_stochastic_tasks, initial_budget)
            created = 0
            for _ in range(initial_count):
                new_task = None
                if self.task_creation_actor is not None:
                    new_task, self.next_task_id = self.task_creation_actor.create_task(
                        graph_state=self.graph_state,
                        pending_tasks=self.pending_tasks,
                        robots=self.robots,
                        current_time=self.current_time,
                        next_task_id=self.next_task_id,
                        training=False,
                    )
                if new_task is not None:
                    new_task.source = 'factoriser'
                    self.pending_tasks.append(new_task)
                    created += 1
                else:
                    # Fallback for seed tasks when no eligible SKU candidates exist
                    ad_hoc, self.next_task_id = generate_random_ad_hoc_tasks(
                        self.graph_state, self.current_time, 1, self.next_task_id
                    )
                    if ad_hoc:
                        ad_hoc[0].source = 'random_adhoc_fallback'
                        self.pending_tasks.extend(ad_hoc)
                        created += len(ad_hoc)
            if created > 0:
                self._record_stochastic_tasks(created)

        self._apply_fallback_pending_rank()

        # Reset metrics
        self.episode_stockouts = 0
        self.cumulative_reward = 0.0
        self.episode_collisions = 0
        self.last_collision_count = 0

        return self._get_state_dict()

    def step(self, dt: float = None):
        """
        Advance simulation by dt seconds (continuous operation).

        This is the NEW continuous step function. Policy assigns tasks externally
        using assign_task_to_robot().

        Args:
            dt: Time step in seconds (default: self.timestep_seconds)

        Returns:
            state_dict: Current state
            reward: Reward for this timestep
            done: False (continuous) or True if max_time reached
            info: Debug information
        """
        if dt is None:
            dt = self.timestep_seconds

        # 1. Update inventory levels (consumption)
        time_delta_hours = dt / 3600.0
        self.graph_state.current_time = self.current_time
        update_inventory_levels(self.graph_state, time_delta_hours)

        # 2. Generate tasks from low-stock nodes
        if self.current_time - self.last_inventory_check >= self.inventory_check_interval:
            new_tasks, self.next_task_id = generate_inventory_tasks(
                self.graph_state, self.current_time, self.next_task_id,
                existing_task_keys=self._build_covered_replenishment_keys()
            )

            if new_tasks:
                for task in new_tasks:
                    task.source = 'deterministic'
                self.pending_tasks.extend(new_tasks)
                self._apply_fallback_pending_rank()

            self.last_inventory_check = self.current_time

        # Stochastic task creation, rate-controlled and hard-capped per simulated hour.
        self._generate_stochastic_tasks(dt)

        # 3. Update edge congestion before motion (for speed adjustments)
        self._update_edge_congestion()

        # 4. Move robots along their paths
        self._update_robot_positions(dt)

        # 5. Check task completions
        completed_tasks = self._check_task_completions()

        # 6. Compute rewards
        reward = self._compute_timestep_reward(completed_tasks)
        self.cumulative_reward += reward

        # 7. Update edge congestion
        self._update_edge_congestion()

        # 8. Collect traversal records and feed to edge cost model
        self._collect_traversal_records()

        # 9. Advance time
        self.current_time += dt

        # 10. Check termination
        done = self.current_time >= self.max_episode_time

        # 11. Build state
        state_dict = self._get_state_dict()

        info = {
            'current_time': self.current_time,
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(completed_tasks),
            'total_robot_tasks': sum(r.num_queued_tasks for r in self.robots),
            'stockouts': sum(1 for n in self.graph_state.nodes if n.is_stockout),
            'collisions': self.last_collision_count,
            'cumulative_reward': self.cumulative_reward,
            'edge_cost_model': self.edge_cost_manager.get_stats(),
            # Per-task completion credits: {parent_task_id: reward}.
            # Used by run_training.py to route rewards to the causal assignment slot.
            'task_completion_credits': self._last_per_task_credits,
            # Ambient monitoring stats (not in reward, just for logging).
            'episode_stockouts': self.episode_stockouts,
            'episode_collisions': self.episode_collisions,
            # Battery management info
            'battery_penalty': self._last_battery_penalty,
            'emergency_tasks': len(self.emergency_tasks),
            'offline_robots': list(self._offline_robots),
            'robots_charging': [rid for rid, sim in enumerate(self.robot_simulators) if sim.is_charging],
        }

        return state_dict, reward, done, info

    def _apply_fallback_pending_rank(self):
        """
        Optional heuristic ranking for pending queue.

        This is intentionally fallback-only. Keep disabled when using learned ranking.
        """
        if not self.use_heuristic_pending_rank or not self.pending_tasks:
            return
        # Lazy import keeps heuristic ranking dependency out of the default path.
        from .tasks.task_state import TaskQueue, rank_tasks

        task_queue = TaskQueue(tasks=self.pending_tasks)
        self.pending_tasks = rank_tasks(task_queue, self.current_time)

    def _prune_stochastic_task_history(self):
        """Keep only stochastic task timestamps within the last simulated hour."""
        cutoff = self.current_time - 3600.0
        self._stochastic_task_timestamps = [
            t for t in self._stochastic_task_timestamps if t >= cutoff
        ]

    def _remaining_stochastic_task_budget(self) -> int:
        """Remaining stochastic tasks allowed in the rolling 1-hour window."""
        self._prune_stochastic_task_history()
        return max(0, self.max_stochastic_tasks_per_hour - len(self._stochastic_task_timestamps))

    def _record_stochastic_tasks(self, count: int):
        """Record creation timestamps for stochastic tasks."""
        if count <= 0:
            return
        self._stochastic_task_timestamps.extend([self.current_time] * count)
        self._prune_stochastic_task_history()

    def _generate_stochastic_tasks(self, dt: float):
        """
        Generate stochastic tasks via the TaskCreationActor pipeline (#7/#8).

        Rate and hard hourly cap are checked first. If the actor finds no eligible
        candidates, falls back to generate_random_ad_hoc_tasks so training is never
        starved of task diversity.
        """
        remaining_budget = self._remaining_stochastic_task_budget()
        if remaining_budget <= 0 or self.stochastic_tasks_per_hour <= 0.0:
            return

        expected_events = self.stochastic_tasks_per_hour * max(float(dt), 0.0) / 3600.0
        p_create = min(max(expected_events, 0.0), 1.0)
        if np.random.random() >= p_create:
            return

        new_task = None
        if self.task_creation_actor is not None:
            new_task, self.next_task_id = self.task_creation_actor.create_task(
                graph_state=self.graph_state,
                pending_tasks=self.pending_tasks,
                robots=self.robots,
                current_time=self.current_time,
                next_task_id=self.next_task_id,
                training=False,
            )

        if new_task is not None:
            new_task.source = 'factoriser'
            self.pending_tasks.append(new_task)
            self._record_stochastic_tasks(1)
            self._apply_fallback_pending_rank()
        else:
            # Fallback: blind random task when no eligible SKU candidates exist
            ad_hoc, self.next_task_id = generate_random_ad_hoc_tasks(
                self.graph_state, self.current_time, 1, self.next_task_id
            )
            if ad_hoc:
                ad_hoc[0].source = 'random_adhoc_fallback'  # Mark as fallback from factoriser
                self.pending_tasks.extend(ad_hoc)
                self._record_stochastic_tasks(len(ad_hoc))
                self._apply_fallback_pending_rank()
                # Track fallback occurrence
                if not hasattr(self, '_factoriser_fallback_count'):
                    self._factoriser_fallback_count = 0
                self._factoriser_fallback_count += 1

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

        # Gate: offline robots cannot accept tasks
        if robot_id in self._offline_robots:
            return False

        # Gate: robot needs to charge first
        if self._should_robot_charge(robot_id):
            return False

        robot = self.robots[robot_id]
        simulator = self.robot_simulators[robot_id]

        # Allow assignment regardless of capacity to avoid HOLD behavior

        was_idle = len(robot.task_queue) == 0

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
            sku_id=task.sku_id,
            category_key=task.category_key,
            category_id=task.category_id,
            sku_stock_level=task.sku_stock_level,
            sku_max_level=task.sku_max_level,
            reorder_point=task.reorder_point,
            par_level=task.par_level,
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
            sku_id=task.sku_id,
            category_key=task.category_key,
            category_id=task.category_id,
            sku_stock_level=task.sku_stock_level,
            sku_max_level=task.sku_max_level,
            reorder_point=task.reorder_point,
            par_level=task.par_level,
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
        if was_idle:
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
            len(self.graph_state.nodes),
            edge_cost_manager=self.edge_cost_manager,
            all_robots=self.robots,
        )

        # Set path in simulator
        simulator.set_path(path, task.task_id, task.num_items)

        # Stamp entry time on the traversal record the simulator just created
        simulator.set_traversal_entry_time(self.current_time)

        # Count approaching robots for the edge the simulator is now on
        if simulator.current_edge_index is not None:
            approaching = self._count_approaching_robots(
                simulator.current_edge_index, exclude_robot_id=robot.robot_id
            )
            simulator.set_traversal_approaching_count(approaching)

        # Update robot state
        robot.planned_path = path
        robot.target_node_index = task.to_location_index
        robot.travel_start_time = self.current_time
        task.estimated_completion_time = self.current_time + distance

    def _update_robot_positions(self, dt: float):
        """
        Move robots along their planned paths.

        Args:
            dt: Time elapsed in seconds
        """
        for robot, simulator in zip(self.robots, self.robot_simulators):
            # Track edge index before update (to detect edge completion)
            prev_edge_idx = simulator.current_edge_index

            # Update simulator (moves robot)
            telemetry = simulator.update(dt)
            telemetry.timestamp = self.current_time

            # Stamp exit time on any traversal that just completed
            if prev_edge_idx is not None and simulator.current_edge_index != prev_edge_idx:
                simulator.finalize_current_traversal(self.current_time)

            # Stamp entry time on any new traversal that just started
            if simulator.current_edge_index is not None and simulator.current_edge_index != prev_edge_idx:
                simulator.set_traversal_entry_time(self.current_time)

            # Update robot telemetry
            robot.update_telemetry(telemetry)

            # If robot is idle but has queued tasks, plan the next valid task
            if (not simulator.path_queue and simulator.current_target_node is None and
                    robot.current_task is not None):
                _max_demotes = len(robot.task_queue)
                _demotes = 0
                while (_demotes < _max_demotes and
                        robot.current_task and robot.current_task.leg_type == "dropoff" and
                        not robot.is_pickup_complete(robot.current_task.parent_task_id)):
                    robot.demote_current_task()
                    _demotes += 1
                if robot.current_task and not (
                    robot.current_task.leg_type == "dropoff" and
                    not robot.is_pickup_complete(robot.current_task.parent_task_id)
                ):
                    self._plan_path_for_robot(robot, robot.current_task, simulator)

            # Trigger charging trip when robot becomes idle and needs charging
            robot_id = robot.robot_id
            if (robot_id not in self._robots_routing_to_charge and
                    robot_id not in self._offline_robots and
                    simulator.is_charging is False and
                    len(robot.task_queue) == 0 and
                    simulator.path_queue == [] and
                    simulator.current_target_node is None and
                    self._should_robot_charge(robot_id)):
                self._send_robot_to_charge(robot_id)

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
                    if task.sku_id:
                        to_node.restock_sku(task.sku_id, task.num_items)
                        to_node.recalc_category_inventory()
                        to_node.stock_level = sum(v.get('stock', 0.0) for v in to_node.category_inventory.values())
                    else:
                        to_node.restock(task.num_items)

                # Unload items from simulator on dropoff
                if task.leg_type == "dropoff":
                    simulator.complete_task(task.num_items)
                    robot.mark_dropoff_complete(task.parent_task_id)

                # Remove task from robot queue
                completed_task = robot.complete_current_task()
                if completed_task:
                    # Add all task legs to reward computation (pickups + dropoffs)
                    completed_tasks.append(completed_task)
                    # Track only dropoffs for episode history
                    if completed_task.leg_type == "dropoff":
                        self.completed_tasks.append(completed_task)

                # If robot has more tasks, plan next one
                if robot.current_task:
                    _max_demotes = len(robot.task_queue)
                    _demotes = 0
                    while (_demotes < _max_demotes and
                            robot.current_task and robot.current_task.leg_type == "dropoff" and
                            not robot.is_pickup_complete(robot.current_task.parent_task_id)):
                        robot.demote_current_task()
                        _demotes += 1
                    if robot.current_task and not (
                        robot.current_task.leg_type == "dropoff" and
                        not robot.is_pickup_complete(robot.current_task.parent_task_id)
                    ):
                        self._plan_path_for_robot(robot, robot.current_task, simulator)

            # Handle charging arrival
            robot_id = robot.robot_id
            if (robot_id in self._robots_routing_to_charge and
                    simulator.active_task_id == -1 and
                    robot.current_node_index is not None):
                node = self.graph_state.nodes[robot.current_node_index]
                if node.node_type == 'hub' and not simulator.is_charging:
                    simulator.start_charging()

            # Handle charging completion
            if (simulator.is_charging is False and robot_id in self._robots_routing_to_charge and
                    simulator.active_task_id == -1):
                # Charging just finished
                self._robots_routing_to_charge.discard(robot_id)
                if robot.task_queue:
                    self._plan_path_for_robot(robot, robot.current_task, simulator)

        return completed_tasks

    def _compute_timestep_reward(self, completed_tasks: List[Task]) -> float:
        """
        Compute reward combining inventory health + task completion + utilization.

        Components:
        A. Ad-hoc completion bonus: +1.0 per task
        B. Pickup intermediate signal: +0.5 per unit effective load
        C. Increased replenishment bonus: 0 to +10 (was +6)
        D. Utilization incentive on dropoff: +2.0 based on load ratio

        All rewards are per-task only — no ambient signals enter the training buffer.
        Each completed task's reward is routed back to the memory slot of the action
        that originally assigned it (via run_training.py's task_to_memory_idx mechanism).
        The per-task credits are stored in self._last_per_task_credits for step() to
        pass through the info dict.

        Ambient environment stats (stockouts, congestion, collisions) are still tracked
        as episode metrics for monitoring but are NOT included in the returned reward.

        Args:
            completed_tasks: List of tasks completed this timestep (both pickups and dropoffs)

        Returns:
            Total completion reward scalar (sum of per-task rewards)
        """
        self._last_per_task_credits = {}
        total_reward = 0.0

        # Per-task completion rewards
        for task in completed_tasks:
            task_reward = 0.0  # No flat base

            # Get assigned robot for load-based calculations
            assigned_robot_id = getattr(task, 'assigned_robot_id', None)
            robot = None
            if assigned_robot_id is not None:
                for r in self.robots:
                    if r.robot_id == assigned_robot_id:
                        robot = r
                        break

            # C: Replenishment dropoff (increased from +6 to +10)
            # Stock health at delivery time: high reward when proactive (before depletion)
            if task.task_type == 'replenishment' and task.leg_type == 'dropoff':
                to_node = self.graph_state.nodes[task.to_location_index]
                if task.sku_id and getattr(to_node, 'sku_inventory', None) and task.sku_id in to_node.sku_inventory:
                    sku_data = to_node.sku_inventory[task.sku_id]
                    pre_stock = max(0.0, float(sku_data['stock']) - task.num_items)
                    pre_ratio = pre_stock / max(float(sku_data['max']), 1.0)
                else:
                    pre_ratio = 0.0
                task_reward += 10.0 * pre_ratio

            # A: Ad-hoc task completion bonus
            if task.task_type == 'ad_hoc' and task.leg_type == 'dropoff':
                task_reward += 1.0

            # B: Pickup intermediate signal
            # Reward for picking up items (intermediate credit before dropoff)
            if task.leg_type == 'pickup' and robot:
                load_ratio = robot.effective_load / max(robot.max_capacity, 1.0)
                task_reward += 0.5 * load_ratio

            # D: Utilization bonus on dropoff
            # Encourage efficient batching (more items per trip)
            if task.leg_type == 'dropoff' and robot:
                load_ratio = robot.effective_load / max(robot.max_capacity, 1.0)
                task_reward += 2.0 * load_ratio

            # F: Depletion penalty - deduct if robot is critically low on battery at task end
            if task.leg_type == 'dropoff' and robot:
                sim = self.robot_simulators[robot.robot_id]
                if sim.battery_level < self.battery_critical_threshold:
                    task_reward -= self.battery_critical_task_penalty

            parent_id = getattr(task, 'parent_task_id', None)
            if parent_id is not None:
                self._last_per_task_credits[parent_id] = task_reward
            total_reward += task_reward

        # ── Ambient monitoring (NOT included in reward) ─────────────────────────
        # Track stockouts for episode metrics / info dict only.
        stockout_count = 0
        for node in self.graph_state.nodes:
            if not node.sku_inventory or not node.consumption_enabled:
                continue
            for sku_data in node.sku_inventory.values():
                if float(sku_data.get("stock", 0.0)) <= 0:
                    stockout_count += 1
        self.episode_stockouts = stockout_count

        # Track collisions for episode metrics / info dict only.
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
        self.episode_collisions += collision_count
        self.last_collision_count = collision_count

        # E: Per-step low-battery ambient penalty
        battery_penalty = 0.0
        for rid in range(len(self.robots)):
            if rid in self._offline_robots:
                continue
            if self._should_robot_charge(rid):
                battery_penalty -= self.battery_low_step_penalty
        self._last_battery_penalty = battery_penalty
        total_reward += battery_penalty

        return total_reward

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
        - task_features: [15]
        - node_continuous: [num_nodes, N] - continuous node features
        - node_categorical: [num_nodes, 1] - node_type_id
        - edge_features: [num_edges, 21] - continuous edge features
        - edge_node_indices: [num_edges, 2] - (from_node_idx, to_node_idx)
        - edge_index: [2, num_edges] - graph connectivity for GNN
        - robot_features: [num_robots, 20]
        - robot_positions: [num_robots, 2]
        - queue_features: [16]
        """
        # Task features (get most urgent pending task, or zeros if none)
        if self.pending_tasks:
            current_task = self.pending_tasks[0]  # Most urgent task
            task_features = current_task.get_features(self.current_time)
        else:
            task_features = np.zeros(15, dtype=np.float32)

        # Node features (v3 uses category stats when available)
        try:
            if getattr(self.graph_state, "category_order", None):
          
                node_continuous, node_categorical = self.graph_state.get_node_features_with_category_stats()

                node_sku_features, node_sku_mask = self.graph_state.get_node_sku_features()
            else:

                node_continuous, node_categorical = self.graph_state.get_node_features_complete()

                node_sku_features, node_sku_mask = (None, None)
        except Exception as e:

            import traceback
            traceback.print_exc()
            raise

        # Ensure node_categorical has 4 columns: [node_type, department, shift, day_type]
        

        if node_categorical.shape[1] != 4:
            num_nodes = node_continuous.shape[0]
            padded = np.zeros((num_nodes, 4), dtype=np.int64)
            # Default department to -1 (unknown), shift/day to 0
            padded[:, 1] = -1
            cols = min(node_categorical.shape[1], 4)
            padded[:, :cols] = node_categorical[:, :cols].astype(np.int64)
            node_categorical = padded

        # Edge features (COMPLETE extraction)
        edge_features_orig, edge_node_indices_orig = self.graph_state.get_edge_features_complete()
        # edge_features_orig: [num_orig_edges, 21]
        # edge_node_indices_orig: [num_orig_edges, 2]

        # Make edges bidirectional to match edge_index
        # Edge index is bidirectional, so edge features must be too
        edge_features = np.vstack([edge_features_orig, edge_features_orig])  # [num_edges * 2, 21]
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
                heading_sin = float(np.sin(robot.telemetry.heading))
                heading_cos = float(np.cos(robot.telemetry.heading))
                max_v = 1.0
                velocity_ratio = float(robot.telemetry.velocity_ms / max_v) if max_v > 0 else 0.0
                remaining_path = robot.telemetry.remaining_path or []
                has_path = 1.0 if (robot.telemetry.current_edge_index is not None or remaining_path) else 0.0
                target_node = remaining_path[0] if remaining_path else -1
                is_blocked = 1.0 if (has_path and robot.telemetry.velocity_ms < 0.05) else 0.0
                feat = [
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
                    float(robot.effective_load),   # effective load: physical + committed pickups
                    heading_sin,
                    heading_cos,
                    velocity_ratio,
                    float(robot.telemetry.is_available),
                    float(has_path),
                    float(len(remaining_path)),
                    float(target_node),
                    float(is_blocked),
                ]
                robot_features.append(feat)
                robot_positions.append([robot.telemetry.x, robot.telemetry.y])
            else:
                robot_features.append(np.zeros(19))
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

        seconds_in_day = self.current_time % 86400.0
        day_frac = seconds_in_day / 86400.0
        time_sin = float(np.sin(2 * np.pi * day_frac))
        time_cos = float(np.cos(2 * np.pi * day_frac))
        day_of_week = int(self.current_time // 86400.0) % 7
        day_norm = day_of_week / 6.0 if 6.0 > 0 else 0.0

        moving_count = sum(1 for r in self.robots if r.telemetry and r.telemetry.is_moving)
        fleet_busy_ratio = moving_count / max(len(self.robots), 1)

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
            float(avg_overflow),
            time_sin,
            time_cos,
            day_norm,
            float(fleet_busy_ratio),
            float(len(self.robots))
        ], dtype=np.float32)

        # FINAL VALIDATION: Ensure node_categorical is 2D [num_nodes, 4]
        assert node_categorical.ndim == 2, \
            f"node_categorical must be 2D, got shape {node_categorical.shape}"
        assert node_categorical.shape[1] == 4, \
            f"node_categorical must have 4 columns, got shape {node_categorical.shape}"

        return {
            'task_features': task_features,
            'node_continuous': node_continuous,
            'node_categorical': node_categorical,
            'node_sku_features': node_sku_features,
            'node_sku_mask': node_sku_mask,
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

    def _update_edge_congestion(self):
        """Update edge and node occupancy from current telemetry."""
        for edge in self.graph_state.edges:
            edge.active_robot_ids.clear()
            edge.active_robot_progress.clear()
            edge.approaching_robot_count = 0

        # Clear node occupancy so it is rebuilt fresh each step.
        # node.add_robot / remove_robot are never called incrementally, so we
        # reconstruct from telemetry here — the same pattern used for edges above.
        for node in self.graph_state.nodes:
            node.current_robot_ids.clear()

        for robot, simulator in zip(self.robots, self.robot_simulators):
            if not robot.telemetry:
                continue
            if robot.telemetry.is_on_edge:
                edge_idx = robot.telemetry.current_edge_index
                if edge_idx is not None and 0 <= edge_idx < len(self.graph_state.edges):
                    edge = self.graph_state.edges[edge_idx]
                    edge.active_robot_ids.append(robot.robot_id)
                    edge.active_robot_progress[robot.robot_id] = (
                        robot.telemetry.edge_progress,
                        simulator.current_node_index,
                        simulator.current_target_node
                    )
            else:
                # Robot is at a node — update node occupancy
                node_idx = robot.telemetry.current_node_index
                if node_idx is not None and 0 <= node_idx < len(self.graph_state.nodes):
                    self.graph_state.nodes[node_idx].add_robot(robot.robot_id)

        for edge_idx, edge in enumerate(self.graph_state.edges):
            edge.approaching_robot_count = self._count_approaching_robots(edge_idx)

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

    def _collect_traversal_records(self):
        """Collect completed traversal records from all simulators and feed to edge cost model."""
        all_records = []
        for simulator in self.robot_simulators:
            records = simulator.get_and_clear_traversal_records()
            all_records.extend(records)

        if all_records:
            self.edge_cost_manager.add_traversal_records(all_records)

    def _build_covered_replenishment_keys(self) -> set:
        """
        Return a set of (sku_id, dest_node_idx) tuples that are already covered
        by a pending or in-flight replenishment task, so the inventory check
        does not create duplicates.

        Non-SKU nodes use (None, dest_node_idx) as the key.

        For split tasks (pickup+dropoff legs), both legs are recognized as covering
        the same destination demand. We extract destination from dropoff leg or from
        the original pending task.
        """
        keys = set()

        # Pending tasks (not yet assigned)
        for task in self.pending_tasks:
            if task.task_type == 'replenishment':
                keys.add((task.sku_id, task.to_location_index))

        # Build a map of parent_task_id -> destination for split tasks
        # This lets us find the real destination even when only the pickup leg is in a robot queue
        parent_to_destination = {}

        for robot in self.robots:
            for task in robot.task_queue:
                if task.task_type == 'replenishment':
                    parent_id = getattr(task, 'parent_task_id', None)
                    if parent_id is not None:
                        # Map parent to its real destination (from dropoff leg if available)
                        leg_type = getattr(task, 'leg_type', 'full')
                        if leg_type == 'dropoff':
                            parent_to_destination[parent_id] = (task.sku_id, task.to_location_index)
                        elif parent_id not in parent_to_destination:
                            # Pickup leg: store placeholder, will be overwritten by dropoff if found
                            parent_to_destination[parent_id] = (task.sku_id, None)

            for task in robot.overflow_queue:
                if task.task_type == 'replenishment':
                    parent_id = getattr(task, 'parent_task_id', None)
                    if parent_id is not None:
                        leg_type = getattr(task, 'leg_type', 'full')
                        if leg_type == 'dropoff':
                            parent_to_destination[parent_id] = (task.sku_id, task.to_location_index)
                        elif parent_id not in parent_to_destination:
                            parent_to_destination[parent_id] = (task.sku_id, None)

        # Now add all destinations for split tasks, skipping None destinations
        for (sku_id, dest_idx) in parent_to_destination.values():
            if dest_idx is not None:
                keys.add((sku_id, dest_idx))

        # Also add any unsplit tasks (leg_type='full')
        for robot in self.robots:
            for task in robot.task_queue:
                if task.task_type == 'replenishment':
                    if getattr(task, 'leg_type', 'full') == 'full':
                        keys.add((task.sku_id, task.to_location_index))
            for task in robot.overflow_queue:
                if task.task_type == 'replenishment':
                    if getattr(task, 'leg_type', 'full') == 'full':
                        keys.add((task.sku_id, task.to_location_index))

        return keys

    def _count_approaching_robots(self, edge_index: int, exclude_robot_id: int = -1) -> int:
        """Count robots with the given edge in their planned path but not currently on it."""
        if edge_index is None or not (0 <= edge_index < len(self.graph_state.edges)):
            return 0
        edge = self.graph_state.edges[edge_index]
        count = 0
        for robot in self.robots:
            if robot.robot_id == exclude_robot_id:
                continue
            path = getattr(robot, 'planned_path', None)
            if not path or len(path) < 2:
                continue
            for i in range(len(path) - 1):
                from_id = self.graph_state.nodes[path[i]].node_id
                to_id = self.graph_state.nodes[path[i + 1]].node_id
                if ((edge.from_node == from_id and edge.to_node == to_id) or
                        (edge.from_node == to_id and edge.to_node == from_id)):
                    count += 1
                    break
        return count

    # ===== BATTERY MANAGEMENT =====

    def _get_hub_node_indices(self) -> list:
        """Get indices of all hub nodes in the graph."""
        return [i for i, n in enumerate(self.graph_state.nodes) if n.node_type == 'hub']

    def _should_robot_charge(self, robot_id: int) -> bool:
        """
        Check if robot battery is below threshold to reach nearest hub.

        Dynamic threshold: battery_needed = dist_to_hub × drain_rate × safety_margin
        Robot should charge if: battery_level <= battery_needed
        """
        if robot_id in self._offline_robots:
            return False
        simulator = self.robot_simulators[robot_id]
        robot = self.robots[robot_id]
        hub_indices = self._get_hub_node_indices()
        if not hub_indices:
            return False

        from .graph_helpers import estimate_travel_distance
        min_dist = min(estimate_travel_distance(robot, h, self.graph_state) for h in hub_indices)
        battery_needed = min_dist * simulator.battery_drain_rate * self.battery_safety_margin
        return simulator.battery_level <= battery_needed

    def _nearest_hub_index(self, robot_id: int) -> int:
        """Find index of nearest hub node to robot."""
        robot = self.robots[robot_id]
        hub_indices = self._get_hub_node_indices()
        from .graph_helpers import estimate_travel_distance
        return min(hub_indices, key=lambda h: estimate_travel_distance(robot, h, self.graph_state))

    def _send_robot_to_charge(self, robot_id: int):
        """Route robot to nearest hub to charge."""
        robot = self.robots[robot_id]
        simulator = self.robot_simulators[robot_id]
        hub_idx = self._nearest_hub_index(robot_id)

        start_node = robot.current_node_index
        if start_node is None:
            start_node = self._find_nearest_node(robot.telemetry.x, robot.telemetry.y)

        path, _ = dijkstra_shortest_path(
            start_node, hub_idx, self.graph_state, len(self.graph_state.nodes),
            edge_cost_manager=self.edge_cost_manager, all_robots=self.robots
        )
        simulator.set_path(path, task_id=-1, num_items=0)
        self._robots_routing_to_charge.add(robot_id)

    def mark_robot_offline(self, robot_id: int):
        """
        Mark robot as manually powered off.
        Items in transit go to emergency_tasks; other tasks go back to pending.
        """
        if robot_id < 0 or robot_id >= len(self.robots):
            return
        robot = self.robots[robot_id]
        simulator = self.robot_simulators[robot_id]

        in_transit_parents = set(robot.picked_up_task_ids)

        for task in robot.task_queue:
            parent_id = getattr(task, 'parent_task_id', None)
            if parent_id in in_transit_parents:
                # Items on robot - human handling
                task.manual_priority = 999
                task.task_type = 'emergency_manual'
                self.emergency_tasks.append(task)
            else:
                # Not yet picked up - reassign to pending
                self.pending_tasks.insert(0, task)

        # Clear robot queues
        robot.task_queue.clear()
        robot.overflow_queue.clear()
        robot.picked_up_task_ids.clear()
        robot.planned_path.clear()
        robot.target_node_index = None

        # Stop simulator
        simulator.path_queue.clear()
        simulator.current_target_node = None
        simulator.velocity_ms = 0.0
        simulator.active_task_id = None
        simulator.is_charging = False

        # Mark offline
        self._offline_robots.add(robot_id)
        self._robots_routing_to_charge.discard(robot_id)

    def bring_robot_online(self, robot_id: int):
        """Mark robot as powered back on."""
        self._offline_robots.discard(robot_id)
