import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..multi_agent_ppo.learned_edge_cost import EdgeCostManager
from ..multi_agent_ppo.task_creation_actor import TaskCreationActor
from ..reliability.telemetry_effects import TelemetryEffectsManager
from ..utils.fleet_event_logger import FleetEventLogger
from ..utils.sku_logger import SKULogger
from .graph_helpers import dijkstra_shortest_path
from .robot.robot_simulator import RobotSimulator
from .robot.robot_state import RobotState, create_default_robot
from .tasks.task_generator import (
    DEFAULT_MIN_REPLENISHMENT_DEADLINE_S,
    _compute_sku_rate,
    generate_inventory_tasks,
    update_inventory_levels,
)
from .tasks.task_state import Task

logger = logging.getLogger(__name__)


class GAPOTaskAssignmentEnv:
    def __init__(
        self,
        num_robots=1,
        num_nodes=3,
        max_episode_time=28800.0,
        timestep_seconds=1.0,
        stochastic_tasks_per_hour: float = 0.0,
        max_stochastic_tasks_per_hour: int = 2,
        initial_stochastic_tasks: int = 0,
        use_task_creation_actor: bool = True,
        device: str = "cpu",
        robot_max_capacity: int = 12,
        fleet_event_logger: Optional["FleetEventLogger"] = None,
        log_dir: Optional[str] = None,
    ):
        super(GAPOTaskAssignmentEnv, self).__init__()

        self.num_robots = num_robots
        self.num_nodes = num_nodes
        self.robot_max_capacity = robot_max_capacity
        self.max_episode_time = max_episode_time
        self.timestep_seconds = timestep_seconds
        self.fleet_event_logger = fleet_event_logger
        self.sku_logger = SKULogger(log_dir) if log_dir is not None else None
        self.graph_state = None
        self.robots = []
        self.robot_simulators = []
        self.next_task_id = 1

        self.pending_tasks = []
        self.completed_tasks = []

        self.current_time = 0.0
        self.last_inventory_check = 0.0
        self.inventory_check_interval = 10.0
        self.stochastic_tasks_per_hour = max(0.0, float(stochastic_tasks_per_hour))
        self.max_stochastic_tasks_per_hour = max(0, int(max_stochastic_tasks_per_hour))
        self.initial_stochastic_tasks = max(0, int(initial_stochastic_tasks))
        self._stochastic_task_timestamps = []

        self.episode_stockouts = 0
        self.cumulative_reward = 0.0
        self._last_per_task_credits: Dict[int, float] = {}
        # Keyed by (task_id, leg_type) because pickup and dropoff legs share task_id
        # after the leg-ID collapse. Without the composite key they would stomp each other.
        self._last_completed_leg_credits: Dict[Tuple[int, str], float] = {}
        # Battery level snapshotted when each leg starts navigating, keyed by (task_id, leg_type)
        # for the same reason. Used to compute per-leg distance and energy at completion.
        self._task_battery_snapshots: Dict[Tuple[int, str], float] = {}
        self.BATTERY_CAPACITY_WH = 500.0

        if use_task_creation_actor:
            self.task_creation_actor = TaskCreationActor(device=device)
        else:
            self.task_creation_actor = None

        self.edge_cost_manager = EdgeCostManager(
            hidden_dim=64,
            lr=1e-3,
            risk_sensitivity=0.5,
            min_train_samples=100,
            train_interval_records=50,
            train_steps_per_interval=10,
        )

        self._offline_robots: set = set()
        self._robots_routing_to_charge: set = set()
        self.battery_safety_margin = 1.5
        self.battery_critical_threshold = 0.10
        self.battery_critical_task_penalty = 1.0
        self.emergency_tasks: list = []

        self.proactive_restock_lead_time_s = 900.0
        self.break_movement_penalty = 0.1
        self.episode_break_stockouts = 0

        self.telemetry_noise_manager = TelemetryEffectsManager(enable=True, graph_bounds=None)
        self.robot_reset_times = []
        self.graph_bounds = None

    def reset(self):
        if hasattr(self, "_custom_graph_state") and self._custom_graph_state is not None:
            self.graph_state = self._custom_graph_state

            for edge in self.graph_state.edges:
                edge.active_robot_ids = []
            for node in self.graph_state.nodes:
                node.current_robot_ids = []
        else:
            raise RuntimeError(
                "No ILC config loaded. Call create_env_from_config_file() before reset()."
            )

        if self.graph_state and self.graph_state.nodes:
            x_coords = [node.x for node in self.graph_state.nodes]
            y_coords = [node.y for node in self.graph_state.nodes]
            padding = 5.0
            self.graph_bounds = (
                min(x_coords) - padding,
                max(x_coords) + padding,
                min(y_coords) - padding,
                max(y_coords) + padding,
            )
            self.telemetry_noise_manager.graph_bounds = self.graph_bounds

        schedule = getattr(self.graph_state, "school_schedule", None)
        if schedule:
            self.proactive_restock_lead_time_s = float(
                schedule.get("proactive_restock_lead_time_s", self.proactive_restock_lead_time_s)
            )
        self.episode_break_stockouts = 0

        self.robots = []
        self.robot_simulators = []
        self.robot_reset_times = [self.current_time] * self.num_robots

        for i in range(self.num_robots):
            initial_node = np.random.randint(0, self.num_nodes)

            simulator = RobotSimulator(
                i, self.graph_state, initial_node, max_capacity=self.robot_max_capacity
            )
            self.robot_simulators.append(simulator)

            robot_state = create_default_robot(i, max_capacity=self.robot_max_capacity)
            telemetry = simulator.get_telemetry()
            telemetry.timestamp = self.current_time
            robot_state.update_telemetry(telemetry)
            self.robots.append(robot_state)

        self.pending_tasks = []
        self.completed_tasks = []
        self.next_task_id = 1
        self.current_time = 0.0
        self.last_inventory_check = 0.0
        self._task_battery_snapshots.clear()
        self.graph_state.current_time = self.current_time
        self._stochastic_task_timestamps = []

        new_tasks, self.next_task_id = generate_inventory_tasks(
            self.graph_state, self.current_time, self.next_task_id, existing_task_keys=set()
        )
        for task in new_tasks:
            task.source = "deterministic"
        self.pending_tasks.extend(new_tasks)

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
                    new_task.source = "factoriser"
                    self.pending_tasks.append(new_task)
                    created += 1
            if created > 0:
                self._record_stochastic_tasks(created)

        self.episode_stockouts = 0
        self.cumulative_reward = 0.0
        return self._get_state_dict()

    def step(self, dt: float = None):
        if dt is None:
            dt = self.timestep_seconds

        time_delta_hours = dt / 3600.0
        self.graph_state.current_time = self.current_time
        self._update_inventory(time_delta_hours)

        if self.current_time - self.last_inventory_check >= self.inventory_check_interval:
            covered = self._build_covered_replenishment_keys()
            new_tasks, self.next_task_id = generate_inventory_tasks(
                self.graph_state, self.current_time, self.next_task_id, existing_task_keys=covered
            )

            if new_tasks:
                for task in new_tasks:
                    task.source = "deterministic"
                self.pending_tasks.extend(new_tasks)

            self.last_inventory_check = self.current_time

        self._generate_stochastic_tasks(dt)

        self._update_edge_congestion()

        self._update_robot_positions(dt)

        completed_tasks = self._check_task_completions()

        reward = self._compute_timestep_reward(completed_tasks)
        self.cumulative_reward += reward

        self._update_edge_congestion()

        self._collect_traversal_records()

        self.current_time += dt

        done = self.current_time >= self.max_episode_time

        state_dict = self._get_state_dict()

        info = {
            "current_time": self.current_time,
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(completed_tasks),
            "total_robot_tasks": sum(r.num_queued_tasks for r in self.robots),
            "stockouts": sum(1 for n in self.graph_state.nodes if n.is_stockout),
            "cumulative_reward": self.cumulative_reward,
            "edge_cost_model": self.edge_cost_manager.get_stats(),
            "task_completion_credits": self._last_per_task_credits,
            "completed_leg_credits": self._last_completed_leg_credits,
            "episode_stockouts": self.episode_stockouts,
            "episode_break_stockouts": self.episode_break_stockouts,
            "completed_task_legs": completed_tasks,
            "emergency_tasks": len(self.emergency_tasks),
            "offline_robots": list(self._offline_robots),
            "robots_charging": [
                rid
                for rid, robot in enumerate(self.robots)
                if robot.telemetry is not None and robot.telemetry.is_charging
            ],
        }

        return state_dict, reward, done, info

    def _update_inventory(self, time_delta_hours: float):
        """Hook for inventory update each step. Override in subclasses for real deployment."""
        update_inventory_levels(self.graph_state, time_delta_hours)
        if self.sku_logger is not None:
            self.sku_logger.maybe_snapshot(self.graph_state, self.current_time)

    def _prune_stochastic_task_history(self):
        cutoff = self.current_time - 3600.0
        self._stochastic_task_timestamps = [
            t for t in self._stochastic_task_timestamps if t >= cutoff
        ]

    def _remaining_stochastic_task_budget(self) -> int:
        self._prune_stochastic_task_history()
        return max(0, self.max_stochastic_tasks_per_hour - len(self._stochastic_task_timestamps))

    def _record_stochastic_tasks(self, count: int):
        if count <= 0:
            return
        self._stochastic_task_timestamps.extend([self.current_time] * count)
        self._prune_stochastic_task_history()

    def _generate_stochastic_tasks(self, dt: float):
        """Generate stochastic tasks via the TaskCreationActor pipeline (#7/#8)."""
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
            new_task.source = "factoriser"
            self.pending_tasks.append(new_task)
            self._record_stochastic_tasks(1)
        else:
            logger.debug(
                "task_creation_actor found no eligible SKU candidates — no stochastic task generated"
            )

    def assign_task_to_robot(self, robot_id: int, task: Task) -> List[Task]:
        """Assign a task to a robot (called by policy/controller)."""
        if robot_id < 0 or robot_id >= len(self.robots):
            return []

        if robot_id in self._offline_robots:
            return []

        if self._should_robot_charge(robot_id):
            return []

        robot = self.robots[robot_id]
        simulator = self.robot_simulators[robot_id]

        was_idle = len(robot.task_queue) == 0
        assignment_stock_level = getattr(
            task, "current_sku_stock_level", getattr(task, "sku_stock_level", None)
        )
        task.sku_stock_level_at_assign = assignment_stock_level

        pickup_task = Task(
            task_id=task.task_id,
            from_location_index=task.from_location_index,
            to_location_index=task.from_location_index,
            deadline=task.deadline,
            arrival_time=task.arrival_time,
            estimated_duration=task.estimated_duration,
            task_type=task.task_type,
            num_items=task.num_items,
            time_to_stockout=task.time_to_stockout,
            initial_time_to_stockout=task.initial_time_to_stockout,
            current_time_to_stockout=task.current_time_to_stockout,
            original_deadline=task.original_deadline,
            current_deadline=task.current_deadline,
            sku_id=task.sku_id,
            category_key=task.category_key,
            category_id=task.category_id,
            sku_stock_level=task.sku_stock_level,
            initial_sku_stock_level=task.initial_sku_stock_level,
            current_sku_stock_level=task.current_sku_stock_level,
            sku_stock_level_at_assign=assignment_stock_level,
            sku_max_level=task.sku_max_level,
            reorder_point=task.reorder_point,
            par_level=task.par_level,
            demand_location_index=task.demand_location_index,
            leg_type="pickup",
            learned_score=task.learned_score,
        )

        dropoff_task = Task(
            task_id=task.task_id,
            from_location_index=task.from_location_index,
            to_location_index=task.to_location_index,
            deadline=task.deadline,
            arrival_time=task.arrival_time,
            estimated_duration=task.estimated_duration,
            task_type=task.task_type,
            num_items=task.num_items,
            time_to_stockout=task.time_to_stockout,
            initial_time_to_stockout=task.initial_time_to_stockout,
            current_time_to_stockout=task.current_time_to_stockout,
            original_deadline=task.original_deadline,
            current_deadline=task.current_deadline,
            sku_id=task.sku_id,
            category_key=task.category_key,
            category_id=task.category_id,
            sku_stock_level=task.sku_stock_level,
            initial_sku_stock_level=task.initial_sku_stock_level,
            current_sku_stock_level=task.current_sku_stock_level,
            sku_stock_level_at_assign=assignment_stock_level,
            sku_max_level=task.sku_max_level,
            reorder_point=task.reorder_point,
            par_level=task.par_level,
            demand_location_index=task.demand_location_index,
            leg_type="dropoff",
            learned_score=task.learned_score,
        )

        robot.add_task(pickup_task)
        robot.add_task(dropoff_task)

        pickup_task.is_assigned = True
        pickup_task.assigned_robot_id = robot_id
        dropoff_task.is_assigned = True
        dropoff_task.assigned_robot_id = robot_id

        if task in self.pending_tasks:
            self.pending_tasks.remove(task)

        if was_idle:
            self._plan_path_for_robot(robot, pickup_task, simulator)

        return [pickup_task, dropoff_task]

    def _plan_path_for_robot(self, robot: RobotState, task: Task, simulator):
        """Plan path for robot to complete task."""
        if task.execution_start_time is None:
            task.execution_start_time = self.current_time

        start_node = robot.current_node_index
        if start_node is None:
            start_node = self._find_nearest_node(robot.telemetry.x, robot.telemetry.y)

        path, _ = dijkstra_shortest_path(
            start_node,
            task.to_location_index,
            self.graph_state,
            len(self.graph_state.nodes),
            edge_cost_manager=self.edge_cost_manager,
            all_robots=self.robots,
        )

        # Store planned route on task so logger can record it at completion.
        # Only set once — the idle check can re-call this after arrival (trivial path),
        # which would overwrite the actual route with a single-node path.
        if not task.planned_path:
            task.planned_path = list(path)
        # Snapshot battery at leg start so completion can compute distance/energy delta.
        # Guard against post-arrival idle-check re-calls overwriting the start value.
        if (task.task_id, task.leg_type) not in self._task_battery_snapshots:
            self._task_battery_snapshots[(task.task_id, task.leg_type)] = simulator.battery_level

        simulator.set_path(path, task.task_id, task.num_items)

        simulator.set_traversal_entry_time(self.current_time)

        if simulator.current_edge_index is not None:
            approaching = self._count_approaching_robots(
                simulator.current_edge_index, exclude_robot_id=robot.robot_id
            )
            simulator.set_traversal_approaching_count(approaching)

        robot.planned_path = path
        robot.target_node_index = task.to_location_index
        robot.travel_start_time = self.current_time

    def _update_robot_positions(self, dt: float):
        """Move robots along their planned paths."""
        for robot, simulator in zip(self.robots, self.robot_simulators):
            prev_edge_idx = simulator.current_edge_index

            telemetry = simulator.update(dt)
            telemetry.timestamp = self.current_time

            robot_id = simulator.robot_id
            elapsed_minutes = (self.current_time - self.robot_reset_times[robot_id]) / 60.0
            num_nodes = len(self.graph_state.nodes) if self.graph_state else 0
            num_edges = len(self.graph_state.edges) if self.graph_state else 0
            telemetry = self.telemetry_noise_manager.apply_noise(
                telemetry, elapsed_minutes, num_nodes=num_nodes, num_edges=num_edges
            )

            if prev_edge_idx is not None and simulator.current_edge_index != prev_edge_idx:
                simulator.finalize_current_traversal(self.current_time)

            if (
                simulator.current_edge_index is not None
                and simulator.current_edge_index != prev_edge_idx
            ):
                simulator.set_traversal_entry_time(self.current_time)

            robot.update_telemetry(telemetry)

            _velocity_threshold = 0.05
            if simulator.velocity_ms < _velocity_threshold:
                robot.consecutive_idle_s += dt
                if robot.consecutive_idle_s >= 10.0 and not robot._stationary_event_fired:
                    robot._stationary_event_fired = True
                    if self.fleet_event_logger:
                        _rid = robot.robot_id
                        if simulator.is_charging:
                            _reason = "charging"
                        elif _rid in self._robots_routing_to_charge:
                            _reason = "routing_to_charge"
                        elif not robot.task_queue:
                            _reason = "no_task"
                        else:
                            _reason = "unknown"
                        self.fleet_event_logger.log_robot_stationary(
                            robot_id=_rid,
                            sim_time=self.current_time,
                            battery_level=simulator.battery_level,
                            duration_s=robot.consecutive_idle_s,
                            current_node=robot.current_node_index,
                            stoppage_reason=_reason,
                        )
            else:
                robot.consecutive_idle_s = 0.0
                robot._stationary_event_fired = False

            self._interrupt_mid_task_for_charging(simulator.robot_id, robot, simulator)

            # If robot is idle but has queued tasks, plan the next valid task
            if (
                not simulator.path_queue
                and simulator.current_target_node is None
                and robot.current_task is not None
            ):
                _max_demotes = len(robot.task_queue)
                _demotes = 0
                while (
                    _demotes < _max_demotes
                    and robot.current_task
                    and robot.current_task.leg_type == "dropoff"
                    and not robot.is_pickup_complete(robot.current_task.task_id)
                ):
                    robot.demote_current_task()
                    _demotes += 1
                if robot.current_task and not (
                    robot.current_task.leg_type == "dropoff"
                    and not robot.is_pickup_complete(robot.current_task.task_id)
                ):
                    self._plan_path_for_robot(robot, robot.current_task, simulator)

            # Trigger charging trip when robot becomes idle and needs charging
            robot_id = robot.robot_id
            if (
                robot_id not in self._robots_routing_to_charge
                and robot_id not in self._offline_robots
                and simulator.is_charging is False
                and len(robot.task_queue) == 0
                and simulator.path_queue == []
                and simulator.current_target_node is None
                and self._should_robot_charge(robot_id)
            ):
                if self.fleet_event_logger:
                    hub_idx = self._nearest_hub_index(robot_id)
                    self.fleet_event_logger.log_charge_dispatch_idle(
                        robot_id=robot_id,
                        sim_time=self.current_time,
                        battery_level=simulator.battery_level,
                        current_node=robot.current_node_index,
                        hub_node=hub_idx,
                    )
                self._send_robot_to_charge(robot_id)

    def _check_task_completions(self) -> List[Task]:
        """Check if any robots have completed their current tasks."""
        completed_tasks = []

        for robot, simulator in zip(self.robots, self.robot_simulators):
            if (
                robot.current_task
                and robot.current_node_index == robot.target_node_index
                and not simulator.path_queue
                and simulator.current_target_node is None
            ):
                task = robot.current_task

                # Refresh live demand-side stock at the exact event moment so downstream
                # logging can distinguish assignment-time stock from pickup/dropoff stock.
                if task.task_type == "replenishment" and task.sku_id and task.leg_type == "pickup":
                    self._refresh_task_inventory_context(task)
                    task.sku_stock_level_at_collection = task.current_sku_stock_level

                if task.leg_type == "pickup":
                    simulator.load_items(task.num_items)
                    robot.mark_pickup_complete(task.task_id)
                elif task.task_type == "replenishment":
                    to_node = self.graph_state.nodes[task.to_location_index]
                    if task.sku_id:
                        to_node.restock_sku(task.sku_id, task.num_items)
                        to_node.recalc_category_inventory()
                        to_node.stock_level = sum(
                            v.get("stock", 0.0) for v in to_node.sku_inventory.values()
                        )
                    else:
                        to_node.restock(task.num_items)

                    if task.sku_id:
                        self._refresh_task_inventory_context(task)
                        task.sku_stock_level_at_dropoff = task.current_sku_stock_level
                        if self.sku_logger is not None:
                            sku_data = to_node.sku_inventory.get(task.sku_id, {})
                            self.sku_logger.log_restock(
                                node_idx=task.to_location_index,
                                node_tag=getattr(to_node, "location_tag", None)
                                or f"node_{task.to_location_index}",
                                floor=getattr(to_node, "floor", None),
                                sku_id=task.sku_id,
                                category=sku_data.get("category", ""),
                                stock_after=task.sku_stock_level_at_dropoff or 0.0,
                                max_stock=float(sku_data.get("max", 0.0)),
                                par=float(sku_data.get("par", 0.0)),
                                reorder=float(sku_data.get("reorder", 0.0)),
                                sim_time=self.current_time,
                            )

                if task.leg_type == "dropoff":
                    simulator.complete_task(task.num_items)
                    robot.mark_dropoff_complete(task.task_id)

                if not getattr(task, "planned_path", None) and robot.planned_path:
                    task.planned_path = list(robot.planned_path)

                task.distance_traveled = self._path_distance_m(task.planned_path or [])
                battery_start = self._task_battery_snapshots.pop(
                    (task.task_id, task.leg_type), simulator.battery_level
                )
                battery_delta = max(0.0, battery_start - simulator.battery_level)
                task.energy_consumed_wh = battery_delta * self.BATTERY_CAPACITY_WH

                completed_task = robot.complete_current_task()
                if completed_task:
                    completed_tasks.append(completed_task)
                    if completed_task.leg_type == "dropoff":
                        self.completed_tasks.append(completed_task)

                if robot.current_task:
                    _max_demotes = len(robot.task_queue)
                    _demotes = 0
                    while (
                        _demotes < _max_demotes
                        and robot.current_task
                        and robot.current_task.leg_type == "dropoff"
                        and not robot.is_pickup_complete(robot.current_task.task_id)
                    ):
                        robot.demote_current_task()
                        _demotes += 1
                    if robot.current_task and not (
                        robot.current_task.leg_type == "dropoff"
                        and not robot.is_pickup_complete(robot.current_task.task_id)
                    ):
                        self._plan_path_for_robot(robot, robot.current_task, simulator)

            # Handle charging completion — runs BEFORE arrival so arrival cannot
            # restart charging on the same step that charging finishes.
            # Requires the robot to be at a hub/storage node so this never fires
            # while the robot is still en route (active_task_id==-1 but not yet
            # at hub) or at a non-hub destination.
            robot_id = robot.robot_id
            _at_hub = robot.current_node_index is not None and self.graph_state.nodes[
                robot.current_node_index
            ].node_type in ("hub", "storage")
            if (
                simulator.is_charging is False
                and robot_id in self._robots_routing_to_charge
                and simulator.active_task_id == -1
                and _at_hub
                and simulator.battery_level >= 0.99
            ):
                if self.fleet_event_logger:
                    self.fleet_event_logger.log_charge_complete(
                        robot_id=robot_id,
                        sim_time=self.current_time,
                        battery_level=simulator.battery_level,
                        hub_node=robot.current_node_index,
                    )
                self._robots_routing_to_charge.discard(robot_id)
                if robot.task_queue:
                    self._plan_path_for_robot(robot, robot.current_task, simulator)

            # Handle charging arrival — runs AFTER completion so a robot that
            # just finished charging is already removed from _robots_routing_to_charge
            # and won't be restarted here.
            if (
                robot_id in self._robots_routing_to_charge
                and simulator.active_task_id == -1
                and robot.current_node_index is not None
            ):
                node = self.graph_state.nodes[robot.current_node_index]
                if node.node_type in ("hub", "storage") and not simulator.is_charging:
                    simulator.start_charging()
                    if self.fleet_event_logger:
                        self.fleet_event_logger.log_charge_start(
                            robot_id=robot_id,
                            sim_time=self.current_time,
                            battery_level=simulator.battery_level,
                            hub_node=robot.current_node_index,
                        )

        return completed_tasks

    def _break_state(self):
        """Return (is_break: bool, secs_to_next_break: float) from the school schedule."""
        schedule = getattr(self.graph_state, "school_schedule", None)
        if not schedule:
            return False, float("inf")
        periods = schedule.get("periods", [])
        break_set = set(schedule.get("break_periods", []))
        if not periods:
            return False, float("inf")

        # Anchor simulation clock to first period's wall-clock start (e.g. 08:30 = 8.5h)
        first_start = periods[0]["start_time"]
        hh, mm = first_start.split(":")
        episode_start_h = int(hh) + int(mm) / 60.0
        t_sim = self.current_time % 86400  # seconds within the current simulated day

        is_break = False
        secs_to_next = float("inf")

        for period in periods:
            s_hh, s_mm = period["start_time"].split(":")
            e_hh, e_mm = period["end_time"].split(":")
            p_start_s = (int(s_hh) + int(s_mm) / 60.0 - episode_start_h) * 3600.0
            p_end_s = (int(e_hh) + int(e_mm) / 60.0 - episode_start_h) * 3600.0

            if p_start_s <= t_sim < p_end_s and period["name"] in break_set:
                is_break = True
                secs_to_next = 0.0
            elif p_start_s > t_sim and period["name"] in break_set:
                secs_to_next = min(secs_to_next, p_start_s - t_sim)

        return is_break, secs_to_next

    def _compute_timestep_reward(self, completed_tasks: List[Task]) -> float:
        """Compute reward combining inventory health + task completion + utilization."""
        self._last_per_task_credits = {}
        self._last_completed_leg_credits = {}
        total_reward = 0.0

        for task in completed_tasks:
            task_reward = 0.0

            assigned_robot_id = getattr(task, "assigned_robot_id", None)
            robot = None
            if assigned_robot_id is not None:
                for r in self.robots:
                    if r.robot_id == assigned_robot_id:
                        robot = r
                        break

            if task.task_type == "replenishment" and task.leg_type == "dropoff":
                to_node = self.graph_state.nodes[task.to_location_index]
                if (
                    task.sku_id
                    and getattr(to_node, "sku_inventory", None)
                    and task.sku_id in to_node.sku_inventory
                ):
                    sku_data = to_node.sku_inventory[task.sku_id]
                    pre_stock = max(0.0, float(sku_data["stock"]) - task.num_items)
                    pre_ratio = pre_stock / max(float(sku_data["max"]), 1.0)
                else:
                    pre_ratio = 0.0
                task_reward += 10.0 * pre_ratio

                is_break, secs_to_break = self._break_state()
                if not is_break and 0 < secs_to_break <= self.proactive_restock_lead_time_s:
                    task_reward += 8.0

            if task.leg_type == "dropoff" and robot:
                load_ratio = robot.current_load / max(robot.max_capacity, 1.0)
                task_reward += 2.0 * load_ratio

            if task.leg_type == "dropoff" and robot:
                battery = robot.telemetry.battery_level if robot.telemetry is not None else 1.0
                if battery < self.battery_critical_threshold:
                    task_reward -= self.battery_critical_task_penalty

            self._last_completed_leg_credits[(task.task_id, task.leg_type)] = task_reward
            if task.leg_type == "dropoff":
                self._last_per_task_credits[task.task_id] = task_reward
            total_reward += task_reward

        stockout_count = 0
        for node in self.graph_state.nodes:
            if not node.sku_inventory or not node.consumption_enabled:
                continue
            for sku_data in node.sku_inventory.values():
                if float(sku_data.get("stock", 0.0)) <= 0:
                    stockout_count += 1
        self.episode_stockouts = stockout_count

        is_break_now, _ = self._break_state()
        if is_break_now:
            self.episode_break_stockouts += stockout_count

        if is_break_now:
            for rid in range(len(self.robots)):
                if rid in self._offline_robots:
                    continue
                sim = self.robot_simulators[rid]
                if sim.current_edge_index is not None:
                    total_reward -= self.break_movement_penalty

        return total_reward

    def refresh_task_inventory_contexts(self) -> None:
        """Refresh live inventory-derived fields for pending and queued replenishment tasks."""
        for task in self.pending_tasks:
            self._refresh_task_inventory_context(task)

        for robot in self.robots:
            for task in robot.task_queue:
                self._refresh_task_inventory_context(task)
            for task in robot.overflow_queue:
                self._refresh_task_inventory_context(task)

    def _refresh_task_inventory_context(self, task: Task) -> None:
        if task.task_type != "replenishment" or not task.sku_id:
            return
        if self.graph_state is None or not getattr(self.graph_state, "nodes", None):
            return

        demand_idx = getattr(task, "demand_location_index", None)
        if demand_idx is None:
            demand_idx = task.to_location_index
        if demand_idx is None or demand_idx < 0 or demand_idx >= len(self.graph_state.nodes):
            return

        node = self.graph_state.nodes[demand_idx]
        sku_inventory = getattr(node, "sku_inventory", None) or {}
        sku_data = sku_inventory.get(task.sku_id)
        if sku_data is None:
            return

        stock = float(sku_data.get("stock", task.get_current_sku_stock_level()))
        max_level = float(sku_data.get("max", task.sku_max_level))
        reorder = float(sku_data.get("reorder", task.reorder_point))
        par_level = float(sku_data.get("par", task.par_level))

        rate = _compute_sku_rate(node, task.sku_id, self.graph_state, self.current_time)
        current_tts = (stock / rate) if rate > 0 else float("inf")
        current_deadline = self.current_time + max(
            current_tts * 3600.0,
            DEFAULT_MIN_REPLENISHMENT_DEADLINE_S,
        )

        task.current_time_to_stockout = current_tts
        task.current_deadline = current_deadline
        task.current_sku_stock_level = stock
        task.sku_max_level = max_level
        task.reorder_point = reorder
        task.par_level = par_level
        task.demand_location_index = demand_idx

    def get_robot_availability_mask(self, task: Task = None) -> np.ndarray:
        """Get boolean mask for which robots can accept a task right now."""
        mask = np.ones(len(self.robots), dtype=bool)
        for rid in range(len(self.robots)):
            if rid in self._offline_robots or self._should_robot_charge(rid):
                mask[rid] = False
        return mask

    def _get_state_dict(self) -> Dict[str, np.ndarray]:
        """Build graph-structured state dictionary for GAPO."""
        self.refresh_task_inventory_contexts()

        if self.pending_tasks:
            current_task = self.pending_tasks[0]
            task_features = current_task.get_features(self.current_time)
        else:
            task_features = np.zeros(10, dtype=np.float32)

        try:
            if getattr(self.graph_state, "category_order", None):
                node_continuous, node_categorical = (
                    self.graph_state.get_node_features_with_category_stats()
                )

                node_sku_features, node_sku_mask = self.graph_state.get_node_sku_features()
            else:
                node_continuous, node_categorical = self.graph_state.get_node_features_complete()

                node_sku_features, node_sku_mask = (None, None)
        except Exception:
            import traceback

            traceback.print_exc()
            raise

        if node_categorical.shape[1] != 4:
            num_nodes = node_continuous.shape[0]
            padded = np.zeros((num_nodes, 4), dtype=np.int64)
            padded[:, 1] = -1
            cols = min(node_categorical.shape[1], 4)
            padded[:, :cols] = node_categorical[:, :cols].astype(np.int64)
            node_categorical = padded

        edge_features_orig, edge_node_indices_orig = self.graph_state.get_edge_features_complete()

        edge_features = np.vstack([edge_features_orig, edge_features_orig])
        edge_node_indices = np.vstack([edge_node_indices_orig, edge_node_indices_orig[:, [1, 0]]])

        edge_index = self._build_edge_index()

        robot_features = []
        robot_positions = []

        for robot in self.robots:
            if robot.telemetry:
                heading_sin = float(np.sin(robot.telemetry.heading))
                heading_cos = float(np.cos(robot.telemetry.heading))
                max_v = 1.0
                velocity_ratio = float(robot.telemetry.velocity_ms / max_v) if max_v > 0 else 0.0
                remaining_path = robot.telemetry.remaining_path or []
                has_path = (
                    1.0
                    if (robot.telemetry.current_edge_index is not None or remaining_path)
                    else 0.0
                )
                target_node = remaining_path[0] if remaining_path else -1
                is_blocked = 1.0 if (has_path and robot.telemetry.velocity_ms < 0.05) else 0.0
                feat = [
                    robot.telemetry.x,
                    robot.telemetry.y,
                    robot.telemetry.velocity_ms,
                    robot.telemetry.battery_level,
                    float(robot.telemetry.is_moving),
                    float(
                        robot.telemetry.current_node_index
                        if robot.telemetry.current_node_index is not None
                        else -1
                    ),
                    float(
                        robot.telemetry.current_edge_index
                        if robot.telemetry.current_edge_index is not None
                        else -1
                    ),
                    robot.telemetry.edge_progress,
                    robot.telemetry.eta_to_next_node,
                    float(robot.num_queued_tasks),
                    float(robot.effective_load),  # effective load: physical + committed pickups
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

        if self.pending_tasks:
            num_pending = len(self.pending_tasks)
            avg_priority = np.mean([t.get_current_time_to_stockout() for t in self.pending_tasks])
            num_urgent = sum(
                1 for t in self.pending_tasks if t.get_current_time_to_stockout() < 1.0
            )
            oldest_age = max([t.get_age(self.current_time) for t in self.pending_tasks])
            num_near_deadline = sum(
                1 for t in self.pending_tasks if t.get_time_to_deadline(self.current_time) < 300
            )
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

        queue_features = np.array(
            [
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
                float(len(self.robots)),
            ],
            dtype=np.float32,
        )

        assert node_categorical.ndim == 2, (
            f"node_categorical must be 2D, got shape {node_categorical.shape}"
        )
        assert node_categorical.shape[1] == 4, (
            f"node_categorical must have 4 columns, got shape {node_categorical.shape}"
        )

        return {
            "task_features": task_features,
            "node_continuous": node_continuous,
            "node_categorical": node_categorical,
            "node_sku_features": node_sku_features,
            "node_sku_mask": node_sku_mask,
            "edge_features": edge_features,
            "edge_node_indices": edge_node_indices,
            "edge_index": edge_index,
            "robot_features": robot_features,
            "robot_positions": robot_positions,
            "queue_features": queue_features,
        }

    def _path_distance_m(self, path: list) -> float:
        """Sum edge distances along a planned path (list of node indices)."""
        if not path or len(path) < 2:
            return 0.0
        nodes = self.graph_state.nodes
        edge_lookup = {(e.from_node, e.to_node): e.distance_m for e in self.graph_state.edges}
        total = 0.0
        for from_idx, to_idx in zip(path, path[1:]):
            key = (nodes[from_idx].node_id, nodes[to_idx].node_id)
            total += edge_lookup.get(key, 0.0)
        return total

    def _build_edge_index(self) -> np.ndarray:
        """Build edge connectivity matrix from graph edges."""
        edge_index = []

        for edge in self.graph_state.edges:
            from_idx = self._find_node_index(edge.from_node)
            to_idx = self._find_node_index(edge.to_node)

            if from_idx is not None and to_idx is not None:
                edge_index.append([from_idx, to_idx])
                edge_index.append([to_idx, from_idx])  # Undirected

        if edge_index:
            return np.array(edge_index, dtype=np.int64).T
        else:
            return np.array([[i, i] for i in range(self.num_nodes)], dtype=np.int64).T

    def _find_node_index(self, node_id: str) -> Optional[int]:
        for i, node in enumerate(self.graph_state.nodes):
            if node.node_id == node_id:
                return i
        return None

    def _update_edge_congestion(self):
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
                    current_node = (
                        simulator.current_node_index
                        if simulator is not None
                        else robot.telemetry.current_node_index
                    )
                    target_node = (
                        simulator.current_target_node
                        if simulator is not None
                        else (
                            robot.telemetry.remaining_path[0]
                            if robot.telemetry.remaining_path
                            else None
                        )
                    )
                    edge.active_robot_progress[robot.robot_id] = (
                        robot.telemetry.edge_progress,
                        current_node,
                        target_node,
                    )
            else:
                node_idx = robot.telemetry.current_node_index
                if node_idx is not None and 0 <= node_idx < len(self.graph_state.nodes):
                    self.graph_state.nodes[node_idx].add_robot(robot.robot_id)

        for edge_idx, edge in enumerate(self.graph_state.edges):
            edge.approaching_robot_count = self._count_approaching_robots(edge_idx)

    def _find_nearest_node(self, x, y):
        min_dist = float("inf")
        nearest_idx = 0

        for i, node in enumerate(self.graph_state.nodes):
            dist = np.sqrt((node.x - x) ** 2 + (node.y - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def _collect_traversal_records(self):
        all_records = []
        for simulator in self.robot_simulators:
            records = simulator.get_and_clear_traversal_records()
            all_records.extend(records)

        if all_records:
            self.edge_cost_manager.add_traversal_records(all_records)

    def _build_covered_replenishment_keys(self) -> set:
        """Return a set of (sku_id, dest_node_idx) tuples that are already covered"""
        keys = set()

        for task in self.pending_tasks:
            if task.task_type == "replenishment":
                keys.add((task.sku_id, task.to_location_index))

        parent_to_destination = {}

        for robot in self.robots:
            for task in robot.task_queue:
                if task.task_type == "replenishment":
                    parent_id = task.task_id
                    leg_type = getattr(task, "leg_type", "full")
                    if leg_type == "dropoff":
                        parent_to_destination[parent_id] = (task.sku_id, task.to_location_index)
                    elif parent_id not in parent_to_destination:
                        parent_to_destination[parent_id] = (task.sku_id, None)

            for task in robot.overflow_queue:
                if task.task_type == "replenishment":
                    parent_id = task.task_id
                    leg_type = getattr(task, "leg_type", "full")
                    if leg_type == "dropoff":
                        parent_to_destination[parent_id] = (task.sku_id, task.to_location_index)
                    elif parent_id not in parent_to_destination:
                        parent_to_destination[parent_id] = (task.sku_id, None)

        for sku_id, dest_idx in parent_to_destination.values():
            if dest_idx is not None:
                keys.add((sku_id, dest_idx))

        for robot in self.robots:
            for task in robot.task_queue:
                if task.task_type == "replenishment":
                    if getattr(task, "leg_type", "full") == "full":
                        keys.add((task.sku_id, task.to_location_index))
            for task in robot.overflow_queue:
                if task.task_type == "replenishment":
                    if getattr(task, "leg_type", "full") == "full":
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
            path = getattr(robot, "planned_path", None)
            if not path or len(path) < 2:
                continue
            for i in range(len(path) - 1):
                from_id = self.graph_state.nodes[path[i]].node_id
                to_id = self.graph_state.nodes[path[i + 1]].node_id
                if (edge.from_node == from_id and edge.to_node == to_id) or (
                    edge.from_node == to_id and edge.to_node == from_id
                ):
                    count += 1
                    break
        return count

    def _get_hub_node_indices(self) -> list:
        """Get indices of all charge-point nodes in the graph."""
        return [
            i for i, n in enumerate(self.graph_state.nodes) if n.node_type in ("hub", "storage")
        ]

    def _should_robot_charge(self, robot_id: int) -> bool:
        """Check if robot battery is below threshold to reach nearest hub."""
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

    def _interrupt_mid_task_for_charging(self, robot_id: int, robot, simulator) -> bool:
        """If a robot is mid-task and battery drops below charging threshold,"""
        if robot_id in self._offline_robots:
            return False
        if robot_id in self._robots_routing_to_charge:
            return False
        if simulator.is_charging:
            return False
        # Only active real tasks (active_task_id == -1 means already on a charge run)
        if simulator.active_task_id is None or simulator.active_task_id == -1:
            return False
        if not self._should_robot_charge(robot_id):
            return False

        # Don't interrupt if robot just arrived at its task destination — the completion
        # check hasn't run yet. Let it complete first; idle charge logic fires after.
        if (
            robot.current_task is not None
            and robot.current_node_index is not None
            and robot.current_node_index == robot.target_node_index
            and not simulator.path_queue
            and simulator.current_target_node is None
        ):
            return False

        # Interrupt: _send_robot_to_charge overwrites simulator path_queue and
        # sets active_task_id = -1. Tasks stay in robot.task_queue so they
        # auto-resume once the charge trip completes.
        robot = self.robots[robot_id]
        simulator = self.robot_simulators[robot_id]
        if self.fleet_event_logger:
            hub_idx = self._nearest_hub_index(robot_id)
            queued_ids = [t.task_id for t in robot.task_queue]
            self.fleet_event_logger.log_charge_interrupt(
                robot_id=robot_id,
                sim_time=self.current_time,
                battery_level=simulator.battery_level,
                interrupted_task_id=simulator.active_task_id,
                queued_task_ids=queued_ids,
                current_node=robot.current_node_index,
                hub_node=hub_idx,
            )
        self._send_robot_to_charge(robot_id)
        return True

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
            start_node,
            hub_idx,
            self.graph_state,
            len(self.graph_state.nodes),
            edge_cost_manager=self.edge_cost_manager,
            all_robots=self.robots,
        )
        simulator.set_path(path, task_id=-1, num_items=0)
        self._robots_routing_to_charge.add(robot_id)

    def mark_robot_offline(self, robot_id: int):
        """Mark robot as manually powered off."""
        if robot_id < 0 or robot_id >= len(self.robots):
            return
        robot = self.robots[robot_id]
        simulator = self.robot_simulators[robot_id]

        in_transit_parents = set(robot.picked_up_task_ids)

        requeued_ids = []
        emergency_ids = []
        for task in robot.task_queue:
            parent_id = task.task_id
            if parent_id in in_transit_parents:
                task.task_type = "emergency_manual"
                self.emergency_tasks.append(task)
                emergency_ids.append(task.task_id)
            else:
                self.pending_tasks.insert(0, task)
                requeued_ids.append(task.task_id)

        if self.fleet_event_logger:
            self.fleet_event_logger.log_robot_offline(
                robot_id=robot_id,
                sim_time=self.current_time,
                battery_level=simulator.battery_level,
                current_node=robot.current_node_index,
                requeued_task_ids=requeued_ids,
                emergency_task_ids=emergency_ids,
            )

        robot.task_queue.clear()
        robot.overflow_queue.clear()
        robot.picked_up_task_ids.clear()
        robot.planned_path.clear()
        robot.target_node_index = None

        simulator.path_queue.clear()
        simulator.current_target_node = None
        simulator.velocity_ms = 0.0
        simulator.active_task_id = None
        simulator.is_charging = False

        self._offline_robots.add(robot_id)
        self._robots_routing_to_charge.discard(robot_id)

    def bring_robot_online(self, robot_id: int):
        """Mark robot as powered back on."""
        self._offline_robots.discard(robot_id)
