from typing import List

from src.exceptions import TelemetryUnavailableError

from .gapo_env import GAPOTaskAssignmentEnv
from .graph_helpers import dijkstra_shortest_path
from .robot.robot_telemetry import RobotTelemetry
from .tasks.task_state import Task


class GAPOTaskAssignmentEnvReal(GAPOTaskAssignmentEnv):
    """GAPOTaskAssignmentEnv variant for real robot deployment via ROS bridge."""

    def __init__(self, robot_backend, **kwargs):
        super().__init__(**kwargs)
        self.robot_backend = robot_backend
        # Matches RobotSimulator.battery_drain_rate default (0.001 per metre).
        self._real_battery_drain_rate = 0.001
        self._real_robots_charging: set = set()
        # Rides out transient Nav2 failures by retrying a few times before aborting.
        self._task_failure_counts: dict = {}
        self._max_task_retries: int = 3

    def reset(self):
        super().reset()

        # assign_task_to_robot() indexes into robot_simulators to pass one to
        # _plan_path_for_robot(), which we override and ignore that arg, so None is safe.
        self.robot_simulators = [None] * self.num_robots

        self.robot_backend.set_graph_state(self.graph_state)

        for robot in self.robots:
            raw = self.robot_backend.get_telemetry(robot.robot_id)
            if raw is None:
                raise TelemetryUnavailableError(
                    f"No telemetry received from ROS bridge for robot_id={robot.robot_id}. "
                    "Ensure the bridge is connected and publishing before calling reset()."
                )

            robot.update_telemetry(self._bridge_to_env_telemetry(robot, raw))

        return self._get_state_dict()

    def _update_inventory(self, time_delta_hours: float):
        """Overwrite node stock levels from bridge location_inventories (sensor readings, no consumption math)."""
        inventories = self.robot_backend.get_location_inventories()
        if not inventories:
            return  # Bridge not connected or no system_state received yet

        for node in self.graph_state.nodes:
            inv_data = inventories.get(node.node_id)
            if inv_data is None or not node.sku_inventory:
                continue
            for sku_id, bridge_sku in inv_data.sku_inventory.items():
                if sku_id in node.sku_inventory:
                    # Bridge sends 'stock_level'; node stores under 'stock'
                    node.sku_inventory[sku_id]["stock"] = float(bridge_sku.get("stock_level", 0.0))
            node.recalc_category_inventory()
            node.stock_level = sum(v.get("stock", 0.0) for v in node.sku_inventory.values())

        # Mirrors sim-mode behavior so post-run analytics has time-series inventory data.
        if self.sku_logger is not None:
            self.sku_logger.maybe_snapshot(self.graph_state, self.current_time)

    def _update_robot_positions(self, dt: float):
        """Pull each robot's state from backend telemetry; no simulator update, noise, or traversal stamps."""
        for robot in self.robots:
            raw = self.robot_backend.get_telemetry(robot.robot_id)
            if raw is None:
                continue
            robot.update_telemetry(self._bridge_to_env_telemetry(robot, raw))

        for robot in self.robots:
            rid = robot.robot_id
            if (
                rid in self._offline_robots
                or rid in self._robots_routing_to_charge
                or rid in self._real_robots_charging
            ):
                continue
            if not self._should_robot_charge(rid):
                continue

            if self.fleet_event_logger:
                hub_idx = self._nearest_hub_index(rid)
                if robot.task_queue:
                    self.fleet_event_logger.log_charge_interrupt(
                        robot_id=rid,
                        sim_time=self.current_time,
                        battery_level=robot.battery_level,
                        interrupted_task_id=(
                            robot.current_task.task_id if robot.current_task else None
                        ),
                        queued_task_ids=[t.task_id for t in robot.task_queue],
                        current_node=robot.current_node_index,
                        hub_node=hub_idx,
                    )
                else:
                    self.fleet_event_logger.log_charge_dispatch_idle(
                        robot_id=rid,
                        sim_time=self.current_time,
                        battery_level=robot.battery_level,
                        current_node=robot.current_node_index,
                        hub_node=hub_idx,
                    )
            self._send_robot_to_charge(rid)

    def _check_task_completions(self) -> List[Task]:
        """Detect completed tasks via the bridge's async completion event queue and run base-class leg-tracking logic."""
        completed_tasks = []

        for robot in self.robots:
            for task_id in self.robot_backend.pop_completed_tasks(robot.robot_id):
                if task_id == -1:
                    # Charge-trip navigation finished → robot has docked at the hub.
                    # We use the bridge's own completion event as the arrival signal
                    # instead of snapping (x, y) to the nearest node. Only honoured
                    # for robots we actually dispatched to charge.
                    if (
                        robot.robot_id in self._robots_routing_to_charge
                        and robot.robot_id not in self._real_robots_charging
                    ):
                        self._real_robots_charging.add(robot.robot_id)
                        if self.fleet_event_logger:
                            self.fleet_event_logger.log_charge_start(
                                robot_id=robot.robot_id,
                                sim_time=self.current_time,
                                battery_level=(
                                    robot.telemetry.battery_level if robot.telemetry else 0.0
                                ),
                                hub_node=robot.current_node_index,
                            )
                    continue
                if not robot.current_task or robot.current_task.task_id != task_id:
                    continue

                task = robot.current_task

                # Pickup / restock logic — identical to base class
                if task.task_type == "replenishment" and task.sku_id and task.leg_type == "pickup":
                    self._refresh_task_inventory_context(task)
                    task.sku_stock_level_at_collection = task.current_sku_stock_level

                if task.leg_type == "pickup":
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
                                node_tag=(
                                    getattr(to_node, "location_tag", None)
                                    or getattr(to_node, "department_tag", None)
                                    or f"node_{task.to_location_index}"
                                ),
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
                    robot.mark_dropoff_complete(task.task_id)

                completed_task = robot.complete_current_task()
                if completed_task:
                    completed_task.distance_traveled = self._path_distance_m(
                        completed_task.planned_path or []
                    )
                    battery_start = self._task_battery_snapshots.pop(
                        (completed_task.task_id, completed_task.leg_type), None
                    )
                    if battery_start is not None and robot.telemetry:
                        battery_delta = max(0.0, battery_start - robot.telemetry.battery_level)
                        completed_task.energy_consumed_wh = battery_delta * self.BATTERY_CAPACITY_WH
                    completed_tasks.append(completed_task)
                    if completed_task.leg_type == "dropoff":
                        self.completed_tasks.append(completed_task)

                # Plan next task with same demote guard as base class
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
                        self._plan_path_for_robot(robot, robot.current_task, None)

        # Navigation-failure recovery. Drain failed/canceled goals and either
        # reset charge state (failed charge trip), ignore an expected charge
        # preemption, or retry/abort a genuinely failed real task.
        for robot in self.robots:
            for task_id in self.robot_backend.pop_failed_tasks(robot.robot_id):
                if task_id == -1:
                    # Charge-trip navigation failed/aborted → clear charge state so
                    # the next dispatch can re-route the robot to a hub.
                    self._robots_routing_to_charge.discard(robot.robot_id)
                    self._real_robots_charging.discard(robot.robot_id)
                    continue
                if robot.robot_id in self._robots_routing_to_charge:
                    # Expected preemption: we diverted this robot to charge, so Nav2
                    # aborted its in-flight real task. The task stays queued and
                    # resumes after charging — not a genuine failure.
                    continue
                self._handle_failed_task(robot, task_id)

        # Charge START is event-driven (handled above, via the bridge's task_id=-1
        # completion). Charge COMPLETE is battery-driven: a scalar threshold on
        # telemetry, with no fragile nearest-node snap required.
        for robot in self.robots:
            robot_id = robot.robot_id
            if robot_id not in self._real_robots_charging:
                continue
            if robot.telemetry is None:
                continue

            battery = robot.telemetry.battery_level
            if battery >= 0.99:
                self._real_robots_charging.discard(robot_id)
                self._robots_routing_to_charge.discard(robot_id)
                if self.fleet_event_logger:
                    self.fleet_event_logger.log_charge_complete(
                        robot_id=robot_id,
                        sim_time=self.current_time,
                        battery_level=battery,
                        hub_node=robot.current_node_index,
                    )
                if robot.current_task:
                    self._plan_path_for_robot(robot, robot.current_task, None)

        return completed_tasks

    def _handle_failed_task(self, robot, task_id: int) -> None:
        """Retry a failed navigation up to _max_task_retries, then drop the task's legs and free the robot."""
        # Act only on the robot's active task; ignore stale failures for tasks
        # that have already completed or been removed.
        if not robot.current_task or robot.current_task.task_id != task_id:
            return

        count = self._task_failure_counts.get(task_id, 0) + 1
        self._task_failure_counts[task_id] = count

        if count <= self._max_task_retries:
            if self.fleet_event_logger:
                self.fleet_event_logger.log_task_failed(
                    robot_id=robot.robot_id,
                    sim_time=self.current_time,
                    task_id=task_id,
                    action="retry",
                    retry_count=count,
                    current_node=robot.current_node_index,
                )
            self._plan_path_for_robot(robot, robot.current_task, None)
            return

        if self.fleet_event_logger:
            self.fleet_event_logger.log_task_failed(
                robot_id=robot.robot_id,
                sim_time=self.current_time,
                task_id=task_id,
                action="abort",
                retry_count=count,
                current_node=robot.current_node_index,
            )
        robot.task_queue[:] = [t for t in robot.task_queue if t.task_id != task_id]
        robot.overflow_queue[:] = [t for t in robot.overflow_queue if t.task_id != task_id]
        robot.picked_up_task_ids.discard(task_id)
        self._task_failure_counts.pop(task_id, None)
        self._task_battery_snapshots.pop((task_id, "pickup"), None)
        self._task_battery_snapshots.pop((task_id, "dropoff"), None)

        if robot.current_task:
            self._plan_path_for_robot(robot, robot.current_task, None)
        else:
            robot.planned_path = []
            robot.target_node_index = None

    def _plan_path_for_robot(self, robot, task, simulator):
        """Run Dijkstra and dispatch the path to the real robot via backend; simulator is unused (base-class signature)."""
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

        self.robot_backend.send_path_command(robot.robot_id, path, task.task_id, task.num_items)

        task.planned_path = path
        robot.planned_path = path
        robot.target_node_index = task.to_location_index
        robot.travel_start_time = self.current_time

        if (task.task_id, task.leg_type) not in self._task_battery_snapshots:
            battery = robot.telemetry.battery_level if robot.telemetry else 1.0
            self._task_battery_snapshots[(task.task_id, task.leg_type)] = battery

    def _collect_traversal_records(self):
        pass  # RealTraversalTracker not yet implemented

    def _should_robot_charge(self, robot_id: int) -> bool:
        """Battery threshold check using telemetry battery level, instead of the simulator object the base class reads."""
        if robot_id in self._offline_robots:
            return False
        robot = self.robots[robot_id]
        hub_indices = self._get_hub_node_indices()
        if not hub_indices:
            return False

        from .graph_helpers import estimate_travel_distance

        min_dist = min(estimate_travel_distance(robot, h, self.graph_state) for h in hub_indices)
        battery_needed = min_dist * self._real_battery_drain_rate * self.battery_safety_margin
        return robot.battery_level <= battery_needed

    def _send_robot_to_charge(self, robot_id: int):
        robot = self.robots[robot_id]
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
        # task_id=-1 signals a charge trip (same convention as base class)
        self.robot_backend.send_path_command(robot_id, path, task_id=-1, num_items=0)
        self._robots_routing_to_charge.add(robot_id)

    def _bridge_to_env_telemetry(self, robot, raw) -> RobotTelemetry:
        """Convert backend RobotTelemetryData to env RobotTelemetry, inferring graph-aware fields from (x, y) since Nav2 has no topology."""
        node_idx = raw.current_node_index
        if node_idx is None:
            node_idx = self._find_nearest_node(raw.x, raw.y)

        # Project onto planned path edges to recover edge/progress state
        path = robot.planned_path
        if len(path) >= 2:
            edge_idx, edge_prog = self._find_nearest_edge(
                raw.x, raw.y, candidate_node_pairs=list(zip(path, path[1:]))
            )
        else:
            edge_idx, edge_prog = None, 0.0

        # If robot is on an edge, current_node_index is the from-node of that edge
        if edge_idx is not None:
            edge = self.graph_state.edges[edge_idx]
            nodes = self.graph_state.nodes
            for i, node in enumerate(nodes):
                if node.node_id == edge.from_node:
                    node_idx = i
                    break

        remaining_path = list(path)
        if node_idx is not None and path:
            for i, n in enumerate(path):
                if n == node_idx:
                    remaining_path = path[i:]
                    break

        return RobotTelemetry(
            timestamp=raw.timestamp,
            robot_id=raw.robot_id,
            x=raw.x,
            y=raw.y,
            heading=raw.heading,
            current_node_index=node_idx,
            current_edge_index=edge_idx,
            edge_progress=edge_prog,
            velocity_ms=raw.velocity_ms,
            is_moving=raw.velocity_ms > 0.01,
            battery_level=raw.battery_level,
            current_capacity=raw.current_capacity,
            is_available=raw.is_available,
            active_task_id=raw.active_task_id,
            remaining_path=remaining_path,
            eta_to_next_node=raw.eta_to_next_node,
            is_charging=(robot.robot_id in self._real_robots_charging),
            needs_charging=False,
        )

    def _find_nearest_edge(self, x: float, y: float, candidate_node_pairs=None):
        """Project (x, y) onto graph edges; returns (None, 0.0) if close enough to an endpoint to count as at-node."""
        AT_NODE_THRESHOLD_M = 0.5  # within this distance of an endpoint → at-node
        ON_EDGE_TOLERANCE_M = 1.5  # max perpendicular distance to count as on-edge

        nodes = self.graph_state.nodes
        edges = self.graph_state.edges

        if candidate_node_pairs is not None:
            check_indices = []
            for from_idx, to_idx in candidate_node_pairs:
                from_id = nodes[from_idx].node_id
                to_id = nodes[to_idx].node_id
                for i, edge in enumerate(edges):
                    if (edge.from_node == from_id and edge.to_node == to_id) or (
                        edge.from_node == to_id and edge.to_node == from_id
                    ):
                        check_indices.append(i)
        else:
            check_indices = range(len(edges))

        best_edge_idx = None
        best_progress = 0.0
        best_dist = float("inf")

        for edge_idx in check_indices:
            edge = edges[edge_idx]
            if edge.entry_point is None or edge.exit_point is None:
                continue
            if not edge.is_point_on_corridor(x, y, tolerance=ON_EDGE_TOLERANCE_M):
                continue

            progress = edge.compute_progress(x, y)

            # Distance from (x,y) to the closest point on the segment
            sx, sy = edge.entry_point
            ex, ey = edge.exit_point
            cx = sx + progress * (ex - sx)
            cy = sy + progress * (ey - sy)
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_edge_idx = edge_idx
                best_progress = progress

        if best_edge_idx is None:
            return None, 0.0

        # If within AT_NODE_THRESHOLD of either endpoint, treat as at-node
        edge = edges[best_edge_idx]
        if edge.distance_m > 0:
            at_node_progress = AT_NODE_THRESHOLD_M / edge.distance_m
            if best_progress < at_node_progress or best_progress > 1.0 - at_node_progress:
                return None, 0.0

        return best_edge_idx, best_progress
