"""
Real-robot deployment environment.

Subclasses GAPOTaskAssignmentEnv, overriding only the methods that differ
between simulation and real deployment:

  _update_inventory        — sync from bridge instead of simulated consumption
  _update_robot_positions  — poll backend telemetry instead of simulator.update()
  _check_task_completions  — drain pop_completed_tasks() instead of node-arrival check
  _plan_path_for_robot     — send_path_command() instead of simulator.set_path()
  _collect_traversal_records — no-op (RealTraversalTracker is future work)
  _should_robot_charge     — use telemetry battery level, no simulator reference
  _send_robot_to_charge    — send_path_command() instead of simulator.set_path()
  _find_nearest_edge       — project (x,y) onto planned path edges → (edge_idx, progress)

All reward, state-dict, task-generation, Dijkstra, and GNN logic is inherited
unchanged from the base class. Sim training is not affected.
"""

from typing import List, Optional

from .gapo_env import GAPOTaskAssignmentEnv
from .robot.robot_telemetry import RobotTelemetry
from .tasks.task_state import Task
from .graph_helpers import dijkstra_shortest_path
from src.exceptions import TelemetryUnavailableError


class GAPOTaskAssignmentEnvReal(GAPOTaskAssignmentEnv):
    """
    GAPOTaskAssignmentEnv variant for real robot deployment via ROS bridge.

    Args:
        robot_backend: ROSBridgeRobotBackend instance (must be started before
                       calling reset() so telemetry is available)
        **kwargs: Forwarded to GAPOTaskAssignmentEnv.__init__()
    """

    def __init__(self, robot_backend, **kwargs):
        super().__init__(**kwargs)
        self.robot_backend = robot_backend
        # Fixed drain rate estimate for battery threshold calculations.
        # Matches RobotSimulator.battery_drain_rate default (0.001 per metre).
        self._real_battery_drain_rate = 0.001

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        state = super().reset()

        # Replace simulator list with None sentinels.
        # assign_task_to_robot() indexes into robot_simulators to get a
        # simulator to pass to _plan_path_for_robot(). We override
        # _plan_path_for_robot() and ignore that arg, so None is safe.
        self.robot_simulators = [None] * self.num_robots

        # Give the backend graph context so waypoint coords can be resolved.
        self.robot_backend.set_graph_state(self.graph_state)

        # Re-init robot telemetry from bridge if data is already available.
        for robot in self.robots:
            raw = self.robot_backend.get_telemetry(robot.robot_id)
            if raw is None:
                raise TelemetryUnavailableError(
                    f"No telemetry received from ROS bridge for robot_id={robot.robot_id}. "
                    "Ensure the bridge is connected and publishing before calling reset()."
                )
            
            robot.update_telemetry(self._bridge_to_env_telemetry(robot, raw))

        return self._get_state_dict()

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def _update_inventory(self, time_delta_hours: float):
        """
        Overwrite node stock levels from bridge location_inventories.

        The bridge publishes actual sensor readings at 1 Hz via system_state
        messages. We overwrite node.sku_inventory['stock'] directly — no
        consumption math. time_delta_hours is accepted for interface
        compatibility but ignored.

        If no bridge data has arrived yet, the node state is left unchanged
        so the env can still step without crashing.
        """
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
                    node.sku_inventory[sku_id]['stock'] = float(
                        bridge_sku.get('stock_level', 0.0)
                    )
            node.recalc_category_inventory()
            node.stock_level = sum(
                v.get('stock', 0.0) for v in node.sku_inventory.values()
            )

    # ------------------------------------------------------------------
    # Robot positions
    # ------------------------------------------------------------------

    def _update_robot_positions(self, dt: float):
        """
        Update robot state from backend telemetry each step.

        No simulator update, noise injection, or traversal stamps.
        _update_edge_congestion() in the base class rebuilds edge/node
        occupancy from telemetry immediately after this call.
        """
        for robot in self.robots:
            raw = self.robot_backend.get_telemetry(robot.robot_id)
            if raw is None:
                continue
            robot.update_telemetry(self._bridge_to_env_telemetry(robot, raw))

    # ------------------------------------------------------------------
    # Task completions
    # ------------------------------------------------------------------

    def _check_task_completions(self) -> List[Task]:
        """
        Detect completed tasks via the bridge completion event queue.

        pop_completed_tasks() drains task IDs that arrived asynchronously
        from the bridge since the last step. We look up each ID in the
        robot's current task and run the same restock/leg-tracking logic
        as the base class.
        """
        completed_tasks = []

        for robot in self.robots:
            for task_id in self.robot_backend.pop_completed_tasks(robot.robot_id):
                if not robot.current_task or robot.current_task.task_id != task_id:
                    continue

                task = robot.current_task

                # Pickup / restock logic — identical to base class
                if task.task_type == 'replenishment' and task.sku_id and task.leg_type == "pickup":
                    self._refresh_task_inventory_context(task)
                    task.sku_stock_level_at_collection = task.current_sku_stock_level

                if task.leg_type == "pickup":
                    robot.mark_pickup_complete(task.parent_task_id)
                elif task.task_type == 'replenishment':
                    to_node = self.graph_state.nodes[task.to_location_index]
                    if task.sku_id:
                        to_node.restock_sku(task.sku_id, task.num_items)
                        to_node.recalc_category_inventory()
                        to_node.stock_level = sum(
                            v.get('stock', 0.0)
                            for v in to_node.category_inventory.values()
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
                                node_tag=getattr(to_node, 'location_tag', None) or f"node_{task.to_location_index}",
                                floor=getattr(to_node, 'floor', None),
                                sku_id=task.sku_id,
                                category=sku_data.get('category', ''),
                                stock_after=task.sku_stock_level_at_dropoff or 0.0,
                                max_stock=float(sku_data.get('max', 0.0)),
                                par=float(sku_data.get('par', 0.0)),
                                reorder=float(sku_data.get('reorder', 0.0)),
                                sim_time=self.current_time,
                                consumption_rate=float(sku_data.get('rate', 0.0)),
                            )

                if task.leg_type == "dropoff":
                    robot.mark_dropoff_complete(task.parent_task_id)

                completed_task = robot.complete_current_task()
                if completed_task:
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
                        and not robot.is_pickup_complete(
                            robot.current_task.parent_task_id
                        )
                    ):
                        robot.demote_current_task()
                        _demotes += 1
                    if robot.current_task and not (
                        robot.current_task.leg_type == "dropoff"
                        and not robot.is_pickup_complete(
                            robot.current_task.parent_task_id
                        )
                    ):
                        self._plan_path_for_robot(robot, robot.current_task, None)

        return completed_tasks

    # ------------------------------------------------------------------
    # Path planning
    # ------------------------------------------------------------------

    def _plan_path_for_robot(self, robot, task, simulator):
        """
        Run Dijkstra and dispatch path to the real robot via backend.

        simulator arg is accepted for interface compatibility (base class
        passes it through assign_task_to_robot) but is not used here.
        """
        start_node = robot.current_node_index
        if start_node is None:
            start_node = self._find_nearest_node(
                robot.telemetry.x, robot.telemetry.y
            )

        path, _ = dijkstra_shortest_path(
            start_node,
            task.to_location_index,
            self.graph_state,
            len(self.graph_state.nodes),
            edge_cost_manager=self.edge_cost_manager,
            all_robots=self.robots,
        )

        self.robot_backend.send_path_command(
            robot.robot_id, path, task.task_id, task.num_items
        )

        robot.planned_path = path
        robot.target_node_index = task.to_location_index
        robot.travel_start_time = self.current_time

    # ------------------------------------------------------------------
    # Traversal records (future work)
    # ------------------------------------------------------------------

    def _collect_traversal_records(self):
        pass  # RealTraversalTracker not yet implemented

    # ------------------------------------------------------------------
    # Battery management
    # ------------------------------------------------------------------

    def _should_robot_charge(self, robot_id: int) -> bool:
        """
        Battery threshold check using telemetry battery level.

        Replaces base class implementation which reads battery_level and
        battery_drain_rate from the simulator object.
        """
        if robot_id in self._offline_robots:
            return False
        robot = self.robots[robot_id]
        hub_indices = self._get_hub_node_indices()
        if not hub_indices:
            return False

        from .graph_helpers import estimate_travel_distance
        min_dist = min(
            estimate_travel_distance(robot, h, self.graph_state)
            for h in hub_indices
        )
        battery_needed = (
            min_dist * self._real_battery_drain_rate * self.battery_safety_margin
        )
        return robot.battery_level <= battery_needed

    def _send_robot_to_charge(self, robot_id: int):
        """Route robot to nearest hub via backend path command."""
        robot = self.robots[robot_id]
        hub_idx = self._nearest_hub_index(robot_id)

        start_node = robot.current_node_index
        if start_node is None:
            start_node = self._find_nearest_node(
                robot.telemetry.x, robot.telemetry.y
            )

        path, _ = dijkstra_shortest_path(
            start_node,
            hub_idx,
            self.graph_state,
            len(self.graph_state.nodes),
            edge_cost_manager=self.edge_cost_manager,
            all_robots=self.robots,
        )
        # task_id=-1 signals a charge trip (same convention as base class)
        self.robot_backend.send_path_command(
            robot_id, path, task_id=-1, num_items=0
        )
        self._robots_routing_to_charge.add(robot_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bridge_to_env_telemetry(self, robot, raw) -> RobotTelemetry:
        """
        Convert RobotTelemetryData (backend) to RobotTelemetry (env).

        Nav2 does not expose graph topology, so graph-aware fields are
        inferred from (x, y) coordinates:
          current_node_index — nearest-node snap from (x, y)
          current_edge_index — projected from (x, y) onto planned path edges
          edge_progress      — scalar progress along that edge (0→1)
          remaining_path     — robot.planned_path (full path, not trimmed)
        """
        node_idx = raw.current_node_index
        if node_idx is None:
            node_idx = self._find_nearest_node(raw.x, raw.y)

        # Project onto planned path edges to recover edge/progress state
        path = robot.planned_path
        if len(path) >= 2:
            edge_idx, edge_prog = self._find_nearest_edge(
                raw.x, raw.y,
                candidate_node_pairs=list(zip(path, path[1:]))
            )
        else:
            edge_idx, edge_prog = None, 0.0

        # If robot is on an edge, current_node_index is the from-node of that edge
        if edge_idx is not None:
            edge = self.graph_state.edges[edge_idx]
            nodes = self.graph_state.nodes
            # find which end of the edge is the from-node (lower progress end)
            for i, node in enumerate(nodes):
                if node.node_id == edge.from_node:
                    node_idx = i
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
            remaining_path=list(path),
            eta_to_next_node=raw.eta_to_next_node,
            is_charging=False,
            needs_charging=False,
        )

    def _find_nearest_edge(self, x: float, y: float, candidate_node_pairs=None):
        """
        Project (x, y) onto graph edges and return the closest match.

        Args:
            x, y: Robot position in map frame.
            candidate_node_pairs: list of (from_node_idx, to_node_idx) integer
                pairs taken from planned_path. Only these edges are checked.
                If None, all graph edges are searched.

        Returns:
            (edge_index, progress) where edge_index is the position in
            graph_state.edges and progress is 0→1 along that edge.
            Returns (None, 0.0) if the robot is close enough to a node to
            be treated as at-node rather than mid-edge.
        """
        AT_NODE_THRESHOLD_M = 0.5   # within this distance of an endpoint → at-node
        ON_EDGE_TOLERANCE_M = 1.5   # max perpendicular distance to count as on-edge

        nodes = self.graph_state.nodes
        edges = self.graph_state.edges

        # Build the set of edge indices to check
        if candidate_node_pairs is not None:
            check_indices = []
            for from_idx, to_idx in candidate_node_pairs:
                from_id = nodes[from_idx].node_id
                to_id = nodes[to_idx].node_id
                for i, edge in enumerate(edges):
                    if ((edge.from_node == from_id and edge.to_node == to_id) or
                            (edge.from_node == to_id and edge.to_node == from_id)):
                        check_indices.append(i)
        else:
            check_indices = range(len(edges))

        best_edge_idx = None
        best_progress = 0.0
        best_dist = float('inf')

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
