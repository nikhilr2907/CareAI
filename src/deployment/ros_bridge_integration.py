"""
ROS Bridge Integration - Integrate ROS bridge data into CareRobotics system.

Handles:
- Updating robot state from bridge telemetry
- Updating location inventory from bridge updates
- Updating consumption rates from bridge
- Synchronizing task status updates
- Integration hooks for task allocation

Status: READY TO USE (not active until Nav2 is deployed)
"""

import asyncio
import logging
import math
import time
from typing import Optional, Dict, Callable, Set
from datetime import datetime

from .ros_bridge_client import ROSBridgeClient, RobotTelemetryData, LocationInventoryData, LocationConsumptionData
from ..environment.robot.robot_state import RobotState
from ..environment.robot.robot_telemetry import RobotTelemetry
from ..environment.graph.graph_state import GraphState


logger = logging.getLogger(__name__)


class ROSBridgeIntegration:
    """
    Integrates ROS bridge data into the CareRobotics environment.

    Provides hooks to:
    - Update robot state
    - Update location inventories
    - Update consumption rates
    - Handle task completions
    """

    def __init__(self, client: ROSBridgeClient, graph_state: GraphState):
        """
        Initialize integration.

        Args:
            client: ROSBridgeClient instance
            graph_state: GraphState instance (for location info)
        """
        self.client = client
        self.graph_state = graph_state

        # Callbacks for task allocation system to override behavior
        self.on_robot_position_updated: Optional[Callable] = None
        self.on_inventory_updated: Optional[Callable] = None
        self.on_consumption_updated: Optional[Callable] = None
        self.on_task_completed: Optional[Callable] = None
        self.on_task_failed: Optional[Callable] = None

        # Lifecycle and validation state
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None
        self._known_task_ids: Set[str] = set()  # Track which tasks we've submitted
        self.robot_states: Dict[int, any] = {}

        # Register handlers with client
        self.client.on_robot_telemetry = self._handle_robot_telemetry
        self.client.on_task_status = self._handle_task_status
        self.client.on_location_inventory = self._handle_location_inventory
        self.client.on_consumption_rates = self._handle_consumption_rates
        self.client.on_system_state = self._handle_system_state

        logger.info("ROS Bridge Integration initialized")

    async def start(self, robot_states: Dict[int, RobotState], robot_id: int = 0):
        """
        Start the ROS bridge client and listen for updates with auto-reconnection.

        Args:
            robot_states: Dict of {robot_id: RobotState}
            robot_id: Which robot to integrate (for multi-robot future, default 0)
        """
        self.robot_states = robot_states
        self._running = True
        self._known_task_ids = set()  # Reset task tracking

        # Start listening with reconnection logic
        self._listen_task = asyncio.create_task(self._listen_with_reconnect())
        logger.info(f"ROS Bridge integration started with auto-reconnection (robot_id={robot_id})")

    async def stop(self):
        """Stop listening and disconnect."""
        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        await self.client.disconnect()
        logger.info("ROS Bridge integration stopped")

    async def _handle_robot_telemetry(self, telemetry: RobotTelemetryData):
        """Update robot state from bridge telemetry."""
        try:
            # Validate message before processing
            if not self._validate_robot_telemetry(telemetry):
                return

            # Check for stale telemetry
            if self._is_telemetry_stale(telemetry.timestamp):
                logger.warning(
                    f"Stale telemetry ({time.time() - telemetry.timestamp:.1f}s old), skipping"
                )
                return

            # MULTI_ROBOT_TODO: Extract robot_id from telemetry once bridge includes it
            # For now, we assume single robot (robot_id=0)
            robot_id = 0

            if robot_id not in self.robot_states:
                logger.warning(f"Robot {robot_id} not in robot_states")
                return

            robot_state = self.robot_states[robot_id]

            # Extract velocity
            velocity_linear_x = telemetry.velocity.get('linear_x', 0.0) if telemetry.velocity else 0.0

            # Resolve the three TODOs:
            # 1. Convert quaternion to heading (yaw)
            heading = self._quaternion_to_heading(telemetry.orientation)

            # 2. Find nearest node in graph
            current_node_index = self._find_nearest_node_index(
                telemetry.position[0], telemetry.position[1]
            )

            # 3. Determine availability: active_task_id is None AND velocity is near zero
            is_available = (
                telemetry.active_task_id is None and abs(velocity_linear_x) < 0.01
            )

            # Create RobotTelemetry object (matches your existing format)
            ros_telemetry = RobotTelemetry(
                timestamp=telemetry.timestamp,
                robot_id=robot_id,
                x=telemetry.position[0],
                y=telemetry.position[1],
                heading=heading,
                current_node_index=current_node_index,
                current_edge_index=None,
                edge_progress=0.0,
                velocity_ms=velocity_linear_x,
                is_moving=velocity_linear_x > 0.01,
                battery_level=telemetry.battery_level,
                current_capacity=telemetry.current_capacity,
                is_available=is_available,
                active_task_id=telemetry.active_task_id,
                remaining_path=[],
                eta_to_next_node=0.0,
            )

            # Update robot state
            robot_state.update_telemetry(ros_telemetry)

            # Call integration callback
            if self.on_robot_position_updated:
                await self._call_callback(
                    self.on_robot_position_updated,
                    {
                        "robot_id": robot_id,
                        "position": telemetry.position,
                        "battery": telemetry.battery_level,
                    },
                )

            logger.debug(
                f"Updated robot {robot_id} telemetry: pos=({telemetry.position[0]:.2f}, "
                f"{telemetry.position[1]:.2f}), heading={heading:.2f}, available={is_available}"
            )

        except Exception as e:
            logger.error(f"Error handling robot telemetry: {e}", exc_info=True)

    async def _handle_task_status(self, data: Dict):
        """Handle task status updates from bridge."""
        try:
            task_id = data.get("task_id")
            status = data.get("status")

            # Warn if task_id was not previously registered
            if task_id not in self._known_task_ids:
                logger.debug(
                    f"task_status for unknown task_id={task_id} — may be stale or orphaned"
                )

            if status == "completed":
                logger.info(f"Task {task_id} completed on bridge")
                if self.on_task_completed:
                    await self._call_callback(
                        self.on_task_completed, {"task_id": task_id}
                    )

            elif status == "failed":
                logger.error(f"Task {task_id} failed on bridge: {data.get('error')}")
                if self.on_task_failed:
                    await self._call_callback(
                        self.on_task_failed,
                        {"task_id": task_id, "error": data.get("error")},
                    )

            elif status == "in_progress":
                logger.debug(
                    f"Task {task_id} in progress: {data.get('progress_percent', 0):.1f}%"
                )

        except Exception as e:
            logger.error(f"Error handling task status: {e}", exc_info=True)

    async def _handle_location_inventory(self, location_updates: list):
        """Update location inventory from bridge."""
        try:
            for loc_update in location_updates:
                location_id = loc_update['location_id']

                # Find matching node in graph
                node = self._find_node_by_id(location_id)
                if not node:
                    logger.warning(f"Location {location_id} not found in graph")
                    continue

                # Update SKU inventory
                sku_inventory = loc_update.get('sku_inventory', {})
                for sku_id, sku_data in sku_inventory.items():
                    if sku_id in node.sku_inventory:
                        node.sku_inventory[sku_id]['stock'] = sku_data.get('stock_level', 0.0)
                        logger.debug(f"Updated {location_id}/{sku_id} stock: {sku_data.get('stock_level', 0.0)}")

                # Call integration callback
                if self.on_inventory_updated:
                    await self._call_callback(
                        self.on_inventory_updated,
                        {'location_id': location_id, 'sku_inventory': sku_inventory}
                    )

            logger.debug(f"Updated inventory for {len(location_updates)} locations")

        except Exception as e:
            logger.error(f"Error handling location inventory: {e}")

    async def _handle_consumption_rates(self, location_consumption: list):
        """Update consumption rates from bridge."""
        try:
            for loc_data in location_consumption:
                location_id = loc_data['location_id']

                # Find matching node in graph
                node = self._find_node_by_id(location_id)
                if not node:
                    logger.warning(f"Location {location_id} not found in graph")
                    continue

                # Update consumption rate
                rate = loc_data.get('consumption_rate', 0.0)
                node.consumption_rate = rate

                # Call integration callback
                if self.on_consumption_updated:
                    await self._call_callback(
                        self.on_consumption_updated,
                        {
                            'location_id': location_id,
                            'consumption_rate': rate,
                            'urgency_level': loc_data.get('urgency_level', 1)
                        }
                    )

                logger.debug(f"Updated {location_id} consumption rate: {rate} items/hour")

        except Exception as e:
            logger.error(f"Error handling consumption rates: {e}")

    async def _handle_system_state(self, system_state):
        """Handle full system state snapshot."""
        try:
            # Process robot telemetry if present
            if system_state.robot_telemetry:
                await self._handle_robot_telemetry(system_state.robot_telemetry)

            # Process inventories if present
            if system_state.location_inventories:
                location_updates = [
                    {
                        'location_id': inv.location_id,
                        'location_name': inv.location_name,
                        'sku_inventory': inv.sku_inventory,
                        'category_inventory': inv.category_inventory
                    }
                    for inv in system_state.location_inventories.values()
                ]
                await self._handle_location_inventory(location_updates)

            # Process consumption if present
            if system_state.consumption_rates:
                location_consumption = [
                    {
                        'location_id': cons.location_id,
                        'location_name': cons.location_name,
                        'consumption_rate': cons.consumption_rate,
                        'time_to_stockout_hours': cons.time_to_stockout_hours,
                        'urgency_level': cons.urgency_level,
                        'category_rates': cons.category_rates
                    }
                    for cons in system_state.consumption_rates.values()
                ]
                await self._handle_consumption_rates(location_consumption)

            logger.debug(f"System state snapshot processed at {system_state.last_update}")

        except Exception as e:
            logger.error(f"Error handling system state: {e}")

    async def _listen_with_reconnect(self):
        """
        Listen for bridge messages with exponential backoff reconnection.

        Runs continuously while _running is True, attempting to reconnect
        if connection is lost, with exponential backoff capped at 30 seconds.
        """
        backoff = 1.0
        while self._running:
            try:
                if not self.client.is_connected:
                    connected = await self.client.connect()
                    if not connected:
                        logger.warning(
                            f"Bridge connection failed, retrying in {backoff:.1f}s..."
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2.0, 30.0)  # Exponential backoff, cap 30s
                        continue

                backoff = 1.0  # Reset on successful connection
                await self.client.listen()

            except asyncio.CancelledError:
                logger.debug("Bridge listener cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Bridge listener error: {e}. Retrying in {backoff:.1f}s...",
                    exc_info=False,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    def _validate_robot_telemetry(self, telemetry: RobotTelemetryData) -> bool:
        """
        Validate that robot telemetry has required fields.

        Args:
            telemetry: RobotTelemetryData to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            if not telemetry.position or len(telemetry.position) < 2:
                logger.warning("robot_telemetry missing position")
                return False
            if telemetry.orientation is None:
                logger.warning("robot_telemetry missing orientation")
                return False
            if telemetry.battery_level is None:
                logger.warning("robot_telemetry missing battery_level")
                return False
            if telemetry.velocity is None:
                logger.warning("robot_telemetry missing velocity")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating telemetry: {e}")
            return False

    def _is_telemetry_stale(
        self, timestamp: float, max_age_s: float = 5.0
    ) -> bool:
        """
        Check if telemetry is too old.

        Args:
            timestamp: Unix timestamp from telemetry
            max_age_s: Maximum age in seconds (default 5.0)

        Returns:
            True if telemetry is older than max_age_s
        """
        age = time.time() - timestamp
        return age > max_age_s

    def _quaternion_to_heading(self, quat: Optional[Dict]) -> float:
        """
        Convert quaternion to heading angle (yaw in radians).

        Handles missing/malformed quaternion gracefully with safe defaults.

        Args:
            quat: Dict with x, y, z, w keys (or None)

        Returns:
            Heading angle in radians (0.0 if quaternion invalid)
        """
        if not quat:
            return 0.0
        try:
            qx = quat.get("x", 0.0)
            qy = quat.get("y", 0.0)
            qz = quat.get("z", 0.0)
            qw = quat.get("w", 1.0)
            # Standard quaternion to yaw formula
            heading = math.atan2(
                2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
            )
            return float(heading)
        except Exception as e:
            logger.warning(f"Error converting quaternion to heading: {e}")
            return 0.0

    def _find_nearest_node_index(self, x: float, y: float) -> Optional[int]:
        """
        Find the index of the node nearest to (x, y) position.

        Uses Euclidean distance to node centers.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Node index if found, None if graph is empty or no nodes
        """
        try:
            if not self.graph_state or not self.graph_state.nodes:
                return None

            best_idx = None
            best_dist = float("inf")

            for i, node in enumerate(self.graph_state.nodes):
                dist = (node.center_x - x) ** 2 + (node.center_y - y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            return best_idx
        except Exception as e:
            logger.error(f"Error finding nearest node: {e}")
            return None

    def _find_node_by_id(self, location_id: str):
        """Find a node in graph by location_id."""
        if not hasattr(self.graph_state, "nodes"):
            return None

        for node in self.graph_state.nodes:
            if node.location_id == location_id or location_id in node.location_ids:
                return node

        return None

    async def _call_callback(self, callback: Callable, data):
        """Safely call a callback (sync or async)."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"Error in callback: {e}")


class ROSBridgeTaskSubmitter:
    """
    Submits tasks to robot via ROS bridge.

    Translates CareRobotics tasks to bridge task format and sends via WebSocket.
    """

    def __init__(self, client: ROSBridgeClient, graph_state: GraphState):
        """
        Initialize submitter.

        Args:
            client: ROSBridgeClient instance
            graph_state: GraphState instance (for node coordinates)
        """
        self.client = client
        self.graph_state = graph_state

    async def submit_task(self, task, robot_id: int = 0) -> bool:
        """
        Submit a task to the robot via bridge.

        Args:
            task: Task object from CareRobotics
            robot_id: Which robot to assign to (default 0)

        Returns:
            True if submitted successfully
        """
        try:
            # Convert task's node path to coordinates
            waypoints = []
            if hasattr(task, 'planned_path') and task.planned_path:
                for node_idx in task.planned_path:
                    node = self.graph_state.nodes[node_idx]
                    waypoints.append({
                        'x': node.center_x,
                        'y': node.center_y,
                        'z': 0.0
                    })

            # Or if task has from/to locations
            elif hasattr(task, 'from_location_index') and hasattr(task, 'to_location_index'):
                from_node = self.graph_state.nodes[task.from_location_index]
                to_node = self.graph_state.nodes[task.to_location_index]
                waypoints = [
                    {'x': from_node.center_x, 'y': from_node.center_y, 'z': 0.0},
                    {'x': to_node.center_x, 'y': to_node.center_y, 'z': 0.0}
                ]

            if not waypoints:
                logger.error(f"No waypoints for task {task.task_id}")
                return False

            # Create task command for bridge
            task_command = {
                'task_id': task.task_id,
                'robot_id': robot_id,
                'waypoints': waypoints,
                'task_metadata': {
                    'task_type': getattr(task, 'task_type', 'unknown'),
                    'num_items': getattr(task, 'num_items', 1),
                    'sku_id': getattr(task, 'sku_id', None),
                    'description': f"Task {task.task_id}"
                }
            }

            # Send to bridge
            success = await self.client.send_task_command(task_command)
            if success:
                logger.info(f"Submitted task {task.task_id} to robot {robot_id}")
            return success

        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return False

    async def cancel_task(self, task_id: int) -> bool:
        """Cancel a task on the robot."""
        return await self.client.send_task_cancel(task_id)
