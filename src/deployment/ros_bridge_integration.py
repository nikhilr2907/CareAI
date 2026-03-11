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
from typing import Optional, Dict, Callable
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

        # Register handlers with client
        self.client.on_robot_telemetry = self._handle_robot_telemetry
        self.client.on_task_status = self._handle_task_status
        self.client.on_location_inventory = self._handle_location_inventory
        self.client.on_consumption_rates = self._handle_consumption_rates
        self.client.on_system_state = self._handle_system_state

        logger.info("ROS Bridge Integration initialized")

    async def start(self, robot_states: Dict[int, RobotState]):
        """
        Start the ROS bridge client and listen for updates.

        Args:
            robot_states: Dict of {robot_id: RobotState}
        """
        self.robot_states = robot_states

        if not await self.client.connect():
            logger.error("Failed to connect to ROS bridge")
            return

        # Start listening in background
        asyncio.create_task(self.client.listen())
        logger.info("ROS Bridge client listening...")

    async def stop(self):
        """Stop listening and disconnect."""
        await self.client.disconnect()

    async def _handle_robot_telemetry(self, telemetry: RobotTelemetryData):
        """Update robot state from bridge telemetry."""
        try:
            # For now, we assume single robot (robot_id=0)
            # In multi-robot, need to identify which robot from telemetry
            robot_id = 0

            if robot_id not in self.robot_states:
                logger.warning(f"Robot {robot_id} not in robot_states")
                return

            robot_state = self.robot_states[robot_id]

            # Create RobotTelemetry object (matches your existing format)
            ros_telemetry = RobotTelemetry(
                timestamp=telemetry.timestamp,
                robot_id=robot_id,
                x=telemetry.position[0],
                y=telemetry.position[1],
                heading=0.0,  # TODO: convert quaternion to heading
                current_node_index=None,  # TODO: find nearest node
                current_edge_index=None,
                edge_progress=0.0,
                velocity_ms=telemetry.velocity.get('linear_x', 0.0),
                is_moving=telemetry.velocity.get('linear_x', 0.0) > 0.01,
                battery_level=telemetry.battery_level,
                current_capacity=telemetry.current_capacity,
                is_available=False,  # TODO: determine from bridge
                active_task_id=telemetry.active_task_id,
                remaining_path=[],
                eta_to_next_node=0.0
            )

            # Update robot state
            robot_state.update_telemetry(ros_telemetry)

            # Call integration callback
            if self.on_robot_position_updated:
                await self._call_callback(
                    self.on_robot_position_updated,
                    {'robot_id': robot_id, 'position': telemetry.position, 'battery': telemetry.battery_level}
                )

            logger.debug(f"Updated robot {robot_id} telemetry: pos=({telemetry.position[0]:.2f}, {telemetry.position[1]:.2f})")

        except Exception as e:
            logger.error(f"Error handling robot telemetry: {e}")

    async def _handle_task_status(self, data: Dict):
        """Handle task status updates from bridge."""
        try:
            task_id = data.get('task_id')
            status = data.get('status')

            if status == 'completed':
                logger.info(f"Task {task_id} completed on bridge")
                if self.on_task_completed:
                    await self._call_callback(self.on_task_completed, {'task_id': task_id})

            elif status == 'failed':
                logger.error(f"Task {task_id} failed on bridge: {data.get('error')}")
                if self.on_task_failed:
                    await self._call_callback(self.on_task_failed, {'task_id': task_id, 'error': data.get('error')})

            elif status == 'in_progress':
                logger.debug(f"Task {task_id} in progress: {data.get('progress_percent', 0):.1f}%")

        except Exception as e:
            logger.error(f"Error handling task status: {e}")

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

    def _find_node_by_id(self, location_id: str):
        """Find a node in graph by location_id."""
        if not hasattr(self.graph_state, 'nodes'):
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
