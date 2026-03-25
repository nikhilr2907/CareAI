"""
ROS Bridge Client - Receives data from ROS2 Bridge Node.

This module connects to the ROS2 bridge via WebSocket and receives:
- Robot telemetry (position, velocity, battery)
- Task status updates
- Location inventory (SKU levels)
- Consumption rates
- System state aggregations

Status: READY TO USE (not active until Nav2 is deployed)
"""

import asyncio
import json
import logging
from typing import Callable, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class RobotTelemetryData:
    """Current robot state from bridge."""
    robot_id: int
    timestamp: float
    position: tuple  # (x, y, z)
    orientation: Dict[str, float]  # quaternion {x, y, z, w}
    velocity: Dict[str, float]  # {linear_x, linear_y, angular_z}
    battery_level: float  # 0.0-1.0
    current_capacity: int
    max_capacity: int
    active_task_id: Optional[int]


@dataclass
class LocationInventoryData:
    """Inventory at a specific location."""
    location_id: str
    location_name: str
    sku_inventory: Dict[str, Dict] = field(default_factory=dict)  # sku_id -> {stock_level, max_level, par, reorder, category}
    category_inventory: Dict[str, Dict] = field(default_factory=dict)  # category -> {total_stock, max_stock, num_skus}


@dataclass
class LocationConsumptionData:
    """Consumption rates at a specific location."""
    location_id: str
    location_name: str
    consumption_rate: float  # items/hour
    time_to_stockout_hours: float
    urgency_level: int  # 1-5
    category_rates: Dict[str, float] = field(default_factory=dict)  # category -> rate


@dataclass
class SystemState:
    """Full system state snapshot."""
    timestamp: float
    robot_telemetry: Optional[RobotTelemetryData] = None
    location_inventories: Dict[str, LocationInventoryData] = field(default_factory=dict)
    consumption_rates: Dict[str, LocationConsumptionData] = field(default_factory=dict)
    last_update: str = ""


class ROSBridgeClient:
    """
    Receives data from ROS2 Bridge via WebSocket.

    Does NOT connect until explicitly started - safe to instantiate without Nav2.
    """

    def __init__(self, bridge_url: str = "ws://localhost:8765"):
        """
        Initialize client (but don't connect yet).

        Args:
            bridge_url: WebSocket URL of ROS bridge (default: localhost:8765)
        """
        self.bridge_url = bridge_url
        self.websocket = None
        self.is_connected = False
        self.is_running = False

        # Local state (always available, even if bridge is down)
        self.robot_telemetry: Optional[RobotTelemetryData] = None
        self.location_inventories: Dict[str, LocationInventoryData] = {}
        self.consumption_rates: Dict[str, LocationConsumptionData] = {}
        self.system_state = SystemState(timestamp=0.0)

        # Message callbacks (user can register handlers)
        self.on_robot_telemetry: Optional[Callable] = None
        self.on_task_status: Optional[Callable] = None
        self.on_location_inventory: Optional[Callable] = None
        self.on_consumption_rates: Optional[Callable] = None
        self.on_system_state: Optional[Callable] = None
        self.on_connection_lost: Optional[Callable] = None

        logger.info(f"ROS Bridge Client initialized (not connected) - bridge URL: {bridge_url}")

    async def connect(self):
        """
        Connect to ROS bridge WebSocket.

        Returns:
            True if connected, False otherwise
        """
        try:
            import websockets
            self.websocket = await websockets.connect(self.bridge_url)
            self.is_connected = True
            self.is_running = True
            logger.info(f"Connected to ROS bridge at {self.bridge_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ROS bridge: {e}")
            self.is_connected = False
            return False

    async def listen(self):
        """
        Listen for messages from bridge (blocking loop).

        Run this in an asyncio task:
            asyncio.create_task(client.listen())
        """
        if not self.is_connected:
            logger.error("Not connected to bridge. Call connect() first.")
            return

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from bridge: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        except Exception as e:
            logger.error(f"Connection lost: {e}")
            self.is_connected = False
            self.is_running = False
            if self.on_connection_lost:
                await self.on_connection_lost()

    async def handle_message(self, data: Dict[str, Any]):
        """Route incoming message to appropriate handler."""
        msg_type = data.get('message_type')

        if msg_type == 'robot_state':
            await self._handle_robot_state(data)
        elif msg_type == 'task_status':
            await self._handle_task_status(data)
        elif msg_type == 'location_inventory':
            await self._handle_location_inventory(data)
        elif msg_type == 'consumption_rates':
            await self._handle_consumption_rates(data)
        elif msg_type == 'system_state':
            await self._handle_system_state(data)
        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_robot_state(self, data: Dict):
        """Handle robot telemetry update."""
        try:
            telemetry = RobotTelemetryData(
                robot_id=data.get('robot_id', 0),
                timestamp=data.get('timestamp', 0.0),
                position=(
                    data['position']['x'],
                    data['position']['y'],
                    data['position'].get('z', 0.0)
                ),
                orientation=data.get('orientation_quat', {'x': 0, 'y': 0, 'z': 0, 'w': 1}),
                velocity=data.get('velocity', {'linear_x': 0, 'linear_y': 0, 'angular_z': 0}),
                battery_level=data.get('battery_level', 0.0),
                current_capacity=data.get('current_capacity', 0),
                max_capacity=data.get('max_capacity', 12),
                active_task_id=data.get('active_task_id')
            )
            self.robot_telemetry = telemetry

            if self.on_robot_telemetry:
                await self._call_callback(self.on_robot_telemetry, telemetry)

            logger.debug(f"Robot state: pos=({telemetry.position[0]:.2f}, {telemetry.position[1]:.2f}), battery={telemetry.battery_level:.1%}")
        except Exception as e:
            logger.error(f"Error parsing robot state: {e}")

    async def _handle_task_status(self, data: Dict):
        """Handle task status update."""
        try:
            if self.on_task_status:
                await self._call_callback(self.on_task_status, data)

            logger.info(f"Task {data.get('task_id')} status: {data.get('status')}")
        except Exception as e:
            logger.error(f"Error handling task status: {e}")

    async def _handle_location_inventory(self, data: Dict):
        """Handle location inventory update."""
        try:
            location_updates = data.get('location_updates', [])

            for loc in location_updates:
                location_id = loc['location_id']
                inventory = LocationInventoryData(
                    location_id=location_id,
                    location_name=loc.get('location_name', location_id),
                    sku_inventory=loc.get('sku_inventory', {}),
                    category_inventory=loc.get('category_inventory', {})
                )
                self.location_inventories[location_id] = inventory

            if self.on_location_inventory:
                await self._call_callback(self.on_location_inventory, location_updates)

            logger.debug(f"Updated inventory for {len(location_updates)} locations")
        except Exception as e:
            logger.error(f"Error parsing location inventory: {e}")

    async def _handle_consumption_rates(self, data: Dict):
        """Handle consumption rates update."""
        try:
            location_consumption = data.get('location_consumption', [])

            for loc in location_consumption:
                location_id = loc['location_id']
                consumption = LocationConsumptionData(
                    location_id=location_id,
                    location_name=loc.get('location_name', location_id),
                    consumption_rate=loc.get('consumption_rate', 0.0),
                    time_to_stockout_hours=loc.get('time_to_stockout_hours', float('inf')),
                    urgency_level=loc.get('urgency_level', 1),
                    category_rates=loc.get('category_rates', {})
                )
                self.consumption_rates[location_id] = consumption

            if self.on_consumption_rates:
                await self._call_callback(self.on_consumption_rates, location_consumption)

            logger.debug(f"Updated consumption rates for {len(location_consumption)} locations")
        except Exception as e:
            logger.error(f"Error parsing consumption rates: {e}")

    async def _handle_system_state(self, data: Dict):
        """Handle full system state snapshot."""
        try:
            # Parse robot telemetry from snapshot
            robot_data = data.get('robot_state')
            robot_telemetry = None
            if robot_data:
                robot_telemetry = RobotTelemetryData(
                    robot_id=robot_data.get('robot_id', 0),
                    timestamp=data.get('timestamp', 0.0),
                    position=(
                        robot_data['position']['x'],
                        robot_data['position']['y'],
                        robot_data['position'].get('z', 0.0)
                    ),
                    orientation=robot_data.get('orientation_quat', {'x': 0, 'y': 0, 'z': 0, 'w': 1}),
                    velocity=robot_data.get('velocity', {'linear_x': 0, 'linear_y': 0, 'angular_z': 0}),
                    battery_level=robot_data.get('battery_level', 0.0),
                    current_capacity=robot_data.get('current_capacity', 0),
                    max_capacity=robot_data.get('max_capacity', 12),
                    active_task_id=robot_data.get('active_task_id')
                )

            # Parse inventory from snapshot
            inventories = {}
            for loc in data.get('location_inventories', []):
                location_id = loc['location_id']
                inventories[location_id] = LocationInventoryData(
                    location_id=location_id,
                    location_name=loc.get('location_name', location_id),
                    sku_inventory=loc.get('sku_inventory', {}),
                    category_inventory=loc.get('category_inventory', {})
                )

            # Parse consumption from snapshot
            consumptions = {}
            for loc in data.get('consumption_rates', []):
                location_id = loc['location_id']
                consumptions[location_id] = LocationConsumptionData(
                    location_id=location_id,
                    location_name=loc.get('location_name', location_id),
                    consumption_rate=loc.get('consumption_rate', 0.0),
                    time_to_stockout_hours=loc.get('time_to_stockout_hours', float('inf')),
                    urgency_level=loc.get('urgency_level', 1),
                    category_rates=loc.get('category_rates', {})
                )

            # Update system state
            self.system_state = SystemState(
                timestamp=data.get('timestamp', 0.0),
                robot_telemetry=robot_telemetry,
                location_inventories=inventories,
                consumption_rates=consumptions,
                last_update=datetime.now().isoformat()
            )

            if self.on_system_state:
                await self._call_callback(self.on_system_state, self.system_state)

            logger.debug(f"System state updated: {len(inventories)} locations, {len(consumptions)} consumption rates")
        except Exception as e:
            logger.error(f"Error parsing system state: {e}")

    async def _call_callback(self, callback: Callable, data: Any):
        """Safely call a callback (sync or async)."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"Error in callback: {e}")

    async def send_task_command(self, task_command: Dict) -> bool:
        """
        Send a task command to the bridge.

        Args:
            task_command: Task dict with {task_id, waypoints, ...}

        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            logger.error("Not connected to bridge")
            return False

        try:
            msg = {
                'message_type': 'task_command',
                **task_command
            }
            await self.websocket.send(json.dumps(msg))
            logger.info(f"Sent task command: {task_command.get('task_id')}")
            return True
        except Exception as e:
            logger.error(f"Failed to send task command: {e}")
            return False

    async def send_task_cancel(self, task_id: int) -> bool:
        """Cancel a task."""
        if not self.is_connected:
            logger.error("Not connected to bridge")
            return False

        try:
            msg = {
                'message_type': 'task_cancel',
                'task_id': task_id
            }
            await self.websocket.send(json.dumps(msg))
            logger.info(f"Sent task cancel for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send task cancel: {e}")
            return False

    async def disconnect(self):
        """Disconnect from bridge."""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        self.is_running = False
        logger.info("Disconnected from ROS bridge")

    def get_robot_state(self) -> Optional[RobotTelemetryData]:
        """Get last received robot telemetry."""
        return self.robot_telemetry

    def get_location_inventory(self, location_id: str) -> Optional[LocationInventoryData]:
        """Get inventory for a specific location."""
        return self.location_inventories.get(location_id)

    def get_all_inventories(self) -> Dict[str, LocationInventoryData]:
        """Get all location inventories."""
        return self.location_inventories.copy()

    def get_consumption_rate(self, location_id: str) -> Optional[LocationConsumptionData]:
        """Get consumption rate for a specific location."""
        return self.consumption_rates.get(location_id)

    def get_all_consumption_rates(self) -> Dict[str, LocationConsumptionData]:
        """Get all consumption rates."""
        return self.consumption_rates.copy()

    def get_system_state(self) -> SystemState:
        """Get last received system state snapshot."""
        return self.system_state
