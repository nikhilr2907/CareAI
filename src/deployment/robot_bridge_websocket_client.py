"""
ROS Bridge Client - Receives data from ROS2 Bridge Node.

This module connects to the ROS2 bridge via WebSocket and receives:
- Robot telemetry (position, velocity, battery)
- Task status updates
- Location inventory (SKU levels)
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
    velocity: Dict[str, float]  # {linear_x, angular_z}
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
class SystemState:
    """Full system state snapshot."""
    timestamp: float
    robot_telemetry: Optional[RobotTelemetryData] = None
    location_inventories: Dict[str, LocationInventoryData] = field(default_factory=dict)
    last_update: str = ""


class RobotBridgeWebSocketClient:
    """
    Receives data from ROS2 Bridge via WebSocket.

    Does NOT connect until explicitly started - safe to instantiate without Nav2.
    """

    def __init__(self, bridge_url: str = "ws://localhost:8765"):
        """
        Initialize client (but don't connect yet).

        Args:
            bridge_url: Base WebSocket URL of ROS bridge (default: localhost:8765).
                        The /ws path is appended automatically on connect.
        """
        self.bridge_url = bridge_url
        self.websocket = None
        self.is_connected = False
        self.is_running = False

        # Local state (always available, even if bridge is down)
        self.robot_telemetry: Optional[RobotTelemetryData] = None
        self.location_inventories: Dict[str, LocationInventoryData] = {}
        self.system_state = SystemState(timestamp=0.0)

        # Message callbacks (user can register handlers)
        self.on_robot_telemetry: Optional[Callable] = None
        self.on_task_status: Optional[Callable] = None
        self.on_location_inventory: Optional[Callable] = None
        self.on_system_state: Optional[Callable] = None
        self.on_connection_lost: Optional[Callable] = None
        self.on_emergency_stop: Optional[Callable] = None
        logger.info(f"ROS Bridge Client initialized (not connected) - bridge URL: {bridge_url}")

    async def connect(self):
        """
        Connect to ROS bridge WebSocket.

        Returns:
            True if connected, False otherwise
        """
        try:
            import websockets
            # Bridge serves the WebSocket endpoint at /ws (FastAPI route)
            url = self.bridge_url.rstrip("/") + "/ws"
            self.websocket = await websockets.connect(url)
            self.is_connected = True
            self.is_running = True
            logger.info(f"Connected to ROS bridge at {url}")
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
        elif msg_type == 'system_state':
            await self._handle_system_state(data)
        elif msg_type == 'emergency_stop':
            await self._handle_emergency_stop(data)
        elif msg_type in ('task_response', 'connected', 'heartbeat'):
            logger.debug(f"Bridge message: {msg_type} status={data.get('status')}")
        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_robot_state(self, data: Dict):
        """Handle robot telemetry update."""
        try:
            pos = data.get('position') or {}
            if 'x' not in pos or 'y' not in pos:
                logger.warning(f"robot_state missing position fields: {list(pos.keys())} — skipping update")
                return
            telemetry = RobotTelemetryData(
                robot_id=data.get('robot_id', 0),
                timestamp=data.get('timestamp', 0.0),
                position=(float(pos['x']), float(pos['y']), float(pos.get('z', 0.0))),
                orientation=data.get('orientation_quat', {'x': 0, 'y': 0, 'z': 0, 'w': 1}),
                velocity=data.get('velocity', {'linear_x': 0, 'angular_z': 0}),
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
            logger.error(f"Error parsing robot state: {e}", exc_info=True)

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

    async def _handle_system_state(self, data: Dict):
        """Handle full system state snapshot."""
        try:
            # Parse robot telemetry from snapshot
            robot_data = data.get('robot_state')
            robot_telemetry = None
            if robot_data:
                rpos = robot_data.get('position') or {}
                if 'x' in rpos and 'y' in rpos:
                    robot_telemetry = RobotTelemetryData(
                        robot_id=robot_data.get('robot_id', 0),
                        timestamp=data.get('timestamp', 0.0),
                        position=(float(rpos['x']), float(rpos['y']), float(rpos.get('z', 0.0))),
                        orientation=robot_data.get('orientation_quat', {'x': 0, 'y': 0, 'z': 0, 'w': 1}),
                        velocity=robot_data.get('velocity', {'linear_x': 0, 'angular_z': 0}),
                        battery_level=robot_data.get('battery_level', 0.0),
                        current_capacity=robot_data.get('current_capacity', 0),
                        max_capacity=robot_data.get('max_capacity', 12),
                        active_task_id=robot_data.get('active_task_id')
                    )
                else:
                    logger.warning(f"system_state robot_state missing position fields: {list(rpos.keys())}")

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

            # Mirror to the standalone-message cache so get_all_inventories()
            # sees these even when the bridge buffers inventory into system_state
            # (i.e. buffer_inventory_updates=true; in that mode no standalone
            # location_inventory messages ever arrive).
            self.location_inventories = inventories.copy()

            # Update system state
            self.system_state = SystemState(
                timestamp=data.get('timestamp', 0.0),
                robot_telemetry=robot_telemetry,
                location_inventories=inventories,
                last_update=datetime.now().isoformat()
            )

            if self.on_system_state:
                await self._call_callback(self.on_system_state, self.system_state)

            logger.debug(f"System state updated: {len(inventories)} locations")
        except Exception as e:
            logger.error(f"Error parsing system state: {e}")

    async def _handle_emergency_stop(self, data: Dict):
        """Handle emergency stop signal. Fires on_emergency_stop callback when active=True."""
        active = data.get('active', True)
        logger.critical(f"Emergency stop received: active={active}")
        if active and self.on_emergency_stop:
            await self._call_callback(self.on_emergency_stop, data)

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

    def get_system_state(self) -> SystemState:
        """Get last received system state snapshot."""
        return self.system_state

