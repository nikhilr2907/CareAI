"""
Robot Bridge Layer - Interface between environment and real/mock robots.

This layer abstracts robot communication, making it easy to swap between:
- Mock robots (simulator) for testing
- Real robots (ROS/MQTT) for deployment
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging
import asyncio
import math


@dataclass
class RobotTelemetryData:
    """Standardized telemetry format from robots."""
    robot_id: int
    timestamp: float

    # Position
    x: float
    y: float
    heading: float

    # Motion
    velocity_ms: float
    current_node_index: Optional[int]

    # Resources
    battery_level: float
    current_capacity: int
    max_capacity: int

    # Status
    is_available: bool
    active_task_id: Optional[int]
    eta_to_next_node: float


class RobotBridge:
    """
    Abstract interface for robot communication.
    Subclasses implement specific backends (mock, ROS, MQTT).
    """

    def get_telemetry(self, robot_id: int) -> RobotTelemetryData:
        """Get current telemetry from a robot."""
        raise NotImplementedError

    def get_all_telemetry(self) -> List[RobotTelemetryData]:
        """Get telemetry from all robots."""
        raise NotImplementedError

    def send_path_command(self, robot_id: int, path: List[int], task_id: int, num_items: int):
        """Command robot to follow a path for a task."""
        raise NotImplementedError

    def get_num_robots(self) -> int:
        """Get number of robots in fleet."""
        raise NotImplementedError


class MockRobotBridge(RobotBridge):
    """
    Mock bridge using simulator for testing.

    In deployment, replace this with ROSRobotBridge or MQTTRobotBridge.
    """

    def __init__(self, robot_simulators: List):
        """
        Initialize with robot simulators.

        Args:
            robot_simulators: List of RobotSimulator objects
        """
        self.robot_simulators = robot_simulators

    def get_telemetry(self, robot_id: int) -> RobotTelemetryData:
        """Get telemetry from simulator."""
        if robot_id >= len(self.robot_simulators):
            raise ValueError(f"Robot {robot_id} not found")

        sim = self.robot_simulators[robot_id]
        telemetry = sim.get_telemetry()

        # Convert simulator telemetry to standardized format
        return RobotTelemetryData(
            robot_id=robot_id,
            timestamp=telemetry.timestamp,
            x=telemetry.x,
            y=telemetry.y,
            heading=telemetry.heading,
            velocity_ms=telemetry.velocity_ms,
            current_node_index=telemetry.current_node_index,
            battery_level=telemetry.battery_level,
            current_capacity=telemetry.current_capacity,
            max_capacity=sim.max_capacity,
            is_available=telemetry.is_available,
            active_task_id=sim.active_task_id,
            eta_to_next_node=telemetry.eta_to_next_node
        )

    def get_all_telemetry(self) -> List[RobotTelemetryData]:
        """Get telemetry from all robots."""
        return [self.get_telemetry(i) for i in range(len(self.robot_simulators))]

    def send_path_command(self, robot_id: int, path: List[int], task_id: int, num_items: int):
        """Send path command to simulator."""
        if robot_id >= len(self.robot_simulators):
            raise ValueError(f"Robot {robot_id} not found")

        sim = self.robot_simulators[robot_id]
        sim.set_path(path, task_id, num_items)

    def update_simulators(self, time_delta: float):
        """
        Update all simulators (mock robots move).

        In real deployment, this wouldn't exist - real robots move on their own.
        """
        for sim in self.robot_simulators:
            sim.update(time_delta)

    def get_num_robots(self) -> int:
        """Get number of robots."""
        return len(self.robot_simulators)


class _BridgeTaskProxy:
    """
    Duck-type Task proxy for ROSBridgeTaskSubmitter.

    Minimal adapter so the task submitter can work with path data from send_path_command.
    Private to robot_bridge.py.
    """

    def __init__(self, task_id: int, planned_path: List[int], num_items: int, graph_state: Any = None):
        self.task_id = task_id
        self.planned_path = planned_path  # List[int] — node indices
        self.num_items = num_items
        self.task_type = "replenishment"
        self.sku_id = None
        self.from_location_index = planned_path[0] if planned_path else None
        self.to_location_index = planned_path[-1] if planned_path else None


class ROSRobotBridge(RobotBridge):
    """
    Bridge to real robots via ROS 2 WebSocket bridge.

    Connects to a ROS 2 bridge node running on a configurable URL (default: ws://localhost:8765).
    The bridge node handles Nav2 integration and publishes telemetry/status updates.

    One client per robot, with automatic reconnection and comprehensive error handling.
    """

    def __init__(self, bridge_url: str = "ws://localhost:8765", num_robots: int = 1):
        """
        Initialize ROS bridge.

        Args:
            bridge_url: WebSocket URL to the ROS bridge node (default: ws://localhost:8765)
            num_robots: Number of robots in the fleet (default: 1)

        Note:
            ROSBridgeClient is imported lazily to avoid ImportError during training
            if websockets is not installed.
        """
        # Deferred import — avoids ImportError during training if websockets not installed
        from .ros_bridge_client import ROSBridgeClient as _ROSBridgeClient
        from .ros_bridge_integration import ROSBridgeTaskSubmitter as _Submitter

        self._num_robots = num_robots
        self._bridge_url = bridge_url
        self._clients: Dict[int, Any] = {}       # robot_id -> ROSBridgeClient
        self._submitters: Dict[int, Any] = {}    # robot_id -> ROSBridgeTaskSubmitter
        self._graph_state = None
        self._logger = logging.getLogger(__name__)

        # Create one client and submitter per robot
        for robot_id in range(num_robots):
            client = _ROSBridgeClient(bridge_url=bridge_url)
            self._clients[robot_id] = client
            self._submitters[robot_id] = _Submitter(client, graph_state=None)

        self._logger.info(
            f"ROSRobotBridge initialized: {num_robots} robot(s), "
            f"bridge_url={bridge_url}"
        )

    def set_graph_state(self, graph_state: Any) -> None:
        """
        Provide graph context so task submitters can resolve node indices to (x,y).

        Call this before using send_path_command if the path contains node indices.

        Args:
            graph_state: GraphState object with nodes list
        """
        self._graph_state = graph_state
        for submitter in self._submitters.values():
            submitter.graph_state = graph_state

    def get_telemetry(self, robot_id: int) -> Optional[RobotTelemetryData]:
        """
        Get current telemetry snapshot from a robot.

        Args:
            robot_id: Robot identifier (0-indexed)

        Returns:
            RobotTelemetryData or None if no data has been received yet
        """
        client = self._clients.get(robot_id)
        if client is None:
            self._logger.warning(f"No client registered for robot_id={robot_id}")
            return None

        raw = client.get_robot_state()
        if raw is None:
            return None  # No data received yet from bridge

        return self._convert_telemetry(robot_id, raw)

    def get_all_telemetry(self) -> List[RobotTelemetryData]:
        """
        Get telemetry from all robots in the fleet.

        Returns:
            List of RobotTelemetryData (empty list if no data available)
        """
        result = []
        for robot_id in range(self._num_robots):
            tel = self.get_telemetry(robot_id)
            if tel is not None:
                result.append(tel)
        return result

    def send_path_command(
        self, robot_id: int, path: List[int], task_id: int, num_items: int
    ) -> None:
        """
        Command a robot to follow a path (list of node indices).

        Schedules the WebSocket task send asynchronously. This method is synchronous
        to match the RobotBridge interface, but internally uses asyncio.ensure_future()
        if an event loop is running, or asyncio.run_until_complete() otherwise.

        Args:
            robot_id: Robot identifier
            path: List of node indices to visit in order
            task_id: Unique task identifier
            num_items: Number of items to deliver
        """
        coro = self._async_send_path_command(robot_id, path, task_id, num_items)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Deployment context: event loop already running (async task)
                asyncio.ensure_future(coro)
            else:
                # Fallback: blocking send for sync callers (e.g. test scripts)
                loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop in current thread — create one
            asyncio.run(coro)
        except Exception as e:
            self._logger.error(
                f"send_path_command failed for robot {robot_id}: {e}", exc_info=True
            )

    async def _async_send_path_command(
        self, robot_id: int, path: List[int], task_id: int, num_items: int
    ) -> bool:
        """
        Async implementation of send_path_command.

        Converts the path + task info into a bridge task and submits via WebSocket.

        Args:
            robot_id: Robot identifier
            path: List of node indices
            task_id: Task identifier
            num_items: Number of items

        Returns:
            True if submitted successfully, False otherwise
        """
        submitter = self._submitters.get(robot_id)
        if submitter is None:
            self._logger.error(f"No submitter registered for robot_id={robot_id}")
            return False

        try:
            proxy = _BridgeTaskProxy(
                task_id=task_id,
                planned_path=path,
                num_items=num_items,
                graph_state=self._graph_state,
            )
            result = await submitter.submit_task(proxy, robot_id=robot_id)
            return result
        except Exception as e:
            self._logger.error(
                f"Error submitting task {task_id} to robot {robot_id}: {e}",
                exc_info=True,
            )
            return False

    def get_num_robots(self) -> int:
        """Get the number of robots in this bridge."""
        return self._num_robots

    def _convert_telemetry(
        self, robot_id: int, raw: Any
    ) -> RobotTelemetryData:
        """
        Convert ROS bridge telemetry format to RobotTelemetryData.

        Handles:
        - Quaternion orientation → heading (yaw angle in radians)
        - Safe defaults for missing fields
        - Type conversions

        Args:
            robot_id: Robot identifier
            raw: Raw telemetry from ROSBridgeClient.get_robot_state()

        Returns:
            RobotTelemetryData in environment format
        """
        # Position tuple or default
        pos = raw.position or (0.0, 0.0, 0.0)

        # Quaternion to yaw conversion
        quat = raw.orientation or {}
        qx = quat.get("x", 0.0)
        qy = quat.get("y", 0.0)
        qz = quat.get("z", 0.0)
        qw = quat.get("w", 1.0)
        # Standard quaternion to yaw formula
        heading = math.atan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
        )

        # Velocity
        vel = raw.velocity or {}
        speed = abs(vel.get("linear_x", 0.0))

        return RobotTelemetryData(
            robot_id=robot_id,
            timestamp=float(raw.timestamp or 0.0),
            x=float(pos[0]),
            y=float(pos[1]),
            heading=heading,
            velocity_ms=speed,
            current_node_index=None,  # Nav2 does not expose graph node index
            battery_level=float(raw.battery_level or 1.0),
            current_capacity=int(raw.current_capacity or 0),
            max_capacity=int(raw.max_capacity or 12),
            is_available=(raw.active_task_id is None),
            active_task_id=raw.active_task_id,
            eta_to_next_node=0.0,  # Not available from Nav2 bridge
        )
