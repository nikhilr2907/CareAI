import asyncio
import logging
import math
import os
import signal
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


def _to_epoch_seconds(ts) -> float:
    """Coerce a timestamp (epoch number or ISO-8601 string) into Unix-epoch seconds, falling back to 0.0 on parse failure."""
    if ts is None or ts == "":
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.rstrip("Z")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


@dataclass
class RobotTelemetryData:
    robot_id: int
    timestamp: float

    x: float
    y: float
    heading: float

    velocity_ms: float
    current_node_index: Optional[int]

    battery_level: float
    current_capacity: int
    max_capacity: int

    is_available: bool
    active_task_id: Optional[int]
    eta_to_next_node: float


class RobotBackend(ABC):
    """Abstract adapter contract for robot communication backends (mock, WebSocket bridge, MQTT, ...), not the ROS bridge node itself."""

    @abstractmethod
    def get_telemetry(self, robot_id: int) -> RobotTelemetryData:
        pass

    @abstractmethod
    def get_all_telemetry(self) -> List[RobotTelemetryData]:
        pass

    @abstractmethod
    def send_path_command(self, robot_id: int, path: List[int], task_id: int, num_items: int):
        pass

    @abstractmethod
    def get_num_robots(self) -> int:
        pass


class MockRobotBackend(RobotBackend):
    """Simulator-backed robot backend for testing; replace with ROSBridgeRobotBackend in deployment."""

    def __init__(self, robot_simulators: List):
        self.robot_simulators = robot_simulators

    def get_telemetry(self, robot_id: int) -> RobotTelemetryData:
        if robot_id >= len(self.robot_simulators):
            raise ValueError(f"Robot {robot_id} not found")

        sim = self.robot_simulators[robot_id]
        telemetry = sim.get_telemetry()

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
            eta_to_next_node=telemetry.eta_to_next_node,
        )

    def get_all_telemetry(self) -> List[RobotTelemetryData]:
        return [self.get_telemetry(i) for i in range(len(self.robot_simulators))]

    def send_path_command(self, robot_id: int, path: List[int], task_id: int, num_items: int):
        if robot_id >= len(self.robot_simulators):
            raise ValueError(f"Robot {robot_id} not found")

        sim = self.robot_simulators[robot_id]
        sim.set_path(path, task_id, num_items)

    def update_simulators(self, time_delta: float):
        """Update all simulators; this has no real-deployment equivalent since real robots move on their own."""
        for sim in self.robot_simulators:
            sim.update(time_delta)

    def get_num_robots(self) -> int:
        return len(self.robot_simulators)


class _BridgeTaskProxy:
    """Duck-type Task proxy letting RobotBridgeTaskDispatcher work with path data from send_path_command."""

    def __init__(self, task_id: int, planned_path: List[int], num_items: int):
        self.task_id = task_id
        self.planned_path = planned_path
        self.num_items = num_items
        self.task_type = "replenishment"
        self.sku_id = None
        self.from_location_index = planned_path[0] if planned_path else None
        self.to_location_index = planned_path[-1] if planned_path else None


class ROSBridgeRobotBackend(RobotBackend):
    """Robot backend that talks to the external ROS bridge over WebSocket, with one client per robot and automatic reconnection."""

    def __init__(self, bridge_url: Optional[str] = None, num_robots: int = 1):
        """Initialize the ROS bridge backend, resolving the bridge URL from argument, .env, or a hardcoded fallback."""
        # Deferred import — avoids ImportError during training if websockets not installed
        from .robot_bridge_state_sync import RobotBridgeTaskDispatcher as _Dispatcher
        from .robot_bridge_websocket_client import RobotBridgeWebSocketClient as _WebSocketClient

        if bridge_url is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            bridge_url = os.getenv("ROS_BRIDGE_URL", "ws://localhost:8765")

        self._num_robots = num_robots
        self._bridge_url = bridge_url
        self._clients: Dict[int, Any] = {}  # robot_id -> RobotBridgeWebSocketClient
        self._submitters: Dict[int, Any] = {}  # robot_id -> RobotBridgeTaskDispatcher
        self._listen_tasks: Dict[int, Any] = {}  # robot_id -> asyncio.Task
        self._running = False
        self._graph_state = None
        self._logger = logging.getLogger(__name__)

        # Task completion callbacks — set by the training/deployment loop
        self.on_task_completed: Optional[Any] = None  # Callable(robot_id, task_id)
        self.on_task_failed: Optional[Any] = None  # Callable(robot_id, task_id, error)

        self._completed_task_queue: Dict[int, Any] = {i: deque() for i in range(num_robots)}
        # Per-robot failure queue, drained each step by gapo_env_real so the env can recover instead of freezing.
        self._failed_task_queue: Dict[int, Any] = {i: deque() for i in range(num_robots)}

        # Deployment loop can poll this flag before sending new commands.
        self.emergency_stop_triggered: bool = False

        for robot_id in range(num_robots):
            client = _WebSocketClient(bridge_url=bridge_url)
            self._clients[robot_id] = client
            self._submitters[robot_id] = _Dispatcher(client, graph_state=None)

        self._logger.info(
            f"ROSBridgeRobotBackend initialized: {num_robots} robot(s), bridge_url={bridge_url}"
        )

    async def start(self) -> None:
        """Connect all robot clients and start listening for telemetry; call before get_telemetry() or send_path_command()."""
        self._running = True
        for robot_id, client in self._clients.items():
            # Capture robot_id in closure
            def make_task_status_handler(rid):
                async def handler(data):
                    await self._handle_task_status(rid, data)

                return handler

            client.on_task_status = make_task_status_handler(robot_id)
            client.on_emergency_stop = self._handle_emergency_stop
            task = asyncio.create_task(self._listen_with_reconnect(robot_id, client))
            self._listen_tasks[robot_id] = task
        self._logger.info(f"ROSBridgeRobotBackend started: {self._num_robots} robot(s)")

    async def stop(self) -> None:
        """Cancel listen tasks and disconnect all robot clients."""
        self._running = False
        for task in self._listen_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        for client in self._clients.values():
            await client.disconnect()
        self._listen_tasks.clear()
        self._logger.info("ROSBridgeRobotBackend stopped")

    async def _listen_with_reconnect(self, robot_id: int, client: Any) -> None:
        """Connect and listen for a single robot client with exponential backoff reconnection, until stop() is called."""
        backoff = 1.0
        while self._running:
            try:
                if not client.is_connected:
                    connected = await client.connect()
                    if not connected:
                        self._logger.warning(
                            f"Robot {robot_id}: bridge connection failed, "
                            f"retrying in {backoff:.1f}s..."
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2.0, 30.0)
                        continue

                backoff = 1.0
                await client.listen()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(
                    f"Robot {robot_id}: listener error: {e}. Retrying in {backoff:.1f}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _handle_task_status(self, robot_id: int, data: Dict) -> None:
        """Route task_status messages from the bridge to registered callbacks."""
        task_id = data.get("task_id")
        status = data.get("status")

        if status == "completed":
            self._logger.info(f"Robot {robot_id}: task {task_id} completed")
            # Optimistically mark robot available in cached telemetry
            client = self._clients.get(robot_id)
            if client and client.robot_telemetry:
                client.robot_telemetry.active_task_id = None
            # Buffer for gapo_env_real to drain synchronously each step
            if task_id is not None:
                self._completed_task_queue[robot_id].append(task_id)
            if self.on_task_completed:
                if asyncio.iscoroutinefunction(self.on_task_completed):
                    await self.on_task_completed(robot_id, task_id)
                else:
                    self.on_task_completed(robot_id, task_id)

        elif status in ("failed", "canceled"):
            error = data.get("error")
            self._logger.error(f"Robot {robot_id}: task {task_id} {status}: {error}")
            # Buffer for gapo_env_real to drain synchronously each step (recovery).
            if task_id is not None:
                self._failed_task_queue[robot_id].append(task_id)
            if self.on_task_failed:
                if asyncio.iscoroutinefunction(self.on_task_failed):
                    await self.on_task_failed(robot_id, task_id, error)
                else:
                    self.on_task_failed(robot_id, task_id, error)

        elif status == "in_progress":
            self._logger.debug(
                f"Robot {robot_id}: task {task_id} in progress "
                f"{data.get('progress_percent', 0):.1f}%"
            )

    async def _handle_emergency_stop(self, data: Dict) -> None:
        """Terminate the process immediately when an active emergency stop is received."""
        self.emergency_stop_triggered = True
        self._logger.critical(
            "EMERGENCY STOP received — terminating deployment process immediately."
        )
        os.kill(os.getpid(), signal.SIGTERM)

    def set_graph_state(self, graph_state: Any) -> None:
        """Provide graph context so task submitters can resolve node indices to (x,y); call before send_path_command."""
        self._graph_state = graph_state
        for submitter in self._submitters.values():
            submitter.graph_state = graph_state

    def get_telemetry(self, robot_id: int) -> Optional[RobotTelemetryData]:
        """Get current telemetry snapshot from a robot, or None if no data has been received yet."""
        client = self._clients.get(robot_id)
        if client is None:
            self._logger.warning(f"No client registered for robot_id={robot_id}")
            return None

        raw = client.get_robot_state()
        if raw is None:
            return None  # No data received yet from bridge

        return self._convert_telemetry(robot_id, raw)

    def get_all_telemetry(self) -> List[RobotTelemetryData]:
        result = []
        for robot_id in range(self._num_robots):
            tel = self.get_telemetry(robot_id)
            if tel is not None:
                result.append(tel)
        return result

    def send_path_command(
        self, robot_id: int, path: List[int], task_id: int, num_items: int
    ) -> None:
        """Command a robot to follow a path; synchronous wrapper that schedules the async send on the running event loop."""
        coro = self._async_send_path_command(robot_id, path, task_id, num_items)
        try:
            loop = asyncio.get_running_loop()
            # Deployment context: called from within a running event loop.
            loop.create_task(coro)
        except RuntimeError:
            # No running event loop — sync caller (e.g. test scripts).
            asyncio.run(coro)
        except Exception as e:
            self._logger.error(f"send_path_command failed for robot {robot_id}: {e}", exc_info=True)

    async def _async_send_path_command(
        self, robot_id: int, path: List[int], task_id: int, num_items: int
    ) -> bool:
        """Async implementation of send_path_command; converts path and task info into a bridge task and submits via WebSocket."""
        submitter = self._submitters.get(robot_id)
        if submitter is None:
            self._logger.error(f"No submitter registered for robot_id={robot_id}")
            return False

        try:
            proxy = _BridgeTaskProxy(
                task_id=task_id,
                planned_path=path,
                num_items=num_items,
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
        return self._num_robots

    def get_location_inventories(self) -> Dict[str, Any]:
        """Get latest location inventories from the bridge; shared across robots, sourced from client 0."""
        client = self._clients.get(0)
        if client is None:
            return {}
        return client.get_all_inventories()

    def pop_completed_tasks(self, robot_id: int) -> List[int]:
        """Drain and return task IDs completed since the last call; not a persistent history."""
        q = self._completed_task_queue.get(robot_id)
        if not q:
            return []
        result = list(q)
        q.clear()
        return result

    def pop_failed_tasks(self, robot_id: int) -> List[int]:
        """Drain and return task IDs that failed/canceled since the last call, same drain-on-read semantics as pop_completed_tasks()."""
        q = self._failed_task_queue.get(robot_id)
        if not q:
            return []
        result = list(q)
        q.clear()
        return result

    def _convert_telemetry(self, robot_id: int, raw: Any) -> RobotTelemetryData:
        """Convert ROS bridge telemetry format to RobotTelemetryData, including quaternion-to-heading conversion."""
        pos = raw.position or (0.0, 0.0, 0.0)

        quat = raw.orientation or {}
        qx = quat.get("x", 0.0)
        qy = quat.get("y", 0.0)
        qz = quat.get("z", 0.0)
        qw = quat.get("w", 1.0)
        # Standard quaternion to yaw formula
        heading = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        vel = raw.velocity or {}
        speed = abs(vel.get("linear_x", 0.0))

        return RobotTelemetryData(
            robot_id=robot_id,
            timestamp=_to_epoch_seconds(raw.timestamp),
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
