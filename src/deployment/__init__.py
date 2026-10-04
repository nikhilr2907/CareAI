"""
Deployment module - interfaces and implementations for robot communication.

- MockRobotBackend: Uses local RobotSimulator (for training/testing)
- ROSBridgeRobotBackend: Connects to the external ROS bridge via WebSocket (for deployment)
- RobotBridgeWebSocketClient: Low-level WebSocket client for bridge communication
- RobotBridgeTaskDispatcher: Converts CareRobotics tasks to bridge wire format
"""

from .robot_backend import (
    RobotBackend,
    MockRobotBackend,
    ROSBridgeRobotBackend,
    RobotTelemetryData,
)
from .robot_bridge_websocket_client import RobotBridgeWebSocketClient
from .robot_bridge_state_sync import RobotBridgeTaskDispatcher

__all__ = [
    "RobotBackend",
    "MockRobotBackend",
    "ROSBridgeRobotBackend",
    "RobotTelemetryData",
    "RobotBridgeWebSocketClient",
    "RobotBridgeTaskDispatcher",
]
