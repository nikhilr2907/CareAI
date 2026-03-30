"""
Deployment module - interfaces and implementations for robot communication.

This module provides abstract and concrete implementations for communicating with robots:
- MockRobotBackend: Uses local RobotSimulator (for training/testing)
- ROSBridgeRobotBackend: Connects to the external ROS bridge via WebSocket (for deployment)

Also includes:
- RobotBridgeWebSocketClient: Low-level WebSocket client for bridge communication
- RobotBridgeStateSync: Integration layer syncing bridge data into CareRobotics
- RobotBridgeTaskDispatcher: Converts CareRobotics tasks to bridge format
"""

from .robot_backend import (
    RobotBackend,
    MockRobotBackend,
    ROSBridgeRobotBackend,
    RobotTelemetryData,
)
from .robot_bridge_websocket_client import RobotBridgeWebSocketClient
from .robot_bridge_state_sync import (
    RobotBridgeStateSync,
    RobotBridgeTaskDispatcher,
)

__all__ = [
    "RobotBackend",
    "MockRobotBackend",
    "ROSBridgeRobotBackend",
    "RobotTelemetryData",
    "RobotBridgeWebSocketClient",
    "RobotBridgeStateSync",
    "RobotBridgeTaskDispatcher",
]
