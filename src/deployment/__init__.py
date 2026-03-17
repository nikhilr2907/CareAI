"""
Deployment module - interfaces and implementations for robot communication.

This module provides abstract and concrete implementations for communicating with robots:
- MockRobotBridge: Uses local RobotSimulator (for training/testing)
- ROSRobotBridge: Connects to real robots via ROS 2 WebSocket bridge (for deployment)

Also includes:
- ROSBridgeClient: Low-level WebSocket client for bridge communication
- ROSBridgeIntegration: Integration layer syncing bridge data into CareRobotics
- ROSBridgeTaskSubmitter: Converts CareRobotics tasks to bridge format
"""

from .robot_bridge import (
    RobotBridge,
    MockRobotBridge,
    ROSRobotBridge,
    RobotTelemetryData,
)
from .ros_bridge_client import ROSBridgeClient
from .ros_bridge_integration import (
    ROSBridgeIntegration,
    ROSBridgeTaskSubmitter,
)

__all__ = [
    "RobotBridge",
    "MockRobotBridge",
    "ROSRobotBridge",
    "RobotTelemetryData",
    "ROSBridgeClient",
    "ROSBridgeIntegration",
    "ROSBridgeTaskSubmitter",
]
