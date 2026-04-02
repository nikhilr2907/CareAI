import numpy as np
from typing import Optional
from ..environment.robot.robot_telemetry import RobotTelemetry


class TelemetryEffectsManager:
    def __init__(self, enable: bool = True, graph_bounds: tuple = None):
        self.enable = enable
        self.graph_bounds = graph_bounds  # (min_x, max_x, min_y, max_y)

    def apply_noise(self, telemetry: Optional[RobotTelemetry],
                    elapsed_minutes: float,
                    num_nodes: int = None,
                    num_edges: int = None) -> Optional[RobotTelemetry]:
        """
        Inject Phase 1 realistic telemetry noise from ESP32 hardware.

        Args:
            telemetry: RobotTelemetry object (modified in-place)
            elapsed_minutes: Time since robot reset/initialization

        Returns:
            Modified telemetry object (same object)
        """
        if telemetry is None or not self.enable:
            return telemetry

        # Clamp elapsed time to avoid unrealistic drift
        elapsed_minutes = max(0.0, min(elapsed_minutes, 60.0))  # Cap at 1 hour

        # Position: accumulating dead reckoning drift (5-10% per minute)
        # Real ESP32 dead reckoning accumulates error ~0.075m per minute
        drift_std = 0.075 * elapsed_minutes
        telemetry.x += np.random.normal(0, max(drift_std, 0.01))
        telemetry.y += np.random.normal(0, max(drift_std, 0.01))

        # Velocity: 25% dropout rate (command didn't execute/wheel slip/sensor lag)
        # When available, add measurement noise
        if np.random.random() < 0.25:
            telemetry.velocity_ms = None
        else:
            telemetry.velocity_ms = max(0.0, telemetry.velocity_ms + np.random.normal(0, 0.1))

        # Heading: IMU drift (BNO055 drifts 1-2° per minute)
        heading_drift_deg = np.random.normal(0, 1.5 * elapsed_minutes)
        telemetry.heading += np.deg2rad(heading_drift_deg)

        # Position bounds validation: clamp to graph bounds
        if self.graph_bounds is not None:
            min_x, max_x, min_y, max_y = self.graph_bounds
            telemetry.x = np.clip(telemetry.x, min_x, max_x)
            telemetry.y = np.clip(telemetry.y, min_y, max_y)

        # Graph position validation: check indices are valid
        if num_nodes is not None:
            if telemetry.current_node_index is not None:
                if not (0 <= telemetry.current_node_index < num_nodes):
                    telemetry.current_node_index = None

        if num_edges is not None:
            if telemetry.current_edge_index is not None:
                if not (0 <= telemetry.current_edge_index < num_edges):
                    telemetry.current_edge_index = None
                    telemetry.edge_progress = 0.0

        # Velocity None handling: convert to 0.0 for state dict
        if telemetry.velocity_ms is None:
            telemetry.velocity_ms = 0.0

        return telemetry
