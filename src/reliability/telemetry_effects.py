from typing import Optional

import numpy as np

from ..environment.robot.robot_telemetry import RobotTelemetry


class TelemetryEffectsManager:
    def __init__(
        self, enable: bool = True, graph_bounds: tuple[float, float, float, float] | None = None
    ):
        self.enable = enable
        self.graph_bounds = graph_bounds  # (min_x, max_x, min_y, max_y)
        self._drift_rates: dict[int, tuple[float, float, float]] = {}
        self._wiener_state: dict[int, tuple[float, float, float]] = {}
        self._last_landmark_time: dict[int, float] = {}

    # Wiener component std per sqrt(minute), smaller than the ramp so the systematic bias
    # stays dominant between landmarks; this is the residual higher-frequency wobble around it.
    _WIENER_SIGMA_POS = 0.03
    _WIENER_SIGMA_HEADING_DEG = 0.6
    _CONFIDENCE_TAU_MIN = 5.0  # time constant for position_confidence's exponential decay

    def reset_robot(self, robot_id: int, current_time: float = 0.0) -> None:
        """Sample this robot's dead-reckoning drift rate once per episode (a hardware property)."""
        self._drift_rates[robot_id] = (
            float(np.random.normal(0, 0.075)),
            float(np.random.normal(0, 0.075)),
            float(np.random.normal(0, 1.5)),
        )
        self.mark_landmark(robot_id, current_time)

    def mark_landmark(self, robot_id: int, current_time: float) -> None:
        """Resync against a known position (arriving at a node) -- wipes accumulated error."""
        self._wiener_state[robot_id] = (0.0, 0.0, 0.0)
        self._last_landmark_time[robot_id] = current_time

    def apply_noise(
        self,
        telemetry: Optional[RobotTelemetry],
        current_time: float,
        dt_seconds: float = 1.0,
        num_nodes: int | None = None,
        num_edges: int | None = None,
    ) -> Optional[RobotTelemetry]:
        """Inject Phase 1 realistic telemetry noise from ESP32 hardware, modifying telemetry in-place."""
        if telemetry is None or not self.enable:
            return telemetry

        # At a node, true position is known -- a stationary robot integrates no odometry error,
        # so skip drift entirely rather than relying on elapsed time happening to be ~0.
        if telemetry.current_edge_index is not None:
            last_landmark = self._last_landmark_time.get(telemetry.robot_id, current_time)
            elapsed_minutes = max(0.0, current_time - last_landmark) / 60.0

            dx_rate, dy_rate, heading_rate_deg = self._drift_rates.get(
                telemetry.robot_id, (0.0, 0.0, 0.0)
            )
            wx, wy, wh = self._wiener_state.get(telemetry.robot_id, (0.0, 0.0, 0.0))
            dt_minutes_sqrt = (max(dt_seconds, 0.0) / 60.0) ** 0.5
            wx += np.random.normal(0, self._WIENER_SIGMA_POS * dt_minutes_sqrt)
            wy += np.random.normal(0, self._WIENER_SIGMA_POS * dt_minutes_sqrt)
            wh += np.random.normal(0, self._WIENER_SIGMA_HEADING_DEG * dt_minutes_sqrt)
            self._wiener_state[telemetry.robot_id] = (wx, wy, wh)

            telemetry.x += dx_rate * elapsed_minutes + wx
            telemetry.y += dy_rate * elapsed_minutes + wy
            telemetry.heading += np.deg2rad(heading_rate_deg * elapsed_minutes + wh)
            telemetry.position_confidence = float(np.exp(-elapsed_minutes / self._CONFIDENCE_TAU_MIN))
        else:
            telemetry.position_confidence = 1.0

        # 25% dropout rate (command didn't execute/wheel slip/sensor lag); otherwise add measurement noise
        if np.random.random() < 0.25:
            telemetry.velocity_ms = 0.0
        else:
            telemetry.velocity_ms = max(0.0, telemetry.velocity_ms + np.random.normal(0, 0.1))

        if self.graph_bounds is not None:
            min_x, max_x, min_y, max_y = self.graph_bounds
            telemetry.x = np.clip(telemetry.x, min_x, max_x)
            telemetry.y = np.clip(telemetry.y, min_y, max_y)

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
