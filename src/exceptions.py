class TelemetryUnavailableError(Exception):
    """Raised when robot telemetry is required but has not yet been received from the ROS bridge."""
