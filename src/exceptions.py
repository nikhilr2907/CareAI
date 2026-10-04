"""
Project-wide custom exceptions.
"""


class TelemetryUnavailableError(Exception):
    """
    Raised when robot telemetry is required but has not yet been received
    from the ROS bridge.

    This is not a runtime fault — it indicates the environment attempted an
    operation (e.g. reset) that depends on knowing the robot's physical
    position before the bridge has published its first telemetry message.
    """
