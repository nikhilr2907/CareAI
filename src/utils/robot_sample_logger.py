"""
Robot-level telemetry sampling log for GAPO runs.

This logger is intentionally separate from TaskLogger:
- TaskLogger owns task lifecycle / reward / cost events
- RobotSampleLogger owns periodic robot execution-state samples
"""
import logging
from pathlib import Path
from typing import Iterable, Optional


class RobotSampleLogger:
    """Log periodic robot telemetry samples to a dedicated text file."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "robot_samples.log"

        self.logger = logging.getLogger("GAPO_RobotSamples")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    @staticmethod
    def _fmt_float(value: Optional[float], precision: int = 1) -> str:
        if value is None:
            return "?"
        return f"{float(value):.{precision}f}"

    @staticmethod
    def _fmt_int(value: Optional[int]) -> str:
        if value is None:
            return "?"
        return str(int(value))

    @staticmethod
    def _fmt_bool(value: bool) -> str:
        return "1" if value else "0"

    @staticmethod
    def _infer_status(robot) -> str:
        telemetry = getattr(robot, "telemetry", None)
        if telemetry is None:
            return "NO_TELEMETRY"
        if getattr(telemetry, "is_charging", False):
            return "CHARGING"
        if len(getattr(robot, "task_queue", [])) == 0:
            return "IDLE"
        if float(getattr(telemetry, "velocity_ms", 0.0)) > 0.1:
            return "ACTIVE"
        return "WAITING_AT_LOCATION"

    def build_sample(self, robot, sim_time: float) -> dict:
        """Build one execution-state sample for a single robot."""
        telemetry = getattr(robot, "telemetry", None)
        current_task = getattr(robot, "current_task", None)

        if telemetry is not None:
            x, y = getattr(robot, "current_position", (0.0, 0.0))
            node = getattr(telemetry, "current_node_index", None)
            edge = getattr(telemetry, "current_edge_index", None)
            progress = getattr(telemetry, "edge_progress", 0.0)
            velocity = getattr(telemetry, "velocity_ms", 0.0)
            battery = getattr(telemetry, "battery_level", 0.0)
            current_load = getattr(telemetry, "current_capacity", 0)
            charging = getattr(telemetry, "is_charging", False)
            available = getattr(telemetry, "is_available", False)
            remaining_path = getattr(telemetry, "remaining_path", []) or []
            eta_next = getattr(telemetry, "eta_to_next_node", None)
        else:
            x = y = 0.0
            node = edge = None
            progress = velocity = battery = None
            current_load = None
            charging = False
            available = False
            remaining_path = []
            eta_next = None

        task_id = getattr(current_task, "task_id", None) if current_task is not None else None
        parent_id = getattr(current_task, "parent_task_id", None) if current_task is not None else None
        leg_type = getattr(current_task, "leg_type", None) if current_task is not None else None
        task_type = getattr(current_task, "task_type", None) if current_task is not None else None

        task_age = None
        arrival_time = getattr(current_task, "arrival_time", None) if current_task is not None else None
        if arrival_time is not None:
            task_age = max(float(sim_time) - float(arrival_time), 0.0)

        leg_time = None
        if current_task is not None and getattr(robot, "travel_start_time", None) is not None:
            leg_time = max(float(sim_time) - float(robot.travel_start_time), 0.0)

        return {
            "robot_id": robot.robot_id,
            "sim_time": float(sim_time),
            "status": self._infer_status(robot),
            "x": x,
            "y": y,
            "current_node_index": node,
            "current_edge_index": edge,
            "edge_progress": progress,
            "velocity_ms": velocity,
            "battery_level": battery,
            "current_load": current_load,
            "task_queue_len": len(getattr(robot, "task_queue", [])),
            "overflow_queue_len": len(getattr(robot, "overflow_queue", [])),
            "is_charging": charging,
            "is_available": available,
            "target_node_index": getattr(robot, "target_node_index", None),
            "remaining_path_len": len(remaining_path),
            "eta_to_next_node": eta_next,
            "task_id": task_id,
            "parent_task_id": parent_id,
            "leg_type": leg_type,
            "task_type": task_type,
            "task_age": task_age,
            "leg_time": leg_time,
        }

    def format_sample(self, sample: dict) -> str:
        """Format a robot sample as a single log line."""
        return (
            f"SAMPLE robot={sample['robot_id']} "
            f"sim_time={self._fmt_float(sample.get('sim_time'), 1)}s "
            f"status={sample.get('status', '?')} "
            f"pos=({self._fmt_float(sample.get('x'), 2)},{self._fmt_float(sample.get('y'), 2)}) "
            f"node={self._fmt_int(sample.get('current_node_index'))} "
            f"edge={self._fmt_int(sample.get('current_edge_index'))} "
            f"progress={self._fmt_float(sample.get('edge_progress'), 2)} "
            f"vel={self._fmt_float(sample.get('velocity_ms'), 2)} "
            f"battery={self._fmt_float(sample.get('battery_level'), 3)} "
            f"load={self._fmt_int(sample.get('current_load'))} "
            f"q={self._fmt_int(sample.get('task_queue_len'))} "
            f"ov={self._fmt_int(sample.get('overflow_queue_len'))} "
            f"charging={self._fmt_bool(bool(sample.get('is_charging')))} "
            f"available={self._fmt_bool(bool(sample.get('is_available')))} "
            f"target={self._fmt_int(sample.get('target_node_index'))} "
            f"rem_path={self._fmt_int(sample.get('remaining_path_len'))} "
            f"eta_next={self._fmt_float(sample.get('eta_to_next_node'), 1)}s "
            f"task={self._fmt_int(sample.get('task_id'))} "
            f"parent={self._fmt_int(sample.get('parent_task_id'))} "
            f"leg={sample.get('leg_type') or '?'} "
            f"task_type={sample.get('task_type') or '?'} "
            f"task_age={self._fmt_float(sample.get('task_age'), 1)}s "
            f"leg_time={self._fmt_float(sample.get('leg_time'), 1)}s"
        )

    def log_sample(self, robot, sim_time: float) -> dict:
        """Log one execution-state sample for a single robot."""
        sample = self.build_sample(robot, sim_time)
        self.logger.info(self.format_sample(sample))
        return sample

    def log_samples(self, robots: Iterable, sim_time: float):
        """Log one sample line per robot."""
        for robot in robots:
            self.log_sample(robot, sim_time)
