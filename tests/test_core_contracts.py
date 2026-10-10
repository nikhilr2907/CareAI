from dataclasses import dataclass

import numpy as np
import pytest

from src.analytics.metrics.helpers import MetricHelpers
from src.environment.graph.edge import GraphEdge
from src.environment.robot.robot_telemetry import RobotTelemetry, TelemetryPayload
from src.environment.tasks.task_state import Task, TaskQueue


def make_task(task_id: int = 1) -> Task:
    return Task(task_id, 0, 1, 600.0, 0.0, 30.0, "replenishment", sku_max_level=100.0)


def test_live_inventory_context_changes_features_without_overwriting_initial_values() -> None:
    task = make_task()
    task.current_sku_stock_level = 25.0
    task.current_time_to_stockout = 0.5
    task.current_deadline = 120.0
    features = task.get_features(60.0)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert features[-2] == pytest.approx(0.25)
    assert task.get_time_to_deadline(60.0) == 60.0
    assert task.get_current_time_to_stockout() == 0.5
    assert task.original_deadline == 600.0
    assert task.initial_sku_stock_level == 0.0


def test_empty_inventory_capacity_produces_finite_features() -> None:
    task = make_task()
    task.sku_max_level = 0.0
    assert np.isfinite(task.get_features(0.0)).all()


def test_removing_task_by_id_keeps_the_other_tasks() -> None:
    first, second = make_task(1), make_task(2)
    queue = TaskQueue([first, second])
    queue.remove_task(1)
    assert queue.get_task(0) is second
    assert queue.get_task(-1) is None
    assert queue.get_task(1) is None


def test_minimal_telemetry_payload_defaults_optional_state() -> None:
    payload: TelemetryPayload = {
        "timestamp": 1.0,
        "robot_id": 0,
        "x": 2.0,
        "y": 3.0,
        "heading": 0.0,
        "velocity_ms": 0.0,
        "is_moving": False,
        "battery_level": 0.8,
        "current_capacity": 0,
        "is_available": True,
    }
    telemetry = RobotTelemetry.from_dict(payload)
    assert telemetry.current_node_index is None
    assert telemetry.remaining_path == []
    assert not telemetry.is_charging
    assert telemetry.to_dict()["battery_level"] == 0.8


def test_opposing_traffic_cost_is_independent_of_robot_ids() -> None:
    edge = GraphEdge("store", "ward", distance_m=10.0)
    edge.active_robot_ids = [0, 1, 2]
    edge.active_robot_progress = {0: (0.2, 0, 1), 1: (0.4, 0, 1), 2: (0.1, 1, 0)}
    original_cost = edge.current_weight
    edge.active_robot_progress = {10: (0.2, 0, 1), 20: (0.4, 0, 1), 30: (0.1, 1, 0)}
    assert edge.current_weight == original_cost
    edge.active_robot_progress = {10: (0.2, 0, 1), 20: (0.4, 0, 1), 30: (0.1, 0, 1)}
    assert edge.current_weight < original_cost


@dataclass
class Location:
    center_x: float
    center_y: float


def test_spatial_metrics_accept_coordinate_only_objects() -> None:
    metrics = MetricHelpers([Location(0.0, 0.0), Location(10.0, 10.0)])
    assert metrics.get_location_quadrant(0) == 0
    assert metrics.get_location_quadrant(1) == 3
    assert metrics.get_location_quadrant(None) == 0
    assert metrics.get_location_quadrant(99) == 0
