import numpy as np
from typing import List, Optional, Set, Tuple, Dict
from .task_state import Task, TaskQueue
from ..graph_helpers import dijkstra_shortest_path

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_MIN_REPLENISHMENT_DEADLINE_S = 300.0


def _parse_time_hhmm(t: str) -> float:
    hh, mm = t.split(":")
    return int(hh) + int(mm) / 60.0


def _weekday_multiplier(weekday_multipliers: Dict[str, float], current_time_seconds: float, start_day: int = 0) -> float:
    if not weekday_multipliers:
        return 1.0
    day_idx = int(current_time_seconds // 86400) % 7
    day_name = _WEEKDAYS[(start_day + day_idx) % 7]
    return float(weekday_multipliers.get(day_name, 1.0))


def _get_current_school_period(school_schedule: dict, current_time_seconds: float) -> Optional[dict]:
    """Return the active school period dict for current_time_seconds, or None if outside schedule.

    current_time_seconds is the simulation clock (0 = episode start). The episode is
    anchored to the first period's start_time (e.g. 08:30), so we offset by that
    amount before comparing against each period's wall-clock window.
    """
    if not school_schedule:
        return None
    periods = school_schedule.get("periods", [])
    if not periods:
        return None
    episode_start_h = _parse_time_hhmm(periods[0]["start_time"])
    wall_clock_h = (episode_start_h + (current_time_seconds % 86400) / 3600.0) % 24.0
    for period in periods:
        start_h = _parse_time_hhmm(period["start_time"])
        end_h = _parse_time_hhmm(period["end_time"])
        if start_h <= wall_clock_h < end_h:
            return period
    return None


def _school_period_weight(
    period_profile: dict,
    school_schedule: dict,
    current_time_seconds: float,
) -> Tuple[float, float]:
    """Return (weight, period_hours) for the currently active school period.

    weight        – the fraction of the SKU's daily demand that falls in this period,
                    taken directly from school_period_profile in the SKU's consumption_model.
    period_hours  – duration of the period in hours, used to convert the daily demand
                    fraction into an instantaneous hourly rate.

    Returns (0.0, 1.0) when the simulation clock is outside all defined periods so
    that consumption falls to zero between the end of one period and the start of the
    next (e.g. before 08:30 or after 16:30).
    """
    period = _get_current_school_period(school_schedule, current_time_seconds)
    if period is None:
        return 0.0, 1.0
    weight = float(period_profile.get(period["name"], 0.0))
    period_hours = max(
        _parse_time_hhmm(period["end_time"]) - _parse_time_hhmm(period["start_time"]),
        1e-3,
    )
    return weight, period_hours


def _compute_sku_rate(node, sku_id: str, graph_state, current_time_seconds: float) -> float:
    """Compute expected hourly consumption rate for a SKU at a given node (ILC mode).

    ILC mode: consumption_model contains 'school_period_profile' (named period keys).
    Demand is scaled by node.foot_traffic_weight (0–1). The active school period is
    resolved from graph_state.school_schedule.
    """
    if not node.consumption_enabled:
        return 0.0
    sku_db = getattr(graph_state, "sku_database", None)
    if not sku_db or sku_id not in sku_db:
        return 0.0
    sku_data = sku_db[sku_id]
    consumption = sku_data.get("consumption_model", {})
    daily_dist = consumption.get("daily_distribution", {})
    daily_mean = float(daily_dist.get("mean", 0.0))
    weekday_multipliers = consumption.get("weekday_multiplier", {})
    weekday_mult = _weekday_multiplier(weekday_multipliers, current_time_seconds)
    scale_factor = float(getattr(graph_state, "consumption_scale", 1.0))

    foot_traffic_weight = float(getattr(node, "foot_traffic_weight", 0.0))
    if foot_traffic_weight <= 0:
        return 0.0
    school_period_profile = consumption.get("school_period_profile", {})
    school_schedule = getattr(graph_state, "school_schedule", None)
    weight, period_hours = _school_period_weight(
        school_period_profile, school_schedule, current_time_seconds
    )
    hourly_rate = (daily_mean * weight / period_hours) * foot_traffic_weight * weekday_mult * scale_factor
    return max(0.0, hourly_rate)


def generate_inventory_tasks(
    graph_state,
    current_time: float,
    next_task_id: int = 0,
    existing_task_keys: Optional[Set[Tuple]] = None,
) -> Tuple[List[Task], int]:
    """
    Generate tasks based on inventory needs.
    Creates replenishment tasks for locations that need restocking.

    Args:
        graph_state: GraphState with inventory information
        current_time: Current simulation time (seconds)
        next_task_id: Next available task ID
        existing_task_keys: Set of (sku_id, dest_node_idx) tuples already covered
            by pending or in-flight tasks. Tasks matching an existing key are
            skipped so we never create duplicate replenishment demand. Pass None
            to skip deduplication (backwards-compatible).

    Returns:
        Tuple of (list of new tasks, next task ID)
    """
    new_tasks = []
    task_id = next_task_id

    # Work on a mutable copy so we can accumulate keys within this call too
    # (prevents duplicates within a single generate call for different SKUs
    # at the same dest node — unlikely but safe).
    seen_keys: Set[Tuple] = set(existing_task_keys) if existing_task_keys else set()

    # Find storage nodes (sources of supplies)
    storage_nodes = [
        i for i, node in enumerate(graph_state.nodes)
        if node.node_type == 'storage'
    ]

    if not storage_nodes:
        return new_tasks, task_id

    # Default to first storage node
    central_storage_idx = storage_nodes[0]

    # Check each node for inventory needs
    for dest_idx, dest_node in enumerate(graph_state.nodes):
        if dest_node.node_type == 'storage':
            continue
        if not dest_node.consumption_enabled:
            continue
        for sku_id, sku_data in dest_node.sku_inventory.items():
            stock = float(sku_data.get("stock", 0.0))
            reorder = float(sku_data.get("reorder", 0.0))
            par_level = float(sku_data.get("par", 0.0))
            max_level = float(sku_data.get("max", 0.0))
            if max_level <= 0:
                continue
            if stock > reorder:
                continue

            # Skip if this (sku, destination) is already pending or in-flight
            key = (sku_id, dest_idx)
            if key in seen_keys:
                continue

            rate = _compute_sku_rate(dest_node, sku_id, graph_state, current_time)
            tts_hours = (stock / rate) if rate > 0 else float('inf')
            tts_seconds = tts_hours * 3600
            deadline = current_time + max(tts_seconds, DEFAULT_MIN_REPLENISHMENT_DEADLINE_S)

            if par_level > 0:
                target = par_level
            else:
                target = max_level
            delivery_amount = min(max_level - stock, max(target - stock, 1.0))
            num_items = max(1, int(delivery_amount))

            _, estimated_duration = dijkstra_shortest_path(
                central_storage_idx, dest_idx, graph_state, len(graph_state.nodes)
            )

            task = Task(
                task_id=task_id,
                from_location_index=central_storage_idx,
                to_location_index=dest_idx,
                deadline=deadline,
                arrival_time=current_time,
                estimated_duration=estimated_duration,
                task_type='replenishment',
                num_items=num_items,
                source_stock_level=dest_node.stock_level,
                time_to_stockout=tts_hours,
                sku_id=sku_id,
                category_key=sku_data.get("category"),
                category_id=float(graph_state.category_order.index(sku_data.get("category"))) if getattr(graph_state, "category_order", None) and sku_data.get("category") in graph_state.category_order else -1.0,
                sku_stock_level=stock,
                sku_max_level=max_level,
                reorder_point=reorder,
                par_level=par_level
            )

            new_tasks.append(task)
            seen_keys.add(key)
            task_id += 1

    return new_tasks, task_id




def update_inventory_levels(graph_state, time_delta_hours: float):
    """
    Update stock levels at all nodes based on consumption rates.

    Args:
        graph_state: GraphState to update
        time_delta_hours: Time elapsed in hours
    """
    for node in graph_state.nodes:
        if node.sku_inventory and node.consumption_enabled:
            category_rates = {}
            total_stock = 0.0
            total_rate = 0.0
            for sku_id, sku_data in node.sku_inventory.items():
                rate = _compute_sku_rate(node, sku_id, graph_state, graph_state.current_time if hasattr(graph_state, "current_time") else 0.0)
                consumed = rate * time_delta_hours
                node.consume_sku(sku_id, consumed)
                total_stock += node.get_sku_stock(sku_id)
                total_rate += rate
                category = sku_data.get("category")
                if category:
                    category_rates[category] = category_rates.get(category, 0.0) + rate

            node.stock_level = total_stock
            node.recalc_category_inventory()
            for cat, rate in category_rates.items():
                if cat in node.category_inventory:
                    node.category_inventory[cat]["rate"] = rate
