"""
Inventory-driven task generation for hospital logistics.
Generates tasks based on stock levels and consumption rates.
"""
import numpy as np
from typing import List, Tuple, Dict
from .task_state import Task, TaskQueue

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _parse_time_hhmm(t: str) -> float:
    hh, mm = t.split(":")
    return int(hh) + int(mm) / 60.0


def _intraday_weight(intraday_profile: Dict[str, float], current_time_seconds: float) -> Tuple[float, float]:
    """Return (weight, bin_hours) for current time of day."""
    if not intraday_profile:
        return 1.0, 24.0
    t_hours = (current_time_seconds % 86400) / 3600.0
    for window, weight in intraday_profile.items():
        start, end = window.split("-")
        start_h = _parse_time_hhmm(start)
        end_h = _parse_time_hhmm(end)
        if end_h < start_h:
            in_bin = t_hours >= start_h or t_hours < end_h
            bin_hours = (24.0 - start_h) + end_h
        else:
            in_bin = start_h <= t_hours < end_h
            bin_hours = end_h - start_h
        if in_bin:
            return float(weight), max(bin_hours, 1e-3)
    return 1.0, 24.0


def _weekday_multiplier(weekday_multipliers: Dict[str, float], current_time_seconds: float, start_day: int = 0) -> float:
    if not weekday_multipliers:
        return 1.0
    day_idx = int(current_time_seconds // 86400) % 7
    day_name = _WEEKDAYS[(start_day + day_idx) % 7]
    return float(weekday_multipliers.get(day_name, 1.0))


def _compute_sku_rate(node, sku_id: str, graph_state, current_time_seconds: float) -> float:
    """Compute expected hourly consumption rate for a SKU."""
    if not node.consumption_enabled:
        return 0.0
    sku_db = getattr(graph_state, "sku_database", None)
    if not sku_db or sku_id not in sku_db:
        return 0.0
    sku_data = sku_db[sku_id]
    consumption = sku_data.get("consumption_model", {})
    daily_dist = consumption.get("daily_distribution", {})
    daily_mean = float(daily_dist.get("mean", 0.0))
    intraday_profile = consumption.get("intraday_profile", {})
    weekday_multipliers = consumption.get("weekday_multiplier", {})

    weight, bin_hours = _intraday_weight(intraday_profile, current_time_seconds)
    weekday_mult = _weekday_multiplier(weekday_multipliers, current_time_seconds)

    # Demand scaling
    served_beds = max(0, int(getattr(node, "served_beds", 0)))
    scale = served_beds / 10.0
    if scale <= 0:
        return 0.0

    category_key = node.sku_inventory.get(sku_id, {}).get("category")
    area_mult = node.sku_inventory.get(sku_id, {}).get("area_multiplier", 1.0)

    demand_profiles = getattr(graph_state, "demand_profiles", {}) or {}
    dept_multipliers = demand_profiles.get("department_category_multipliers_normalized", {})
    dept_tag = getattr(node, "department_tag", None) or consumption.get("base_context", {}).get("baseline_department_tag")
    dept_cat_mult = 1.0
    if dept_tag and dept_tag in dept_multipliers:
        dept_cat_mult = float(dept_multipliers[dept_tag].get(category_key, dept_multipliers[dept_tag].get("*", 1.0)))

    scale_factor = float(getattr(graph_state, "consumption_scale", 1.0))
    hourly_rate = (daily_mean * weight / bin_hours) * scale * dept_cat_mult * float(area_mult) * weekday_mult
    hourly_rate *= scale_factor
    return max(0.0, hourly_rate)


def generate_inventory_tasks(
    graph_state,
    current_time: float,
    next_task_id: int = 0
) -> Tuple[List[Task], int]:
    """
    Generate tasks based on inventory needs.
    Creates replenishment tasks for locations that need restocking.

    Args:
        graph_state: GraphState with inventory information
        current_time: Current simulation time (seconds)
        next_task_id: Next available task ID

    Returns:
        Tuple of (list of new tasks, next task ID)
    """
    new_tasks = []
    task_id = next_task_id

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
        if dest_node.sku_inventory:
            for sku_id, sku_data in dest_node.sku_inventory.items():
                stock = float(sku_data.get("stock", 0.0))
                reorder = float(sku_data.get("reorder", 0.0))
                par_level = float(sku_data.get("par", 0.0))
                max_level = float(sku_data.get("max", 0.0))
                if max_level <= 0:
                    continue
                if stock > reorder:
                    continue

                rate = _compute_sku_rate(dest_node, sku_id, graph_state, current_time)
                tts_hours = (stock / rate) if rate > 0 else float('inf')
                tts_seconds = tts_hours * 3600
                deadline = current_time + max(tts_seconds, 60)

                if par_level > 0:
                    target = par_level
                else:
                    target = max_level
                delivery_amount = min(max_level - stock, max(target - stock, 1.0))
                num_items = max(1, int(delivery_amount))

                urgency = 5 if tts_hours < 0.5 else 4 if tts_hours < 1.0 else 3 if tts_hours < 2.0 else 2
                estimated_duration = 120.0

                task = Task(
                    task_id=task_id,
                    from_location_index=central_storage_idx,
                    to_location_index=dest_idx,
                    manual_priority=urgency,
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
                task_id += 1
        else:
            if dest_node.needs_restock:
                target_stock = dest_node.buffer_time * dest_node.consumption_rate
                delivery_amount = min(
                    dest_node.max_stock - dest_node.stock_level,
                    target_stock
                )
                num_items = max(1, int(delivery_amount))
                urgency = dest_node.urgency_level
                tts_seconds = dest_node.time_to_stockout * 3600
                deadline = current_time + max(tts_seconds, 60)
                estimated_duration = 120.0
                task = Task(
                    task_id=task_id,
                    from_location_index=central_storage_idx,
                    to_location_index=dest_idx,
                    manual_priority=urgency,
                    deadline=deadline,
                    arrival_time=current_time,
                    estimated_duration=estimated_duration,
                    task_type='replenishment',
                    num_items=num_items,
                    source_stock_level=dest_node.stock_level,
                    time_to_stockout=dest_node.time_to_stockout
                )
                new_tasks.append(task)
                task_id += 1

    return new_tasks, task_id


def generate_random_ad_hoc_tasks(
    graph_state,
    current_time: float,
    num_tasks: int,
    next_task_id: int = 0
) -> Tuple[List[Task], int]:
    """
    Generate random ad-hoc tasks (inter-ward transfers, pharmacy deliveries).

    Args:
        graph_state: GraphState object
        current_time: Current simulation time
        num_tasks: Number of ad-hoc tasks to generate
        next_task_id: Next available task ID

    Returns:
        Tuple of (list of new tasks, next task ID)
    """
    new_tasks = []
    task_id = next_task_id
    num_nodes = len(graph_state.nodes)

    for _ in range(num_tasks):
        # Random source and destination
        from_idx = np.random.randint(0, num_nodes)
        to_idx = np.random.randint(0, num_nodes)

        # Allow same-room tasks some of the time
        if np.random.random() >= 0.3:
            while to_idx == from_idx:
                to_idx = np.random.randint(0, num_nodes)

        # Random task type
        task_type = np.random.choice(
            ['ad_hoc', 'returns', 'emergency'],
            p=[0.7, 0.25, 0.05]
        )

        # Priority based on type
        if task_type == 'emergency':
            priority = 5
            deadline_offset = np.random.uniform(60, 300)  # 1-5 minutes
        elif task_type == 'ad_hoc':
            priority = np.random.randint(2, 4)
            deadline_offset = np.random.uniform(600, 1800)  # 10-30 minutes
        else:  # returns
            priority = np.random.randint(1, 3)
            deadline_offset = np.random.uniform(1800, 3600)  # 30-60 minutes

        # Random number of items
        num_items = np.random.randint(1, 5)

        task = Task(
            task_id=task_id,
            from_location_index=from_idx,
            to_location_index=to_idx,
            manual_priority=priority,
            deadline=current_time + deadline_offset,
            arrival_time=current_time,
            estimated_duration=np.random.uniform(60, 180),
            task_type=task_type,
            num_items=num_items
        )

        new_tasks.append(task)
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
            node.consumption_rate = total_rate
            node.recalc_category_inventory()
            for cat, rate in category_rates.items():
                if cat in node.category_inventory:
                    node.category_inventory[cat]["rate"] = rate
        elif node.consumption_rate > 0:
            node.consume_stock(time_delta_hours)
