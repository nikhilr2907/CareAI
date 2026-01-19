"""
Inventory-driven task generation for hospital logistics.
Generates tasks based on stock levels and consumption rates.
"""
import numpy as np
from typing import List, Tuple
from .task_state import Task, TaskQueue


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
        # Skip nodes without inventory (corridors, hubs)
        if dest_node.consumption_rate <= 0:
            continue

        # Skip storage nodes (they don't need restocking)
        if dest_node.node_type == 'storage':
            continue

        # Check if node needs restocking
        if dest_node.needs_restock:
            # Calculate delivery amount (refill to max or buffer level)
            target_stock = dest_node.buffer_time * dest_node.consumption_rate
            delivery_amount = min(
                dest_node.max_stock - dest_node.stock_level,
                target_stock
            )

            # Ensure at least 1 item
            num_items = max(1, int(delivery_amount))

            # Calculate urgency based on time to stockout
            urgency = dest_node.urgency_level

            # Calculate deadline based on time to stockout
            tts_seconds = dest_node.time_to_stockout * 3600  # Convert hours to seconds
            deadline = current_time + max(tts_seconds, 60)  # At least 60 seconds

            # Estimate travel time (rough approximation)
            estimated_duration = 120.0  # Default 2 minutes

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
        if node.consumption_rate > 0:
            node.consume_stock(time_delta_hours)
