"""
Task Source Layer - Interface for receiving tasks from hospital systems.

Abstracts where tasks come from:
- Mock task generator (for testing)
- Hospital management system API
- Manual task entry system
- Inventory monitoring system
"""
from typing import List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class IncomingTask:
    """Task from external hospital system."""
    task_id: int
    task_type: str  # 'replenishment', 'returns', 'ad_hoc'
    from_location: int
    to_location: int
    num_items: int
    priority: int  # 1-5
    deadline: Optional[float] = None
    created_at: float = 0.0


class TaskSource:
    """Abstract interface for task sources."""

    def get_new_tasks(self, current_time: float) -> List[IncomingTask]:
        """Get newly arrived tasks since last check."""
        raise NotImplementedError


class MockTaskSource(TaskSource):
    """
    Mock task source using environment's task generators.

    In deployment, replace with APITaskSource or DatabaseTaskSource.
    """

    def __init__(self, graph_state, next_task_id: int = 0):
        """
        Initialize with graph state for generating realistic tasks.

        Args:
            graph_state: Hospital graph for task generation
            next_task_id: Starting task ID
        """
        self.graph_state = graph_state
        self.next_task_id = next_task_id
        self.last_inventory_check = 0.0
        self.inventory_check_interval = 10.0  # Check every 10 seconds

    def get_new_tasks(self, current_time: float) -> List[IncomingTask]:
        """Generate mock tasks based on inventory and random events."""
        from ..environment.tasks.task_generator import (
            generate_inventory_tasks,
            generate_random_ad_hoc_tasks
        )

        new_tasks = []

        # Check inventory periodically
        if current_time - self.last_inventory_check >= self.inventory_check_interval:
            inventory_tasks, self.next_task_id = generate_inventory_tasks(
                self.graph_state, current_time, self.next_task_id
            )

            for task in inventory_tasks:
                new_tasks.append(IncomingTask(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    from_location=task.from_location_index,
                    to_location=task.to_location_index,
                    num_items=task.num_items,
                    priority=task.manual_priority,
                    deadline=task.deadline,
                    created_at=current_time
                ))

            self.last_inventory_check = current_time

        # Random ad-hoc tasks
        if np.random.random() < 0.1:  # 10% chance
            ad_hoc_tasks, self.next_task_id = generate_random_ad_hoc_tasks(
                self.graph_state, current_time, 1, self.next_task_id
            )

            for task in ad_hoc_tasks:
                new_tasks.append(IncomingTask(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    from_location=task.from_location_index,
                    to_location=task.to_location_index,
                    num_items=task.num_items,
                    priority=task.manual_priority,
                    deadline=task.deadline,
                    created_at=current_time
                ))

        return new_tasks


class APITaskSource(TaskSource):
    """
    Task source from hospital management system API.

    PLACEHOLDER - Implement when connecting to real hospital system.
    """

    def __init__(self, api_endpoint: str, api_key: str):
        """
        Initialize API connection.

        Args:
            api_endpoint: URL of hospital task API
            api_key: Authentication key
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.last_task_id = None
        raise NotImplementedError("API task source not yet implemented")

    def get_new_tasks(self, current_time: float) -> List[IncomingTask]:
        """Poll API for new tasks since last check."""
        # TODO: HTTP GET request to API
        # GET /api/tasks?since={last_task_id}
        raise NotImplementedError


class DatabaseTaskSource(TaskSource):
    """
    Task source from hospital database.

    PLACEHOLDER - Implement when connecting to database.
    """

    def __init__(self, db_connection_string: str):
        """
        Initialize database connection.

        Args:
            db_connection_string: Database connection string
        """
        self.db_connection_string = db_connection_string
        raise NotImplementedError("Database task source not yet implemented")

    def get_new_tasks(self, current_time: float) -> List[IncomingTask]:
        """Query database for new tasks."""
        # TODO: SQL query for unassigned tasks
        raise NotImplementedError
