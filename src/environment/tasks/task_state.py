from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Task:
    task_id: int
    from_location_index: int  # Node index
    to_location_index: int  # Node index

    deadline: float  # Absolute time when task must be completed
    arrival_time: float  # When this task entered the system

    # Task metadata
    estimated_duration: float  # Expected time to complete task
    task_type: str  # 'replenishment' in normal operation; 'emergency_manual' for manual shutdown
    num_items: int = 1  # Number of items to deliver (for capacity management)

    # Inventory context
    time_to_stockout: float = 999.0  # Legacy alias for initial time-to-stockout, in hours.
    initial_time_to_stockout: Optional[float] = None
    current_time_to_stockout: Optional[float] = None
    original_deadline: Optional[float] = None
    current_deadline: Optional[float] = None
    sku_id: Optional[str] = None
    category_key: Optional[str] = None
    category_id: float = -1.0
    sku_stock_level: float = 0.0
    initial_sku_stock_level: Optional[float] = None
    current_sku_stock_level: Optional[float] = None
    sku_stock_level_at_assign: Optional[float] = None
    sku_stock_level_at_collection: Optional[float] = None
    sku_stock_level_at_dropoff: Optional[float] = None
    sku_max_level: float = 0.0
    reorder_point: float = 0.0
    par_level: float = 0.0
    demand_location_index: Optional[int] = None

    # Computed fields (set by environment)
    queue_position: int = 0  # Position in sorted queue (0 = most urgent)
    planned_path: List[int] = field(default_factory=list)  # Inclusive node path for the current leg

    # Assignment tracking
    is_assigned: bool = False
    assigned_robot_id: Optional[int] = None
    leg_type: str = "full"  # "pickup", "dropoff", or "full"
    learned_score: float = 0.0

    # Execution timing
    execution_start_time: Optional[float] = None  # When robot first starts navigating this task

    # Task source tracking
    source: str = "unknown"  # 'deterministic', 'factoriser'

    # Intra-room coordinates (for precise positioning within nodes)
    from_coordinates: Optional[tuple] = None  # (x, y) exact pickup point in meters
    to_coordinates: Optional[tuple] = None    # (x, y) exact delivery point in meters

    def __post_init__(self):
        """Populate live inventory fields from creation-time context by default."""
        if self.initial_time_to_stockout is None:
            self.initial_time_to_stockout = self.time_to_stockout
        if self.current_time_to_stockout is None:
            self.current_time_to_stockout = self.initial_time_to_stockout
        if self.original_deadline is None:
            self.original_deadline = self.deadline
        if self.current_deadline is None:
            self.current_deadline = self.deadline
        if self.initial_sku_stock_level is None:
            self.initial_sku_stock_level = self.sku_stock_level
        if self.current_sku_stock_level is None:
            self.current_sku_stock_level = self.sku_stock_level
        if self.demand_location_index is None:
            self.demand_location_index = self.to_location_index

    def get_current_time_to_stockout(self) -> float:
        """Return the latest known time-to-stockout, falling back to creation-time TTS."""
        if self.current_time_to_stockout is not None:
            return self.current_time_to_stockout
        if self.initial_time_to_stockout is not None:
            return self.initial_time_to_stockout
        return self.time_to_stockout

    def get_current_sku_stock_level(self) -> float:
        """Return the latest known SKU stock level for this task's demand location."""
        if self.current_sku_stock_level is not None:
            return self.current_sku_stock_level
        return self.sku_stock_level

    def get_age(self, current_time: float) -> float:
        """Calculate how long this task has been waiting."""
        return current_time - self.arrival_time

    def get_time_to_deadline(self, current_time: float) -> float:
        """Calculate time remaining until the latest known deadline."""
        deadline = self.current_deadline if self.current_deadline is not None else self.deadline
        return deadline - current_time

    def get_features(self, current_time: float) -> np.ndarray:
        """
        Extract task features for ML model.

        Returns features: [to_idx, duration, age, time_to_deadline,
                          queue_position, num_items, time_to_stockout,
                          category_id, sku_stock_ratio, sku_reorder_ratio]
        Total: 10 features
        """
        age = self.get_age(current_time)
        time_to_deadline = self.get_time_to_deadline(current_time)
        current_stock = self.get_current_sku_stock_level()
        stock_ratio = (current_stock / self.sku_max_level) if self.sku_max_level > 0 else 0.0
        reorder_ratio = (self.reorder_point / self.sku_max_level) if self.sku_max_level > 0 else 0.0

        return np.array([
            float(self.to_location_index),
            self.estimated_duration,
            age,
            time_to_deadline,
            float(self.queue_position),
            float(self.num_items),
            self.get_current_time_to_stockout(),
            float(self.category_id),
            stock_ratio,
            reorder_ratio,
        ], dtype=np.float32)


@dataclass
class TaskQueue:
    """Manages a queue of tasks with ranking capabilities."""
    tasks: List[Task] = field(default_factory=list)

    def add_task(self, task: Task):
        """Add a task to the queue."""
        self.tasks.append(task)

    def remove_task(self, task_id: int):
        """Remove a task by ID."""
        self.tasks = [t for t in self.tasks if t.task_id != task_id]

    def get_task(self, task_index: int) -> Optional[Task]:
        """Get task by index."""
        if 0 <= task_index < len(self.tasks):
            return self.tasks[task_index]
        return None

    def get_task_count(self) -> int:
        """Get current number of tasks."""
        return len(self.tasks)

    def clear(self):
        """Clear all tasks from queue."""
        self.tasks.clear()
