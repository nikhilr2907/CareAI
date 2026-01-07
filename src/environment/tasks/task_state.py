from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Task:
    """Represents a single delivery task with urgency and temporal features."""
    task_id: int
    from_location_index: int  # Node index
    to_location_index: int  # Node index

    # Urgency features (for ranking)
    manual_priority: int  # 1-5 (explicit urgency from user/system)
    deadline: float  # Absolute time when task must be completed
    arrival_time: float  # When this task entered the system

    # Task metadata
    estimated_duration: float  # Expected time to complete task
    task_type: str  # 'replenishment', 'returns', 'ad_hoc', 'emergency'
    num_items: int = 1  # Number of items to deliver (for capacity management)

    # Inventory context
    source_stock_level: float = 0.0  # Stock level at destination when task created
    time_to_stockout: float = 999.0  # Hours until stockout when task created

    # Computed fields (set by ranking or environment)
    urgency_score: float = 0.0  # Computed by rank_tasks()
    queue_position: int = 0  # Position in sorted queue (0 = most urgent)

    # Assignment tracking
    is_assigned: bool = False
    assigned_robot_id: Optional[int] = None

    # Intra-room coordinates (for precise positioning within nodes)
    from_coordinates: Optional[tuple] = None  # (x, y) exact pickup point in meters
    to_coordinates: Optional[tuple] = None    # (x, y) exact delivery point in meters
    # If None, defaults to node center. If specified, enables intra-room navigation.

    def get_age(self, current_time: float) -> float:
        """Calculate how long this task has been waiting."""
        return current_time - self.arrival_time

    def get_time_to_deadline(self, current_time: float) -> float:
        """Calculate time remaining until deadline."""
        return self.deadline - current_time

    def get_features(self, current_time: float) -> np.ndarray:
        """
        Extract task features for ML model (NO manual_priority - handled by ranking).

        Returns features: [from_idx, to_idx, duration, age, time_to_deadline,
                          queue_position, num_items, time_to_stockout, task_type_flags(4)]
        Total: 12 features
        """
        age = self.get_age(current_time)
        time_to_deadline = self.get_time_to_deadline(current_time)

        return np.array([
            float(self.from_location_index),
            float(self.to_location_index),
            self.estimated_duration,
            age,
            time_to_deadline,
            float(self.queue_position),
            float(self.num_items),
            self.time_to_stockout,
            # Task type one-hot encoding
            float(self.task_type == 'replenishment'),
            float(self.task_type == 'returns'),
            float(self.task_type == 'ad_hoc'),
            float(self.task_type == 'emergency'),
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

    def get_queue_summary(self) -> np.ndarray:
        """
        Get summary statistics about the queue.

        Returns: [num_tasks, avg_priority, num_urgent, oldest_age, num_near_deadline]
        Total: 5 features
        """
        if not self.tasks:
            return np.zeros(5, dtype=np.float32)

        num_tasks = len(self.tasks)
        avg_priority = np.mean([t.manual_priority for t in self.tasks])
        num_urgent = sum(1 for t in self.tasks if t.manual_priority >= 4)

        # Oldest task age (requires current_time, so we use urgency_score as proxy)
        oldest_age = max([t.urgency_score for t in self.tasks], default=0.0)

        # Tasks near deadline (using urgency_score as indicator)
        num_near_deadline = sum(1 for t in self.tasks if t.urgency_score > 50)

        return np.array([
            float(num_tasks),
            avg_priority,
            float(num_urgent),
            oldest_age,
            float(num_near_deadline)
        ], dtype=np.float32)

    def clear(self):
        """Clear all tasks from queue."""
        self.tasks.clear()


def compute_urgency_score(task: Task, current_time: float) -> float:
    """
    Compute heuristic urgency score for a task.
    Higher score = more urgent.

    Args:
        task: Task to score
        current_time: Current simulation time

    Returns:
        Urgency score (0-100+)
    """
    # 1. Base score from manual priority
    base_score = task.manual_priority * 10  # 10-50

    # 2. Deadline pressure (exponential as deadline approaches)
    time_to_deadline = task.get_time_to_deadline(current_time)
    if time_to_deadline < 300:  # Less than 5 minutes
        deadline_multiplier = np.exp(-time_to_deadline / 100)  # 1.0 to ~20.0
    else:
        deadline_multiplier = 1.0

    # 3. Age bonus (waiting time penalty)
    age = task.get_age(current_time)
    age_bonus = min(age / 60, 5.0)  # +0 to +5 based on minutes waited

    # 4. Task type modifiers
    type_multiplier = {
        'emergency': 3.0,
        'replenishment': 1.0,
        'returns': 0.8,
        'ad_hoc': 1.2
    }.get(task.task_type, 1.0)

    # Combine all factors
    urgency = (base_score * deadline_multiplier + age_bonus) * type_multiplier

    return urgency


def rank_tasks(task_queue: TaskQueue, current_time: float) -> List[Task]:
    """
    Sort tasks by urgency score (descending).
    Updates each task's urgency_score and queue_position.

    Args:
        task_queue: Queue of tasks to rank
        current_time: Current simulation time

    Returns:
        List of tasks sorted by urgency (most urgent first)
    """
    # Compute urgency score for each task
    scored_tasks = []
    for task in task_queue.tasks:
        score = compute_urgency_score(task, current_time)
        task.urgency_score = score
        scored_tasks.append((task, score))

    # Sort by score (highest first)
    sorted_tasks = sorted(scored_tasks, key=lambda x: x[1], reverse=True)

    # Assign queue positions
    ranked_tasks = []
    for i, (task, score) in enumerate(sorted_tasks):
        task.queue_position = i
        ranked_tasks.append(task)

    return ranked_tasks


def create_random_tasks(
    num_tasks: int,
    num_nodes: int = 10,
    current_time: float = 0.0,
    base_deadline: float = 600.0
) -> TaskQueue:
    """
    Create a random set of tasks with urgency features.

    Args:
        num_tasks: Number of tasks to create
        num_nodes: Number of nodes in the graph
        current_time: Current simulation time
        base_deadline: Base deadline offset (seconds)

    Returns:
        TaskQueue with random tasks
    """
    task_queue = TaskQueue()

    for i in range(num_tasks):
        from_idx = np.random.randint(0, num_nodes)
        to_idx = np.random.randint(0, num_nodes)

        # Ensure from and to are different
        while to_idx == from_idx:
            to_idx = np.random.randint(0, num_nodes)

        # Random task type
        task_type = np.random.choice(
            ['replenishment', 'returns', 'ad_hoc', 'emergency'],
            p=[0.4, 0.3, 0.25, 0.05]  # Emergency is rare
        )

        # Random priority (1-5)
        manual_priority = np.random.randint(1, 6)

        # Random arrival time (0 to current_time)
        arrival_time = current_time + np.random.uniform(-300, 0)  # Arrived in last 5 min

        # Deadline based on priority and type
        if task_type == 'emergency':
            deadline_offset = np.random.uniform(60, 300)  # 1-5 minutes
        elif manual_priority >= 4:
            deadline_offset = np.random.uniform(300, 900)  # 5-15 minutes
        else:
            deadline_offset = np.random.uniform(900, 1800)  # 15-30 minutes

        deadline = current_time + deadline_offset

        # Estimated duration
        estimated_duration = np.random.uniform(30, 180)  # 30s to 3min

        task = Task(
            task_id=i,
            from_location_index=from_idx,
            to_location_index=to_idx,
            manual_priority=manual_priority,
            deadline=deadline,
            arrival_time=arrival_time,
            estimated_duration=estimated_duration,
            task_type=task_type
        )

        task_queue.add_task(task)

    return task_queue
