from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Set
import numpy as np
from .robot_telemetry import RobotTelemetry

if TYPE_CHECKING:
    from ..tasks.task_state import Task


@dataclass
class RobotState:
    robot_id: int
    max_capacity: int = 12  # Maximum items robot can carry

    # ===== TASK ASSIGNMENTS (Environment-controlled) =====
    task_queue: List['Task'] = field(default_factory=list)  # Ordered list of Task objects
    overflow_queue: List['Task'] = field(default_factory=list)  # Deferred tasks when capacity is full
    picked_up_task_ids: Set[int] = field(default_factory=set)  # Parent task IDs with pickup completed
    target_node_index: Optional[int] = None  # Ultimate destination for current task
    planned_path: List[int] = field(default_factory=list)  # Full path for current task
    travel_start_time: float = 0.0  # When robot started current journey

    # ===== TELEMETRY (Robot simulator/hardware reports) =====
    telemetry: Optional[RobotTelemetry] = None  # Latest sensor data

    def update_telemetry(self, telemetry: RobotTelemetry):
        """Receive new telemetry update from robot."""
        self.telemetry = telemetry

    @property
    def num_queued_tasks(self) -> int:
        """Get number of original (parent) tasks queued, not counting pickup+dropoff legs separately."""
        seen = set()
        for task in self.task_queue:
            pid = task.parent_task_id if task.leg_type in ("pickup", "dropoff") else task.task_id
            seen.add(pid)
        return len(seen)

    @property
    def current_task(self) -> Optional['Task']:
        """Get the current task (first in queue)."""
        return self.task_queue[0] if self.task_queue else None

    @property
    def current_position(self) -> tuple:
        """Get latest reported position from telemetry."""
        if self.telemetry:
            return (self.telemetry.x, self.telemetry.y)
        return (0.0, 0.0)

    @property
    def current_node_index(self) -> Optional[int]:
        """Get current node if robot is at a node (from telemetry)."""
        if self.telemetry:
            return self.telemetry.current_node_index
        return None

    @property
    def battery_level(self) -> float:
        """Get current battery level from telemetry."""
        if self.telemetry:
            return self.telemetry.battery_level
        return 0.0

    @property
    def current_load(self) -> int:
        """Get current number of items physically loaded (from telemetry)."""
        if self.telemetry:
            return self.telemetry.current_capacity
        return 0

    @property
    def effective_load(self) -> int:
        """Physical load plus items committed by queued pickup legs not yet executed."""
        committed = sum(
            t.num_items for t in self.task_queue
            if getattr(t, "leg_type", "full") == "pickup"
        )
        return self.current_load + committed

    @property
    def available_capacity(self) -> int:
        """How many more items can be loaded, accounting for committed pickups."""
        return max(0, self.max_capacity - self.effective_load)

    @property
    def is_available(self) -> bool:
        """Available if no queued tasks AND telemetry confirms idle."""
        if self.telemetry:
            return len(self.task_queue) == 0 and self.telemetry.is_available
        return len(self.task_queue) == 0

    @property
    def can_accept_task(self) -> bool:
        """Robot can accept tasks even when busy if it has capacity and not charging."""
        return self.available_capacity > 0 and not self.is_charging

    @property
    def is_charging(self) -> bool:
        """Check if robot is currently charging."""
        if self.telemetry:
            return self.telemetry.is_charging
        return False

    def add_task(self, task: 'Task'):
        """
        Add task to robot's queue, maintaining priority order (high priority first).

        Args:
            task: Task to add to queue
        """
        if self.task_queue:
            current_task = self.task_queue[0]
            remaining = self.task_queue[1:] + [task]
        else:
            current_task = None
            remaining = [task]

        # Sort by priority (high to low), then pickup before dropoff, then urgency
        def leg_order(t):
            return 0 if getattr(t, "leg_type", "full") in ("pickup", "full") else 1

        remaining.sort(key=lambda t: (leg_order(t), -t.learned_score, -t.manual_priority, -t.urgency_score))
        if current_task:
            self.task_queue = [current_task] + remaining
        else:
            self.task_queue = remaining

    def defer_pickups_if_full(self):
        """Move pickup tasks to overflow when at capacity (keeps current task)."""
        if self.current_load >= self.max_capacity:
            if not self.task_queue:
                return
            current_task = self.task_queue[0]
            remaining = self.task_queue[1:]
            keep = []
            deferred_parent_ids = set()
            for task in remaining:
                if getattr(task, "leg_type", "full") == "pickup":
                    self.overflow_queue.append(task)
                    if task.parent_task_id is not None:
                        deferred_parent_ids.add(task.parent_task_id)
                else:
                    keep.append(task)
            if deferred_parent_ids:
                still_keep = []
                for task in keep:
                    if getattr(task, "leg_type", "full") == "dropoff" and task.parent_task_id in deferred_parent_ids:
                        self.overflow_queue.append(task)
                    else:
                        still_keep.append(task)
                keep = still_keep
            self.task_queue = [current_task] + keep

    def promote_from_overflow(self):
        """Promote top-scored overflow tasks into main queue based on capacity."""
        if self.current_load >= self.max_capacity or not self.overflow_queue:
            return

        if self.task_queue:
            current_task = self.task_queue[0]
            remaining = self.task_queue[1:]
        else:
            current_task = None
            remaining = []

        # Sort overflow by learned score/priority
        def leg_order(t):
            return 0 if getattr(t, "leg_type", "full") in ("pickup", "full") else 1
        self.overflow_queue.sort(key=lambda t: (leg_order(t), -t.learned_score, -t.manual_priority, -t.urgency_score))

        available_slots = max(0, self.max_capacity - self.current_load)
        promoted = []
        deferred = []
        # Seed pickup_count from ALL pickups already committed in the main queue
        # (current task + remaining), not just the current task, to avoid over-promotion.
        pickup_count = 0
        pickup_ready_parents = set(self.picked_up_task_ids)
        for task in self.task_queue:
            if getattr(task, "leg_type", "full") == "pickup":
                pickup_count += task.num_items
                if task.parent_task_id is not None:
                    pickup_ready_parents.add(task.parent_task_id)

        for task in self.overflow_queue:
            if getattr(task, "leg_type", "full") == "pickup":
                if pickup_count + task.num_items <= available_slots:
                    promoted.append(task)
                    pickup_count += task.num_items  # count items, not tasks
                    if task.parent_task_id is not None:
                        pickup_ready_parents.add(task.parent_task_id)
                else:
                    deferred.append(task)
            else:
                if task.parent_task_id is None or task.parent_task_id in pickup_ready_parents:
                    promoted.append(task)
                else:
                    deferred.append(task)

        remaining = remaining + promoted
        remaining.sort(key=lambda t: (leg_order(t), -t.learned_score, -t.manual_priority, -t.urgency_score))

        self.overflow_queue = deferred
        if current_task:
            self.task_queue = [current_task] + remaining
        else:
            self.task_queue = remaining

    def defer_current_task(self):
        """Defer the current task to overflow (used when prerequisites are unmet)."""
        if not self.task_queue:
            return
        current_task = self.task_queue.pop(0)
        self.overflow_queue.append(current_task)

    def demote_current_task(self):
        """Move current task to the end of the main queue."""
        if len(self.task_queue) <= 1:
            return
        current_task = self.task_queue.pop(0)
        self.task_queue.append(current_task)

    def resort_queue(self):
        """Re-sort queued tasks (keeps current task fixed)."""
        if not self.task_queue:
            return
        current_task = self.task_queue[0]
        remaining = self.task_queue[1:]
        def leg_order(t):
            return 0 if getattr(t, "leg_type", "full") in ("pickup", "full") else 1
        remaining.sort(key=lambda t: (leg_order(t), -t.learned_score, -t.manual_priority, -t.urgency_score))
        self.task_queue = [current_task] + remaining

    def resort_overflow(self):
        """Re-sort overflow queue."""
        def leg_order(t):
            return 0 if getattr(t, "leg_type", "full") in ("pickup", "full") else 1
        self.overflow_queue.sort(key=lambda t: (leg_order(t), -t.learned_score, -t.manual_priority, -t.urgency_score))

    def enforce_capacity_limits(self):
        """
        Ensure pickup legs in main queue do not exceed available capacity.
        Moves excess pickups (and their matching dropoffs) to overflow.
        """
        available_slots = max(0, self.max_capacity - self.current_load)
        if not self.task_queue:
            return
        current_task = self.task_queue[0]
        remaining = self.task_queue[1:]

        kept = []
        deferred_parent_ids = set()
        pickup_count = 0

        # Count current pickup if applicable (in items, not task count)
        if getattr(current_task, "leg_type", "full") == "pickup":
            pickup_count = current_task.num_items

        for task in remaining:
            if getattr(task, "leg_type", "full") == "pickup":
                if pickup_count + task.num_items <= available_slots:
                    kept.append(task)
                    pickup_count += task.num_items
                else:
                    self.overflow_queue.append(task)
                    if task.parent_task_id is not None:
                        deferred_parent_ids.add(task.parent_task_id)
            else:
                kept.append(task)

        if deferred_parent_ids:
            still_keep = []
            for task in kept:
                if getattr(task, "leg_type", "full") == "dropoff" and task.parent_task_id in deferred_parent_ids:
                    self.overflow_queue.append(task)
                else:
                    still_keep.append(task)
            kept = still_keep

        self.task_queue = [current_task] + kept

    def mark_pickup_complete(self, parent_task_id: Optional[int]):
        """Record that a pickup leg for a parent task is complete."""
        if parent_task_id is not None:
            self.picked_up_task_ids.add(parent_task_id)

    def mark_dropoff_complete(self, parent_task_id: Optional[int]):
        """Record that a dropoff leg for a parent task is complete."""
        if parent_task_id is not None and parent_task_id in self.picked_up_task_ids:
            self.picked_up_task_ids.remove(parent_task_id)

    def is_pickup_complete(self, parent_task_id: Optional[int]) -> bool:
        """Check if pickup for parent task has completed."""
        if parent_task_id is None:
            return True
        return parent_task_id in self.picked_up_task_ids

    def complete_current_task(self) -> Optional['Task']:
        """
        Mark current task complete and remove from queue.

        Returns:
            Completed task, or None if no tasks in queue
        """
        if self.task_queue:
            completed = self.task_queue.pop(0)
            # Reset path if no more tasks
            if not self.task_queue:
                self.planned_path.clear()
                self.target_node_index = None
            return completed
        return None

    def get_total_queued_items(self) -> int:
        """Get total number of items across all queued tasks."""
        return sum(task.num_items for task in self.task_queue)

    def can_accept_items(self, num_items: int) -> bool:
        """Check if robot can accept additional items."""
        total_items = self.current_load + self.get_total_queued_items()
        return (total_items + num_items) <= self.max_capacity

    def compute_distances_to_nodes(self, graph_state) -> np.ndarray:
        """
        Compute Euclidean distances from robot's current position to all nodes.

        Args:
            graph_state: GraphState object containing nodes

        Returns:
            numpy array of distances to all nodes
        """
        distances = []
        robot_x, robot_y = self.current_position

        for node in graph_state.nodes:
            distance = np.sqrt((node.center_x - robot_x)**2 + (node.center_y - robot_y)**2)
            distances.append(distance)

        return np.array(distances, dtype=np.float32)


def create_default_robot(robot_id: int) -> RobotState:
    """
    Create a robot with default state.
    Telemetry will be populated by robot simulator.
    """
    return RobotState(
        robot_id=robot_id,
        max_capacity=12
    )
