from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

# --- High-Level Action Space (Meta-Controller) ---

class HighLevelActionType(Enum):
    """
    Defines the strategic choices for a robot.
    """
    WAIT = auto()  # Do nothing, stay available.
    HOLD = auto()  # Wait due to a specific condition, e.g., congestion.
    ASSIGN_TASK = auto()  # Undertake a complex task.

@dataclass
class HighLevelAction:
    """
    Represents an action from the high-level actor.
    
    The 'decision' determines the overall strategy, and 'task_id' is
    only relevant if that strategy is to assign a task.
    """
    decision: HighLevelActionType
    task_id: Optional[int] = None

    def __post_init__(self):
        if self.decision == HighLevelActionType.ASSIGN_TASK and self.task_id is None:
            raise ValueError("ASSIGN_TASK decision requires a task_id.")
        if self.decision != HighLevelActionType.ASSIGN_TASK and self.task_id is not None:
            raise ValueError("task_id should only be provided for ASSIGN_TASK decision.")

# --- Low-Level Action Space (Controller) ---

class SubTaskType(Enum):
    """
    Defines the building blocks of a complex task.
    """
    NAVIGATE_TO = auto()
    PICKUP_ITEM = auto()
    DROPOFF_ITEM = auto()
    DOCK_AND_CHARGE = auto()

@dataclass
class SubTask:
    """
    Represents a single, concrete step a robot can take.
    The low-level actor's job is to choose the best one from a list
    of available sub-tasks to progress the overall high-level assignment.
    
    Example:
        - SubTask(type=SubTaskType.NAVIGATE_TO, target_node="WardA")
        - SubTask(type=SubTaskType.PICKUP_ITEM, item_id="bandages")
    """
    type: SubTaskType
    target_node: Optional[str] = None
    item_id: Optional[str] = None

@dataclass
class LowLevelAction:
    """
    Represents an action from the low-level actor.
    
    This action is the choice of which sub-task to perform next from a
    list of currently available sub-tasks for a given assignment.
    """
    chosen_sub_task: SubTask

# --- Example of How They Work Together ---

def example_usage():
    """
    Demonstrates the hierarchical action space concept.
    """
    
    # 1. High-level actor decides to assign a task to a robot.
    high_level_action = HighLevelAction(decision=HighLevelActionType.ASSIGN_TASK, task_id=101)
    print(f"High-Level Actor chose: {high_level_action}")

    # 2. Based on task_id=101, the system determines the list of sub-tasks required.
    # For a simple delivery, this might be:
    task_sub_tasks = [
        SubTask(type=SubTaskType.NAVIGATE_TO, target_node="Hub"),
        SubTask(type=SubTaskType.PICKUP_ITEM, item_id="medication_A"),
        SubTask(type=SubTaskType.NAVIGATE_TO, target_node="WardC"),
        SubTask(type=SubTaskType.DROPOFF_ITEM, item_id="medication_A")
    ]
    
    # At the start, let's say only the first navigation sub-task is available.
    available_sub_tasks = [task_sub_tasks[0]]
    
    # 3. Low-level actor receives the set of available sub-tasks and chooses one.
    # In this simple case, it has only one choice.
    chosen_sub_task = available_sub_tasks[0]
    low_level_action = LowLevelAction(chosen_sub_task=chosen_sub_task)
    print(f"Low-Level Actor chose: {low_level_action}")

    # After the robot completes this sub-task, the system would update the
    # list of available sub-tasks (e.g., PICKUP_ITEM would become available),
    # and the low-level actor would be invoked again to choose the next step.
    
if __name__ == "__main__":
    example_usage()
