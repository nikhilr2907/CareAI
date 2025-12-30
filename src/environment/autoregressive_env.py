"""
Autoregressive environment for hospital robot task assignment.
Processes tasks one-by-one with state updates after each assignment.
"""
import gym
from gym import spaces
import numpy as np
from typing import List, Tuple, Dict

from .graph.graph_state import GraphState
from .robot.robot_state import RobotState, create_default_robot
from .tasks.task_state import Task, TaskQueue, create_random_tasks, rank_tasks
from .autoregressive_state_builder import build_autoregressive_state, get_state_dim


class AutoregressiveTaskAssignmentEnv(gym.Env):
    """
    Environment for autoregressive task assignment.

    At each timestep, the model assigns ONE task to a robot or defers it (HOLD).
    State is updated after each assignment before the next decision.
    """

    def __init__(self, num_robots=5, num_nodes=10):
        super(AutoregressiveTaskAssignmentEnv, self).__init__()

        self.num_robots = num_robots
        self.num_nodes = num_nodes

        # Action space: [robot_0, robot_1, ..., robot_N, HOLD]
        self.action_space = spaces.Discrete(num_robots + 1)
        self.HOLD_ACTION = num_robots  # Last action index

        # Observation space
        state_dim = get_state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Initialize components
        self.graph_state = GraphState()
        self.robots = [create_default_robot(i, num_nodes) for i in range(num_robots)]
        self.task_queue = TaskQueue()

        # Episode tracking
        self.current_time = 0.0
        self.episode_assignments = []
        self.episode_deferred = []

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial state (first task to assign)
        """
        # Reset graph
        self.graph_state = GraphState()

        # Reset robots
        self.robots = [create_default_robot(i, self.num_nodes) for i in range(self.num_robots)]

        # Generate random tasks
        num_tasks = np.random.randint(3, 11)  # 3-10 tasks
        self.task_queue = create_random_tasks(
            num_tasks=num_tasks,
            num_nodes=self.num_nodes,
            current_time=self.current_time
        )

        # Rank tasks by urgency
        self.ranked_tasks = rank_tasks(self.task_queue, self.current_time)

        # Reset episode tracking
        self.current_task_index = 0
        self.episode_assignments = []
        self.episode_deferred = []

        # Build state for first task
        if self.ranked_tasks:
            current_task = self.ranked_tasks[0]
            remaining_tasks = self.ranked_tasks[1:]
            state = build_autoregressive_state(
                current_task, self.robots, self.graph_state,
                remaining_tasks, self.current_time
            )
        else:
            state = np.zeros(get_state_dim(), dtype=np.float32)

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one assignment decision.

        Args:
            action: Robot index (0 to num_robots-1) or HOLD (num_robots)

        Returns:
            next_state, reward, done, info
        """
        if self.current_task_index >= len(self.ranked_tasks):
            # No more tasks
            return np.zeros(get_state_dim(), dtype=np.float32), 0.0, True, {}

        current_task = self.ranked_tasks[self.current_task_index]

        # Process action
        reward = 0.0

        if action == self.HOLD_ACTION:
            # Defer this task
            self.episode_deferred.append(current_task)
            reward = self._compute_hold_reward(current_task)

        else:
            # Assign task to robot
            robot = self.robots[action]
            self.episode_assignments.append((current_task, robot))

            # Update robot state
            self._assign_task_to_robot(robot, current_task)

            # Compute reward
            reward = self._compute_assignment_reward(current_task, robot)

        # Move to next task
        self.current_task_index += 1

        # Check if done
        done = (self.current_task_index >= len(self.ranked_tasks))

        # Build next state
        if not done:
            next_task = self.ranked_tasks[self.current_task_index]
            remaining_tasks = self.ranked_tasks[self.current_task_index + 1:]
            next_state = build_autoregressive_state(
                next_task, self.robots, self.graph_state,
                remaining_tasks, self.current_time
            )
        else:
            next_state = np.zeros(get_state_dim(), dtype=np.float32)

        # Info
        info = {
            'num_assigned': len(self.episode_assignments),
            'num_deferred': len(self.episode_deferred),
            'current_time': self.current_time
        }

        return next_state, reward, done, info

    def _assign_task_to_robot(self, robot: RobotState, task: Task):
        """
        Update robot state when task is assigned.

        Args:
            robot: Robot to assign task to
            task: Task being assigned
        """
        from .graph_helpers import estimate_travel_time

        # Mark task as assigned
        task.is_assigned = True
        task.assigned_robot_id = robot.robot_id

        # Add to robot's queue
        robot.queued_tasks.append(task.task_id)

        # Update robot availability
        if robot.is_available and len(robot.queued_tasks) == 1:
            # Robot was idle, now working on this task
            travel_time = estimate_travel_time(robot, task, self.graph_state)
            completion_time = self.current_time + travel_time + task.estimated_duration
            robot.current_task_completion_time = completion_time
            robot.is_available = False
        # If robot is busy, task goes into queue (completion time not updated yet)

        # Update graph congestion (simplified - mark edges as having planned robot)
        # In full implementation, would mark specific path edges

    def _compute_assignment_reward(self, task: Task, robot: RobotState) -> float:
        """
        Compute reward for assigning task to robot.

        Args:
            task: Task that was assigned
            robot: Robot task was assigned to

        Returns:
            Reward value
        """
        from .graph_helpers import estimate_travel_time, estimate_path_congestion

        reward = 0.0

        # 1. Base reward for assignment
        reward += 5.0

        # 2. Travel efficiency (closer robot is better)
        travel_time = estimate_travel_time(robot, task, self.graph_state)
        max_travel_time = 300  # 5 minutes max
        travel_efficiency = 1.0 - min(travel_time / max_travel_time, 1.0)
        reward += 5.0 * travel_efficiency

        # 3. Battery check
        if robot.battery_level < 0.2:  # Low battery
            reward -= 3.0  # Penalize assigning to low-battery robot

        # 4. Load balancing
        fleet_avg_load = np.mean([r.num_queued_tasks for r in self.robots])
        load_diff = abs(robot.num_queued_tasks - fleet_avg_load)
        reward -= 0.5 * load_diff

        # 5. Congestion penalty
        path_congestion = estimate_path_congestion(robot, task, self.graph_state)
        reward -= 2.0 * path_congestion

        # 6. Urgency bonus (high urgency tasks assigned = good)
        if task.queue_position == 0:  # Most urgent task
            reward += 3.0
        elif task.queue_position < 3:  # Top 3 urgent
            reward += 1.0

        return reward

    def _compute_hold_reward(self, task: Task) -> float:
        """
        Compute reward for deferring a task (HOLD).

        Args:
            task: Task that was deferred

        Returns:
            Reward value
        """
        reward = 0.0

        # 1. Base penalty for deferring
        reward -= 1.0

        # 2. Penalty increases with task urgency
        urgency_factor = task.urgency_score / 100.0  # Normalize
        reward -= 5.0 * urgency_factor

        # 3. If all robots are busy, HOLD is acceptable
        all_busy = all(not r.is_available for r in self.robots)
        if all_busy:
            reward += 2.0  # Reduce penalty

        # 4. If deferring low-priority task while urgent tasks remain, that's okay
        if task.queue_position > 5:  # Low priority
            reward += 0.5

        return reward

    def render(self, mode='human'):
        """Optional: Visualization."""
        pass

    def close(self):
        """Clean up."""
        pass
