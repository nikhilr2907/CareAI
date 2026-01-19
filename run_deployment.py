"""
Deployment Module - Free-running task assignment system.

This module runs continuously in a hospital, receiving tasks from hospital
systems and assigning them to real robots using a trained RL policy.

For now, uses mock bridges (simulator) but designed to swap in real
robot connections (ROS/MQTT) and real task sources (API/Database).
"""
import time
import numpy as np
from pathlib import Path

from src.environment.gapo_env import GAPOTaskAssignmentEnv
from src.environment.graph.hospital_config import HospitalConfig
from src.deployment.robot_bridge import MockRobotBridge
from src.deployment.task_source import MockTaskSource
from src.environment.tasks.task_state import Task


def run_deployment():
    """
    Main deployment loop.

    Runs continuously, monitoring for new tasks and assigning them to robots.
    """
    print("=" * 80)
    print("HOSPITAL ROBOT DEPLOYMENT - Free-Running Mode")
    print("=" * 80)

    # ========== CONFIGURATION ==========
    # Load hospital layout
    hospital_config = None  # Uses default 10-node layout
    # hospital_config = HospitalConfig.from_file('configs/hospital_layout.json')

    num_robots = 5
    num_nodes = 10

    # Policy configuration
    use_trained_policy = False  # Set True when you have trained weights
    policy_checkpoint = "checkpoints/gapo/gapo_final.pth"

    # Deployment settings
    cycle_time = 1.0  # Seconds between assignment cycles
    max_runtime = 3600.0  # Run for 1 hour (simulation time)

    print(f"\nConfiguration:")
    print(f"  Hospital Config: {'default' if hospital_config is None else hospital_config.name}")
    print(f"  Robots: {num_robots}")
    print(f"  Nodes: {num_nodes}")
    print(f"  Policy: {'Trained' if use_trained_policy else 'Heuristic (greedy)'}")
    print(f"  Cycle Time: {cycle_time}s")
    print("=" * 80)

    # ========== INITIALIZE SYSTEM ==========
    print("\nInitializing system...")

    # Create environment (for state representation and graph)
    env = GAPOTaskAssignmentEnv(
        num_robots=num_robots,
        num_nodes=num_nodes,
        max_episode_time=max_runtime,
        timestep_seconds=cycle_time,
        hospital_config=hospital_config
    )
    env.reset()

    # Create robot bridge (connects to real or mock robots)
    robot_bridge = MockRobotBridge(env.robot_simulators)
    print(f"  [OK] Robot bridge initialized ({robot_bridge.get_num_robots()} robots)")

    # Create task source (receives tasks from hospital systems)
    task_source = MockTaskSource(env.graph_state, next_task_id=0)
    print(f"  [OK] Task source initialized (mock inventory-based)")

    # Load policy (or use heuristic)
    if use_trained_policy:
        from src.multi_agent_ppo.gapo_ppo import GAPOPPO
        ppo = GAPOPPO.load(policy_checkpoint)
        print(f"  [OK] Loaded trained policy from {policy_checkpoint}")
    else:
        ppo = None
        print(f"  [OK] Using heuristic policy (assign to nearest available robot)")

    # ========== DEPLOYMENT LOOP ==========
    print("\n" + "=" * 80)
    print("Starting deployment loop...")
    print("=" * 80 + "\n")

    current_time = 0.0
    cycle_count = 0
    total_tasks_assigned = 0
    total_tasks_completed = 0

    try:
        while current_time < max_runtime:
            cycle_start_time = time.time()

            # ===== 1. GET NEW TASKS FROM HOSPITAL SYSTEMS =====
            incoming_tasks = task_source.get_new_tasks(current_time)

            if incoming_tasks:
                print(f"[{current_time:.1f}s] Received {len(incoming_tasks)} new task(s)")

                # Convert incoming tasks to environment Task objects
                for incoming_task in incoming_tasks:
                    task = Task(
                        task_id=incoming_task.task_id,
                        task_type=incoming_task.task_type,
                        from_location_index=incoming_task.from_location,
                        to_location_index=incoming_task.to_location,
                        num_items=incoming_task.num_items,
                        manual_priority=incoming_task.priority,
                        arrival_time=incoming_task.created_at,
                        deadline=incoming_task.deadline if incoming_task.deadline else current_time + 3600,
                        estimated_duration=30.0  # Default 30s estimate
                    )
                    env.pending_tasks.append(task)

            # ===== 2. ASSIGN PENDING TASKS =====
            if env.pending_tasks:
                # Sort by priority and age
                env.pending_tasks.sort(key=lambda t: (-t.manual_priority, t.arrival_time))

                task = env.pending_tasks[0]

                # Get robot availability (compute manually)
                action_mask = get_robot_availability_for_task(task, env.robots, env.num_robots)

                # Select robot using policy or heuristic
                if ppo is not None:
                    # Use trained RL policy
                    state_dict = env._get_state_dict()
                    action = ppo.select_action_greedy(state_dict, action_mask)
                else:
                    # Use heuristic: assign to nearest available robot
                    action = select_nearest_robot(task, env.robots, env.graph_state, action_mask)

                # Assign task
                robot_id = action
                success = env.assign_task_to_robot(robot_id, task)

                if success:
                    total_tasks_assigned += 1
                    print(f"[{current_time:.1f}s] Task {task.task_id} assigned to Robot {robot_id}")
                    print(f"  - Type: {task.task_type}, Priority: {task.manual_priority}")
                    print(f"  - Route: Node {task.from_location_index} -> {task.to_location_index}")
                    print(f"  - Items: {task.num_items}")

            # ===== 3. UPDATE ROBOT POSITIONS (SIMULATE MOVEMENT) =====
            # In real deployment, robots move on their own - this simulates that
            robot_bridge.update_simulators(cycle_time)

            # Sync telemetry back to environment
            for robot, simulator in zip(env.robots, env.robot_simulators):
                telemetry = simulator.get_telemetry()
                telemetry.timestamp = current_time
                robot.update_telemetry(telemetry)

            # ===== 4. CHECK TASK COMPLETIONS =====
            completed_tasks = env._check_task_completions()

            if completed_tasks:
                total_tasks_completed += len(completed_tasks)
                for task in completed_tasks:
                    print(f"[{current_time:.1f}s] Task {task.task_id} COMPLETED by Robot {task.assigned_robot_id}")

            # ===== 5. UPDATE INVENTORY =====
            from src.environment.tasks.task_generator import update_inventory_levels
            time_delta_hours = cycle_time / 3600.0
            update_inventory_levels(env.graph_state, time_delta_hours)

            # ===== 6. LOG STATUS =====
            if cycle_count % 60 == 0:  # Every 60 cycles (60 seconds)
                print(f"\n--- Status at {current_time:.1f}s ---")
                print(f"  Pending tasks: {len(env.pending_tasks)}")
                print(f"  Assigned tasks: {total_tasks_assigned}")
                print(f"  Completed tasks: {total_tasks_completed}")

                for i, robot in enumerate(env.robots):
                    print(f"  Robot {i}: queued={robot.num_queued_tasks}, "
                          f"load={robot.current_load}/{robot.max_capacity}, "
                          f"battery={robot.battery_level:.1f}%")

            # ===== 7. ADVANCE TIME =====
            current_time += cycle_time
            cycle_count += 1

            # Real-time sync (for visualization)
            elapsed = time.time() - cycle_start_time
            sleep_time = max(0, cycle_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Shutting down gracefully...")

    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("Deployment Summary")
    print("=" * 80)
    print(f"Runtime: {current_time:.1f}s ({current_time/3600:.2f} hours)")
    print(f"Total cycles: {cycle_count}")
    print(f"Tasks assigned: {total_tasks_assigned}")
    print(f"Tasks completed: {total_tasks_completed}")
    print(f"Tasks pending: {len(env.pending_tasks)}")

    print("\nFinal robot status:")
    for i, robot in enumerate(env.robots):
        print(f"  Robot {i}: queued={robot.num_queued_tasks}, "
              f"load={robot.current_load}/{robot.max_capacity}, "
              f"battery={robot.battery_level:.1f}%")

    print("\n" + "=" * 80)


def get_robot_availability_for_task(task, robots, num_robots):
    """
    Compute which robots can accept the task.

    Args:
        task: Task to check
        robots: List of RobotState objects
        num_robots: Total number of robots

    Returns:
        Boolean mask [num_robots]
    """
    return np.ones(num_robots, dtype=bool)


def select_nearest_robot(task, robots, graph_state, action_mask):
    """
    Heuristic policy: assign task to nearest available robot.

    Args:
        task: Task to assign
        robots: List of RobotState objects
        graph_state: Graph for distance calculation
        action_mask: Binary mask of available actions

    Returns:
        robot_id
    """
    # Get available robots
    available_robots = [i for i in range(len(robots)) if action_mask[i] == 1]

    if not available_robots:
        available_robots = list(range(len(robots)))

    # Find nearest robot to task start location
    task_node = graph_state.nodes[task.from_location_index]
    min_distance = float('inf')
    best_robot = None

    for robot_id in available_robots:
        robot = robots[robot_id]
        robot_x, robot_y = robot.current_position

        distance = np.sqrt(
            (task_node.center_x - robot_x)**2 +
            (task_node.center_y - robot_y)**2
        )

        if distance < min_distance:
            min_distance = distance
            best_robot = robot_id

    return best_robot if best_robot is not None else 0


if __name__ == '__main__':
    run_deployment()
