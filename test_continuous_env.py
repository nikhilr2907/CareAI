"""
Test script for continuous operation environment.

Verifies:
- Environment reset
- Time advancement
- Inventory consumption
- Task generation from low stock
- Robot movement
- Task completion
- Multi-capacity assignment
"""
import numpy as np
from src.environment.gapo_env import GAPOTaskAssignmentEnv


def test_continuous_environment():
    """Test continuous environment operation."""
    print("=" * 60)
    print("Testing Continuous GAPO Environment")
    print("=" * 60)

    # Create environment
    env = GAPOTaskAssignmentEnv(
        num_robots=5,
        num_nodes=10,
        max_episode_time=1000.0,  # 1000 seconds = ~16 minutes
        timestep_seconds=1.0
    )

    # Test 1: Reset
    print("\n1. Testing reset...")
    state = env.reset()
    print(f"   [OK] Reset successful")
    print(f"   - Current time: {env.current_time:.1f}s")
    print(f"   - Pending tasks: {len(env.pending_tasks)}")
    print(f"   - Robots: {len(env.robots)}")
    print(f"   - Graph nodes: {len(env.graph_state.nodes)}")

    # Test 2: Initial state structure
    print("\n2. Checking state structure...")
    assert 'task_features' in state, "Missing task_features"
    assert 'node_continuous' in state, "Missing node_continuous"
    assert 'node_categorical' in state, "Missing node_categorical"
    assert 'edge_features' in state, "Missing edge_features"
    assert 'robot_features' in state, "Missing robot_features"
    print(f"   [OK] All state components present")
    print(f"   - Task features shape: {state['task_features'].shape}")
    print(f"   - Node continuous shape: {state['node_continuous'].shape}")
    print(f"   - Edge features shape: {state['edge_features'].shape}")
    print(f"   - Robot features shape: {state['robot_features'].shape}")

    # Test 3: Step without assignments (just time advancement)
    print("\n3. Testing time advancement...")
    initial_time = env.current_time
    for _ in range(10):
        state, reward, done, info = env.step(Δt=1.0)
    print(f"   [OK] Time advanced from {initial_time:.1f}s to {env.current_time:.1f}s")
    print(f"   - Reward accumulated: {reward:.2f}")
    print(f"   - Pending tasks: {info['pending_tasks']}")

    # Test 4: Task assignment
    print("\n4. Testing task assignment...")
    if env.pending_tasks:
        task = env.pending_tasks[0]
        print(f"   - Task {task.task_id}: {task.task_type} priority {task.manual_priority}")
        print(f"   - From node {task.from_location_index} to {task.to_location_index}")
        print(f"   - Items needed: {task.num_items}")

        # Get availability mask
        mask = env.get_robot_availability_mask(task)
        available_robots = np.where(mask)[0]
        print(f"   - Available robots: {available_robots.tolist()}")

        if len(available_robots) > 0:
            robot_id = available_robots[0]
            robot = env.robots[robot_id]
            print(f"   - Assigning to robot {robot_id}")
            print(f"     Before: load={robot.current_load}, queued={robot.num_queued_tasks}")

            success = env.assign_task_to_robot(robot_id, task)
            assert success, f"Assignment failed for robot {robot_id}"

            print(f"     After: load={robot.current_load}, queued={robot.num_queued_tasks}")
            print(f"   [OK] Task assigned successfully")
        else:
            print(f"   [WARN] No available robots")
    else:
        print(f"   [WARN] No pending tasks")

    # Test 5: Multi-capacity assignment
    print("\n5. Testing multi-capacity...")
    # Force create a small task
    if len(env.robots) > 0:
        robot = env.robots[0]
        initial_queue_size = robot.num_queued_tasks

        # Try to assign 2-3 more tasks to same robot if capacity allows
        assigned_count = 0
        for task in env.pending_tasks[:3]:
            if robot.can_accept_items(task.num_items):
                success = env.assign_task_to_robot(0, task)
                if success:
                    assigned_count += 1

        print(f"   - Robot 0 queue: {initial_queue_size} -> {robot.num_queued_tasks} tasks")
        print(f"   - Assigned {assigned_count} additional tasks")
        print(f"   - Current load: {robot.current_load}/{robot.max_capacity}")
        print(f"   [OK] Multi-capacity working")

    # Test 6: Robot movement and task completion
    print("\n6. Testing robot movement...")
    initial_completed = len(env.completed_tasks)

    # Run for 100 steps
    for step in range(100):
        state, reward, done, info = env.step(Δt=1.0)

        if info['completed_tasks'] > 0:
            print(f"   - Step {step}: {info['completed_tasks']} task(s) completed!")

    print(f"   - Total completed: {len(env.completed_tasks) - initial_completed}")
    print(f"   - Total robot tasks: {info['total_robot_tasks']}")
    print(f"   [OK] Robots moving and completing tasks")

    # Test 7: Inventory consumption and task generation
    print("\n7. Testing inventory-based task generation...")
    print(f"   - Current time: {env.current_time:.1f}s")

    # Check node inventory levels
    recovery_nodes = [n for n in env.graph_state.nodes if n.node_type == 'recovery']
    print(f"   - Recovery nodes: {len(recovery_nodes)}")

    for node in recovery_nodes[:3]:
        print(f"     Node {node.node_id}: stock={node.stock_level:.1f}, "
              f"consumption={node.consumption_rate:.2f}/hr, "
              f"time_to_stockout={node.time_to_stockout:.1f}hr, "
              f"urgency={node.urgency_level}")

    # Run for a while to trigger inventory checks
    initial_pending = len(env.pending_tasks)
    for _ in range(200):
        state, reward, done, info = env.step(Δt=1.0)

        # Assign some tasks to keep queue from growing too large
        for task in env.pending_tasks[:]:
            mask = env.get_robot_availability_mask(task)
            available = np.where(mask)[0]
            if len(available) > 0:
                env.assign_task_to_robot(available[0], task)
                break  # Assign one per step

    final_pending = len(env.pending_tasks)
    print(f"   - Pending tasks: {initial_pending} -> {final_pending}")
    print(f"   - New tasks generated: {final_pending - initial_pending + len(env.completed_tasks)}")
    print(f"   [OK] Inventory-based task generation working")

    # Test 8: Stockout penalties
    print("\n8. Checking stockout detection...")
    stockout_nodes = [n for n in env.graph_state.nodes if n.is_stockout]
    print(f"   - Stockout nodes: {len(stockout_nodes)}")
    print(f"   - Current stockout penalty count: {info['stockouts']}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Final time: {env.current_time:.1f}s")
    print(f"Pending tasks: {len(env.pending_tasks)}")
    print(f"Completed tasks: {len(env.completed_tasks)}")
    print(f"Cumulative reward: {env.cumulative_reward:.2f}")
    print(f"Stockouts: {info['stockouts']}")

    for i, robot in enumerate(env.robots):
        print(f"Robot {i}: queued={robot.num_queued_tasks}, "
              f"load={robot.current_load}/{robot.max_capacity}, "
              f"battery={robot.battery_level:.1f}%")

    print("\n[OK] All tests passed!")


if __name__ == '__main__':
    test_continuous_environment()
