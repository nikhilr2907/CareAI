"""
GAPO Training Script with De-biasing.

Trains hospital robot task allocation using:
- Graph Attention-Based Policy Optimization (GAPO)
- Graph Neural Networks for hospital + robot encoding
- Cross-attention for task-robot-node
- De-biasing mechanisms for autoregressive decisions
- Continuous time simulation with telemetry
- Curriculum learning with multiple hospital configs

Usage:
    python run_training.py --curriculum adaptive --num-random 5 --iterations 10000
    python run_training.py --no-curriculum --config configs/my_hospital.json
"""
import torch
import numpy as np
from pathlib import Path
import time
import argparse

from src.environment.gapo_env import GAPOTaskAssignmentEnv
from src.environment.graph.hospital_config import HospitalConfig
from src.multi_agent_ppo.gapo_ppo import GAPOPPO, Memory


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train GAPO policy for hospital robot task assignment')

    # Curriculum settings
    parser.add_argument('--curriculum', type=str, default='adaptive', choices=['adaptive', 'fixed', 'random', 'none'],
                        help='Curriculum learning schedule (default: adaptive)')
    parser.add_argument('--no-curriculum', action='store_true',
                        help='Disable curriculum learning (train on single config)')
    parser.add_argument('--num-random', type=int, default=0,
                        help='Number of random configs to add to curriculum (default: 0)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom hospital config JSON (for single-config training)')

    # Training hyperparameters
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of training iterations (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension for GNN (default: 64)')
    parser.add_argument('--no-gnn', action='store_true',
                        help='Disable GNN encoders (use MLP fallback)')
    parser.add_argument('--no-debiasing', action='store_true',
                        help='Disable de-biasing loss')

    # Curriculum schedule parameters
    parser.add_argument('--config-interval', type=int, default=500,
                        help='Config switch interval for fixed schedule (default: 500)')
    parser.add_argument('--perf-threshold', type=float, default=50.0,
                        help='Performance threshold for adaptive schedule (default: 50.0)')

    # System
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def generate_curriculum_configs(num_random=0, include_default=True):
    """
    Generate a curriculum of hospital configurations with increasing complexity.

    Args:
        num_random: Number of additional random configs to generate
        include_default: Whether to include predefined configs

    Returns:
        List of (config, description) tuples
    """
    configs = []

    if include_default:
        # Stage 1: Simple layouts (10 nodes, 5 robots)
        configs.extend([
            (None, "Default 10-node hospital"),  # Default config
            (HospitalConfig(HospitalConfig.generate_random_grid(
                rows=2, cols=3, spacing=10.0, storage_ratio=0.3, recovery_ratio=0.5
            )), "Predefined: Small 2x3 grid"),
            (HospitalConfig(HospitalConfig.generate_random_grid(
                rows=3, cols=2, spacing=12.0, storage_ratio=0.2, recovery_ratio=0.6
            )), "Predefined: Compact 3x2 grid"),
        ])

        # Stage 2: Medium layouts (12-15 nodes)
        configs.extend([
            (HospitalConfig(HospitalConfig.generate_random_grid(
                rows=3, cols=4, spacing=10.0, storage_ratio=0.25, recovery_ratio=0.5
            )), "Predefined: Medium 3x4 grid"),
            (HospitalConfig(HospitalConfig.generate_random_grid(
                rows=4, cols=3, spacing=11.0, storage_ratio=0.3, recovery_ratio=0.4
            )), "Predefined: Medium 4x3 grid"),
        ])

        # Stage 3: Complex layouts (16-20 nodes)
        configs.extend([
            (HospitalConfig(HospitalConfig.generate_random_grid(
                rows=4, cols=4, spacing=10.0, storage_ratio=0.2, recovery_ratio=0.6
            )), "Predefined: Large 4x4 grid"),
            (HospitalConfig(HospitalConfig.generate_random_grid(
                rows=5, cols=3, spacing=12.0, storage_ratio=0.25, recovery_ratio=0.5
            )), "Predefined: Large 5x3 grid"),
        ])

    # Add random configs
    for i in range(num_random):
        # Randomly sample complexity
        rows = np.random.randint(2, 6)
        cols = np.random.randint(2, 6)
        spacing = np.random.uniform(8.0, 15.0)
        storage_ratio = np.random.uniform(0.15, 0.35)
        recovery_ratio = np.random.uniform(0.4, 0.7)

        config = HospitalConfig(HospitalConfig.generate_random_grid(
            rows=rows, cols=cols, spacing=spacing,
            storage_ratio=storage_ratio, recovery_ratio=recovery_ratio
        ))

        configs.append((config, f"Random {i+1}: {rows}x{cols} grid"))

    return configs


def main():
    # Parse command-line arguments
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    ############## Hyperparameters (from args) ##############
    num_robots = 5
    num_nodes = 10  # Will be overridden by config
    max_episode_time = 28800.0  # 8 hours
    timestep_seconds = 1.0

    # Multi-configuration training settings
    use_curriculum = not args.no_curriculum and args.curriculum != 'none'
    curriculum_schedule = args.curriculum if args.curriculum != 'none' else 'adaptive'
    config_switch_interval = args.config_interval
    performance_threshold = args.perf_threshold

    # Generate curriculum configs (mix of predefined + random)
    if use_curriculum:
        curriculum_configs = generate_curriculum_configs(
            num_random=args.num_random,
            include_default=True  # Always include predefined configs
        )
    elif args.config:
        # Single custom config from file
        custom_config = HospitalConfig.from_file(args.config)
        curriculum_configs = [(custom_config, f"Custom: {args.config}")]
    else:
        # Single default config
        curriculum_configs = [(None, "Default 10-node hospital")]

    current_config_idx = 0  # Start with simplest config

    # GAPO parameters
    hidden_dim = args.hidden_dim
    num_attention_heads = 4
    use_gnn = not args.no_gnn
    use_debiasing = not args.no_debiasing

    # PPO parameters
    lr = args.lr
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    lambda_debias = 0.1
    lambda_gae = 0.95

    # Training parameters
    max_training_iterations = args.iterations
    rollout_steps = 2000  # Collect experience for 2000 timesteps before update
    timesteps_per_decision = 60.0  # Make assignment decision every 60 seconds
    save_interval = 100  # Save every 100 iterations
    log_interval = 10  # Log every 10 iterations

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/gapo")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GAPO TRAINING - Graph Attention + De-biasing")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"GNN Enabled: {use_gnn}")
    print(f"De-biasing Enabled: {use_debiasing}")
    print(f"Hidden Dimension: {hidden_dim}")
    print(f"Attention Heads: {num_attention_heads}")
    print(f"Learning Rate: {lr}")
    print(f"De-bias Lambda: {lambda_debias}")

    if use_curriculum:
        print(f"Curriculum Learning: Enabled ({curriculum_schedule} schedule)")
        print(f"  Stages: {len(curriculum_configs)} configurations")
        if curriculum_schedule == 'fixed':
            print(f"  Switch interval: {config_switch_interval} iterations")
        elif curriculum_schedule == 'adaptive':
            print(f"  Performance threshold: {performance_threshold} avg reward")
    else:
        print("Curriculum Learning: Disabled (single config)")

    print("=" * 80)

    # Helper function to create/reset environment with config
    def create_env_with_config(config_idx):
        """Create environment with specified curriculum config."""
        if use_curriculum:
            hospital_config, config_desc = curriculum_configs[config_idx]
            print(f"\n{'='*80}")
            print(f"Loading Config {config_idx + 1}/{len(curriculum_configs)}: {config_desc}")
            print(f"{'='*80}")
        else:
            hospital_config = None
            config_desc = "Default 10-node hospital"

        # Determine num_nodes from config
        if hospital_config is not None and hasattr(hospital_config, 'nodes'):
            config_num_nodes = len(hospital_config.nodes)
        else:
            config_num_nodes = num_nodes

        env = GAPOTaskAssignmentEnv(
            num_robots=num_robots,
            num_nodes=config_num_nodes,
            max_episode_time=max_episode_time,
            timestep_seconds=timestep_seconds,
            hospital_config=hospital_config
        )

        return env, config_desc

    # Create initial environment
    env, current_config_desc = create_env_with_config(current_config_idx)

    # Create GAPO PPO
    ppo = GAPOPPO(
        node_continuous_dim=15,
        num_node_types=4,
        edge_feat_dim=12,
        robot_feat_dim=12,
        task_feat_dim=12,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        lr=lr,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        use_gnn=use_gnn,
        use_debiasing=use_debiasing,
        lambda_debias=lambda_debias,
        lambda_gae=lambda_gae,
        device=device
    )

    # Training metrics
    memory = Memory()
    running_reward = 0
    iteration_rewards = []
    config_rewards = []  # Track rewards per config for curriculum
    total_timesteps = 0
    iterations_on_current_config = 0

    print("\nStarting Continuous Training...")
    print("-" * 80)

    start_time = time.time()

    # Initialize environment (continuous operation, no resets during training!)
    state_dict = env.reset()

    for iteration in range(1, max_training_iterations + 1):
        iterations_on_current_config += 1
        iteration_reward = 0.0
        num_assignments = 0
        num_holds = 0
        iteration_start_time = time.time()

        # Collect rollout_steps timesteps of experience
        for step in range(rollout_steps):
            total_timesteps += 1

            # ===== AUTOREGRESSIVE TASK ASSIGNMENT PHASE =====
            # Assign ALL pending tasks before advancing time
            # Each assignment updates state for next decision
            while len(env.pending_tasks) > 0:
                # Get current task to assign
                task = env.pending_tasks[0]

                # Build action mask (which robots can handle this task)
                action_mask = np.ones(env.num_robots + 1, dtype=bool)
                for i, robot in enumerate(env.robots):
                    if not robot.can_accept_items(task.num_items):
                        action_mask[i] = False

                # Select action with CURRENT state
                action = ppo.select_action(state_dict, memory, action_mask)

                # Handle HOLD action (defer remaining tasks)
                if action == env.num_robots:
                    num_holds += 1
                    reward = -0.01  # Small penalty for holding

                    # Store experience and break (don't assign more tasks this timestep)
                    memory.rewards.append(reward)
                    memory.is_terminals.append(False)
                    iteration_reward += reward
                    break  # Stop assigning, advance time

                # Assign task to robot
                success = env.assign_task_to_robot(action, task)
                if success:
                    num_assignments += 1
                    reward = 1.0  # Immediate reward for assignment

                    # ✅ UPDATE STATE IMMEDIATELY after assignment
                    state_dict = env._get_state_dict()
                    # Next assignment will see updated robot availability!
                else:
                    reward = -1.0  # Penalty for failed assignment

                # Store experience
                iteration_reward += reward
                memory.rewards.append(reward)
                memory.is_terminals.append(False)

            # ===== SIMULATION TIME STEP =====
            # Advance simulation time (robots move, tasks complete, inventory depletes)
            state_dict, step_reward, done, info = env.step(Δt=timesteps_per_decision)

            # Accumulate step rewards (task completions, penalties, stockouts)
            if step_reward != 0:
                iteration_reward += step_reward

                # Credit step reward to most recent assignment decision
                # (if any assignments were made this timestep)
                if len(memory.rewards) > 0:
                    memory.rewards[-1] += step_reward

            # If we've reached max episode time, reset (but this is rare)
            if done:
                state_dict = env.reset()

        # Store final state for bootstrapping
        next_state_dict = state_dict

        # Update policy with GAE
        ppo.update(memory, next_state_dict)
        memory.clear_memory()

        # Iteration metrics
        iteration_time = time.time() - iteration_start_time
        iteration_rewards.append(iteration_reward)
        config_rewards.append(iteration_reward)  # Track for curriculum
        running_reward = 0.05 * iteration_reward + (1 - 0.05) * running_reward

        # ===== CURRICULUM SWITCHING LOGIC =====
        should_switch_config = False

        if use_curriculum and current_config_idx < len(curriculum_configs) - 1:
            if curriculum_schedule == 'fixed':
                # Switch every N iterations
                if iterations_on_current_config >= config_switch_interval:
                    should_switch_config = True

            elif curriculum_schedule == 'adaptive':
                # Switch when performance threshold met
                if len(config_rewards) >= 20:  # Need at least 20 iterations
                    recent_avg = np.mean(config_rewards[-20:])
                    if recent_avg >= performance_threshold:
                        should_switch_config = True

            elif curriculum_schedule == 'random':
                # Randomly switch (10% chance per iteration after 100 iters)
                if iterations_on_current_config >= 100 and np.random.random() < 0.1:
                    should_switch_config = True

        if should_switch_config:
            # Advance to next config
            current_config_idx += 1
            print(f"\n{'='*80}")
            print(f"CURRICULUM ADVANCE: Moving to config {current_config_idx + 1}/{len(curriculum_configs)}")
            print(f"  Previous config avg reward: {np.mean(config_rewards[-20:]):.2f}")
            print(f"{'='*80}\n")

            # Reset environment with new config
            env, current_config_desc = create_env_with_config(current_config_idx)
            state_dict = env.reset()

            # Reset curriculum tracking
            config_rewards = []
            iterations_on_current_config = 0

        # Logging
        if iteration % log_interval == 0:
            avg_reward = np.mean(iteration_rewards[-log_interval:])
            loss_info = ppo.get_last_loss_info()

            print(f"\nIteration {iteration}")
            if use_curriculum:
                print(f"  Config: {current_config_idx + 1}/{len(curriculum_configs)} - {current_config_desc}")
                print(f"  Iterations on config: {iterations_on_current_config}")
            print(f"  Iteration Reward: {iteration_reward:.2f}")
            print(f"  Avg Reward ({log_interval} iters): {avg_reward:.2f}")
            print(f"  Running Reward: {running_reward:.2f}")
            print(f"  Tasks Assigned: {num_assignments}")
            print(f"  Tasks Deferred (HOLD): {num_holds}")
            print(f"  Total Timesteps: {total_timesteps}")
            print(f"  Simulation Time: {env.current_time / 3600:.2f} hours")
            print(f"  Pending Tasks: {len(env.pending_tasks)}")
            print(f"  Completed Tasks: {len(env.completed_tasks)}")
            print(f"  Iteration Time: {iteration_time:.2f}s")

            if loss_info:
                print(f"\n  Loss Breakdown:")
                print(f"    Total: {loss_info.get('total_loss', 0):.4f}")
                print(f"    Actor: {loss_info.get('actor_loss', 0):.4f}")
                print(f"    Critic: {loss_info.get('critic_loss', 0):.4f}")
                print(f"    Entropy: {loss_info.get('entropy_loss', 0):.4f}")

                if use_debiasing:
                    print(f"    De-bias Total: {loss_info.get('debias_loss', 0):.4f}")
                    print(f"      - State Delta: {loss_info.get('state_delta', 0):.4f}")
                    print(f"      - Consistency: {loss_info.get('consistency', 0):.4f}")

            print("-" * 80)

        # Save model
        if iteration % save_interval == 0:
            save_path = checkpoint_dir / f"gapo_iter{iteration}.pth"
            ppo.save(str(save_path))
            print(f"✓ Model saved to {save_path}\n")

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Total Training Time: {total_time / 3600:.2f} hours")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Avg Iteration Reward: {np.mean(iteration_rewards):.2f}")
    print(f"Final Running Reward: {running_reward:.2f}")
    print(f"Final Simulation Time: {env.current_time / 3600:.2f} hours")
    print(f"Total Completed Tasks: {len(env.completed_tasks)}")
    print("=" * 80)

    # Save final model
    final_model_path = checkpoint_dir / "gapo_final.pth"
    ppo.save(str(final_model_path))
    print(f"\n✓ Final model saved to {final_model_path}")

    # Save training metrics
    metrics_path = checkpoint_dir / "training_metrics.npz"
    np.savez(
        metrics_path,
        iteration_rewards=np.array(iteration_rewards),
        running_reward=running_reward,
        total_timesteps=total_timesteps
    )
    print(f"✓ Training metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
